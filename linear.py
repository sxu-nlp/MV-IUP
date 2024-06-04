# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap UER-py for classification.
"""
import torch
import math
import os
import sys
from sys import argv
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.model_builder import build_model
from uer.utils.optimizers import BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model
import pickle
import torch.nn.functional as F
from transformers import BertModel
from einops import rearrange


# 定义分类器
class BertClassifier(nn.Module):
    # 定义构造函数
    def __init__(self, args, model):  # 定义类的初始化函数，用户传入的参数
        super(BertClassifier, self).__init__()  # 调用父类nn.module的初始化方法，初始化必要的变量和参数
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.multi_head_attention = torch.nn.MultiheadAttention(768, 8)
        self.sigmoid = nn.Sigmoid()
        self.batchsize=args.batch_size
        # 双向lstm
        bidir = args.high_encoder == 'bi-lstm'
        self.direc = 2 if bidir else 1
        lstm_hidden_size = 200  # lstm_hidden_size的规格
        # 将LSTM的各项参数赋给rnn
        self.rnn = nn.LSTM(input_size=768,  # 输入size=768
                           hidden_size=lstm_hidden_size,  # 赋值hidden_size
                           num_layers=2,  # num_layers=2
                           batch_first=True,  # 布尔赋值
                           bidirectional=bidir
                           )

        self.labels_num = args.labels_num  # 将args参数赋给self参数
        self.pooling = args.pooling
        self.high_encoder = args.high_encoder

        self.head_num = 1  # head_num值为1
        if args.pooling == 'attention':  # 判断args.pooling取值
            # 线性回归
            self.attn_weight = nn.Linear(lstm_hidden_size * self.direc, 1)  # 输入与输出维度
        elif args.pooling == 'multi-head':  # else args.pooling取值
            self.emo_vec = self.load_emo_vec(args.emo_vec_path)  # 加载
            self.head_num = self.emo_vec.shape[0]
            self.bilinear = nn.Linear(lstm_hidden_size * self.direc, self.emo_vec.shape[-1], bias=False)
            self.emo_weight = nn.Linear(self.emo_vec.shape[-1], self.head_num, bias=False)  # 定义权重，无偏置
            self.emo_weight.weight.data = self.emo_vec
            self.attn_weight = nn.Sequential(
                self.bilinear,
                self.emo_weight,
            )
        self.transformer_enconder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_enconder = nn.TransformerEncoder(self.transformer_enconder_layer, num_layers=6)

        # self.W_s = torch.nn.Linear(27648, 768)
        # self.W_c = torch.nn.Linear(147456, 768)
        # self.linear_s = nn.Linear(768, 768)
        self.linear_s1 = nn.Linear(768, 768)
        self.linear_t = nn.Linear(768, 768)
        self.linear_t1 = nn.Linear(768, 768)
        self.linear_s2 = nn.Linear(768, 768)
        self.linear_t2 = nn.Linear(768, 768)


        self.linear_mpoa = nn.Linear(in_features=1200, out_features=1200)
        self.linear_c = nn.Linear(in_features=1200, out_features=1200)
        self.linear_r1 = nn.Linear(in_features=1200, out_features=1200)
        self.linear_r2 = nn.Linear(in_features=1200, out_features=1200)
        self.linear_r3 = nn.Linear(in_features=1200, out_features=1200)
        # self.linear_r4 = nn.Linear(in_features=1200, out_features=1200)
        self.W_c = torch.nn.Linear(1200, lstm_hidden_size)
        self.W_r = torch.nn.Linear(1200, lstm_hidden_size)
        output_size = lstm_hidden_size * self.direc * self.head_num  # 输出的规格
        self.output_layer_1 = nn.Linear(1200, lstm_hidden_size)
        self.output_layer_2 = nn.Linear(lstm_hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)  # softmax维度
        self.criterion = nn.NLLLoss()  # 损失函数（NLLLoss 函数输入 input 之前，需要对 input 进行 log_softmax 处理）

        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(1200))
        self.fc = nn.Linear(64, 1200)
        self.fr1=nn.Linear(64, 1200)
        self.fr2=nn.Linear(64, 1200)
        self.fr3=nn.Linear(64, 1200)
        self.fr4 = nn.Linear(64, 1200)
        self.fs = nn.Linear(768, 1200)
        self.linear_s = nn.Linear(in_features=1200, out_features=1200)
        self.W_s = torch.nn.Linear(1200, lstm_hidden_size)
        self.W_r2 = torch.nn.Linear(1200, lstm_hidden_size)
        self.W_r3 = torch.nn.Linear(1200, lstm_hidden_size)
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)
        self.alpha3 = nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)
        self.alpha4 = nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)


    def att(self, x, d):
        x=self.fc(x)
        M = d * self.tanh1(x)  #
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = x * alpha
        out = torch.sum(out, 1)
        return out

    # 定义attention函数
    def attention(self, H, mask):
        # mask (batch_size, seq_length)
        mask = (mask > 0).unsqueeze(-1).repeat(1, 1, self.head_num)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = self.attn_weight(H)  # 分数
        hidden_size = H.size(-1)
        scores /= math.sqrt(float(hidden_size))
        scores += mask
        probs = nn.Softmax(dim=-2)(scores)
        H = H.transpose(-1, -2)
        output = torch.bmm(H, probs)
        output = torch.reshape(output, (-1, hidden_size * self.head_num))
        return output

    def load_emo_vec(self, path):  # 定义load_emo_vec（）函数
        with open(path, 'r', encoding='utf-8') as f:
            emo_vec = json.load(f)  # 打开文件
            return torch.tensor(list(emo_vec.values())[:3])  # 返回张量 取值前三列

    def orthogonal_loss(self, input):  # 定义orthogonal_loss（）函数
        norm_query = input / torch.norm(input, dim=-1, keepdim=True)
        dot_res = torch.matmul(norm_query, norm_query.t())
        dot_res = torch.abs(dot_res)
        reg = torch.sum(dot_res) - torch.trace(dot_res)
        return reg

    def forward(self, src, label, mask, user_c, mask_c, user_r1, user_r2, user_r3, user_s, mask_s):  # 定义forward函数，实现该模块的前向过程
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size] label
            mask: [batch_size x seq_length]
        """
        # Embedding. BertEmbedding
        emb = self.embedding(src, mask)
        # Encoder.BertEncoder
        output = self.encoder(emb, mask)  # 通过bert的文本表示
        emb_s = self.embedding(user_s, mask_s)
        output_s = self.encoder(emb_s, mask_s)
        # att_s = self.multi_head_attention(output_s, output, output)
        # att_s1 = att_s[0]
        # z_s = self.sigmoid(self.linear_s1(att_s1))
        # z_output = self.sigmoid(self.linear_t(output))
        # z_att_s = self.sigmoid(z_s + z_output)
        # user_s = output_s + self.sigmoid(self.alpha) * z_att_s

        # att_s = self.multi_head_attention(user_s, output, output)
        # att_s2 = att_s[0]
        # z_s = self.sigmoid(self.linear_s1(att_s2))
        # z_output = self.sigmoid(self.linear_t1(output))
        # z_att_s = self.sigmoid(z_s + z_output)
        # user_s = user_s + self.sigmoid(self.alpha) * z_att_s

        att_s = self.multi_head_attention(output_s, output, output)  # 3
        att_s2 = att_s[0]
        z_s = self.sigmoid(self.linear_s2(att_s2))
        z_output = self.sigmoid(self.linear_t2(output))
        z_att_s = self.sigmoid(z_s + z_output)
        user_s = z_att_s * att_s2

        emb_c = self.embedding(user_c, mask_c)
        output_c = self.encoder(emb_c, mask_c)  # torch.Size([8, 192, 768])
        output_c = rearrange(output_c, 'B S E->S B E')
        User_c = self.transformer_enconder(output_c, src_key_padding_mask=mask_c)   #torch.Size([192, 8, 768])
        User_c = rearrange(User_c, 'S B E->B S E')

        # high_encoder:
        # 池化
        if self.high_encoder != 'none':
            output, _ = self.rnn(output)
            User_c, _ = self.rnn(User_c)
            user_s, _ = self.rnn(user_s)
        # # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)  # 取平均值mean
            # User_c = torch.mean(User_c, dim=1)  # 取平均值mean
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]  # 取最大值
        elif self.pooling == "last":
            output = output[:, -1, :]  # 取最小值
        elif self.pooling == "attention" or 'multi-head':  # 注意力机制or多头注意力
            output = self.attention(output, mask)
            User_c = self.attention(User_c, mask_c)
            User_s = self.attention(user_s, mask_s)
        else:
            output = output[:, 0, :]
        H_mpoa = torch.unsqueeze(output, dim=1)

        user_r = torch.stack((user_r1, user_r2, user_r3), dim=1)
        user_r = self.att(user_r, H_mpoa)
        # user_r=self.fr1(user_r1)

        # user_rs = torch.stack([user_r1, user_r2, user_r3], dim=1)
        # user_r = user_rs[:, 0, :]
        # pos_index = torch.where(user_r > 0)
        # pos_user = torch.max(user_rs, dim=1).values[pos_index]
        # user_r[pos_index] = pos_user

        H_mpoa = H_mpoa.reshape(self.batchsize,-1)
        # mpoa_input = torch.tanh(H_mpoa+User_c+User_s+user_r)
        # mpoa_input = torch.tanh(self.output_layer_1(H_mpoa) + self.W_c(User_c))
        mpoa_input = torch.tanh(self.output_layer_1(H_mpoa) + self.W_c(User_c) + self.W_r(user_r)+self.W_s(User_s))
        # mpoa_input = torch.tanh(self.output_layer_1(H_mpoa) +self.W_r(user_r)+ self.W_c(User_c))
        # mpoa_input = torch.tanh(self.output_layer_1(H_mpoa) +self.W_c(User_c)+self.W_s(User_s))
        # mpoa_input = torch.tanh(self.output_layer_1(H_mpoa) +  self.W_r(user_r)+self.W_s(User_s))
        logits = self.output_layer_2(mpoa_input)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))  # 损失函数
        if self.pooling == "multi-head":  # 多头机制
            loss = 0.9 * loss + 0.1 * self.orthogonal_loss(self.emo_weight.weight)
        return loss, logits


# 定义主函数
def main():
    root = 'logger/'  # 日志文件
    file_name = root + 'mdv-logger-11.17.txt'

    if not os.path.exists(file_name):  # 判断文件路径是否存在
        log_file = open(file_name, 'w', encoding='utf-8')
        log_file.close()

    log_file = open(file_name, 'a', encoding='utf-8')
    def logger(*args):
        str_list = " ".join([str(arg) for arg in args])
        print(str_list)
        log_file.write(str_list + '\n')
        log_file.flush()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 命令行参数解析包

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    # 训练验证和测试集
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--emo_vec_path", type=str, default="emo_vector.json")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    # batch_size 不易太大
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=36,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")

    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--high_encoder", choices=["bi-lstm", "lstm", "none"], default="bi-lstm")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last", "attention", "multi-head"],
                        default="multi-head",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to logger prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    args = parser.parse_args()
    logger(argv)

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:  # 读取文件
        for line_id, line in enumerate(f):  # try except处理异常
            # print(line)
            # print(line_id)
            try:
                line = line.strip().split("\t")  # strip()去除句子前后的空格，按照Tab键分割
                # print(line)
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])  # label标签
                labels_set.add(label)
            except:
                pass

    args.labels_num = len(labels_set)  # label长度

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build classification model.
    model = BertClassifier(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # if torch.cuda.device_count() > 1:
    #     logger("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    # 定义batch_loader函数
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, user_c_ids, mask_c_ids,
                    user_r1_ids, user_r2_ids, user_r3_ids, user_s_ids, mask_s_ids):  # batch_loader函数，四个参数
        instances_num = input_ids.size()[0]  # 给instances_num赋值
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]  # 给input_ids_batch赋值
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]  # 给label_ids_batch赋值
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]  # 给mask_ids_batch赋值
            user_c_ids_batch = user_c_ids[i * batch_size:(i + 1) * batch_size, :]  # 给user_c_ids_batch赋值
            mask_c_ids_batch = mask_c_ids[i * batch_size:(i + 1) * batch_size, :]
            user_r1_ids_batch = user_r1_ids[i * batch_size:(i + 1) * batch_size, :]
            user_r2_ids_batch = user_r2_ids[i * batch_size:(i + 1) * batch_size, :]
            user_r3_ids_batch = user_r3_ids[i * batch_size:(i + 1) * batch_size, :]
            user_s_ids_batch = user_s_ids[i * batch_size:(i + 1) * batch_size, :]  # 给user_s_ids_batch赋值
            mask_s_ids_batch = mask_s_ids[i * batch_size:(i + 1) * batch_size, :]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, user_c_ids_batch, mask_c_ids_batch, user_r1_ids_batch, user_r2_ids_batch, user_r3_ids_batch, user_s_ids_batch, mask_s_ids_batch
        if instances_num > instances_num // batch_size * batch_size:  # 判断之后赋值
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            user_c_ids_batch = user_c_ids[instances_num // batch_size * batch_size:]
            mask_c_ids_batch = mask_c_ids[instances_num // batch_size * batch_size:, :]
            user_r1_ids_batch = user_r1_ids[instances_num // batch_size * batch_size:]
            user_r2_ids_batch = user_r2_ids[instances_num // batch_size * batch_size:, :]
            user_r3_ids_batch = user_r3_ids[instances_num // batch_size * batch_size:, :]
            user_s_ids_batch = user_s_ids[instances_num // batch_size * batch_size:, :]
            mask_s_ids_batch = mask_s_ids[instances_num // batch_size * batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, user_c_ids_batch, mask_c_ids_batch, user_r1_ids_batch, user_r2_ids_batch, user_r3_ids_batch, user_s_ids_batch, mask_s_ids_batch

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)  # 建立tokenizer

    def User_s(a):
        tokens = [vocab.get(t) for t in tokenizer.tokenize(a)]
        tokens =tokens
        mask = [1] * len(tokens)
        if len(tokens) > 6:  # 判断tokens长度和args.seq_length长度大小
            tokens = tokens[:6]  # 截取args.seq_length长度
            mask = mask[:6]  # 截取args.seq_length长度
        while len(tokens) < 6:
            tokens.append(0)  # 直接添加
            mask.append(0)
        return tokens, mask

    def User_s1(a):
        tokens = [vocab.get(t) for t in tokenizer.tokenize(a)]
        tokens = [CLS_ID] + tokens
        mask = [1] * len(tokens)
        if len(tokens) > 24:  # 判断tokens长度和args.seq_length长度大小
            tokens = tokens[:24]  # 截取args.seq_length长度
            mask = mask[:24]  # 截取args.seq_length长度
        while len(tokens) < 24:
            tokens.append(0)  # 直接添加
            mask.append(0)
        return tokens, mask

    def User_c(document):
        tokens = [vocab.get(t) for t in tokenizer.tokenize(document)]
        tokens = [CLS_ID] + tokens
        mask = [1] * len(tokens)
        if len(tokens) > 192:  # 判断tokens长度和args.seq_length长度大小
            tokens = tokens[:192]  # 截取args.seq_length长度
            mask = mask[:192]  # 截取args.seq_length长度
        while len(tokens) < 192:
            tokens.append(0)  # 直接添加
            mask.append(0)
        return tokens, mask

    # Read dataset.
    # 定义读取数据集
    def read_dataset(path):  # 定义read_dataset函数
        dataset = []
        file1 = open("data/model_user_embedding1.pkl", "rb")
        user_r1_emb = pickle.load(file1)
        file2 = open("data/model_user_embedding2.pkl", "rb")
        user_r2_emb = pickle.load(file2)
        file3 = open("data/model_user_embedding3.pkl", "rb")
        user_r3_emb = pickle.load(file3)
        with open(path, mode="r", encoding="utf-8") as f:  # 打开文件
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                try:
                    line = line.strip().split('\t')  # 按 Tab键分割文本
                    if len(line) == 9:  # 如果line有3行
                        label = int(line[columns["label"]])  # 赋值label
                        text = line[columns["text"]]  # 赋值text
                        sex = line[columns["sex"]]
                        address = line[columns["address"]]
                        flag = line[columns["flag"]]
                        document = line[columns["document"]]
                        user_id = int(line[columns["user_id"]])
                        tokens = [vocab.get(t) for t in tokenizer.tokenize(text)]
                        tokens = [CLS_ID] + tokens
                        mask = [1] * len(tokens)
                        sex_tokens, mask_sex = User_s(sex)
                        address_tokens, mask_address = User_s(address)
                        flag_tokens, mask_flag = User_s1(flag)
                        user_s = flag_tokens + sex_tokens + address_tokens
                        mask_s = mask_flag + mask_sex + mask_address
                        user_c, mask_c = User_c(document)
                        i = int(user_id)
                        if i < 3112:
                            user_r1 = user_r1_emb[i]
                            user_r2 = user_r2_emb[i]
                            user_r3 = user_r3_emb[i]
                        else:
                            user_r1 = torch.tensor([0 for i in range(64)])
                            user_r2 = torch.tensor([0 for i in range(64)])
                            user_r3 = torch.tensor([0 for i in range(64)])
                        if len(tokens) > args.seq_length:  # 判断tokens长度和args.seq_length长度大小
                            tokens = tokens[:args.seq_length]  # 截取args.seq_length长度
                            mask = mask[:args.seq_length]  # 截取args.seq_length长度
                        while len(tokens) < args.seq_length:
                            tokens.append(0)  # 直接添加
                            mask.append(0)
                        # dataset.append((tokens, label, mask))
                        dataset.append((tokens, label, mask, user_c, mask_c, user_r1, user_r2, user_r3, user_s, mask_s))
                        # print(line_id)
                except:
                    pass
            print("dataset:", len(dataset))
        return dataset

    # Evaluation function.
    def evaluate(args, is_test):
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)
        # 输入、标签、mask的id
        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        user_c_ids = torch.LongTensor([sample[3] for sample in dataset])
        mask_c_ids = torch.LongTensor([sample[4] for sample in dataset])
        user_r1_ids = torch.tensor([sample[5].cpu().detach().numpy() for sample in dataset]).cuda()
        user_r2_ids = torch.tensor([sample[6].cpu().detach().numpy() for sample in dataset]).cuda()
        user_r3_ids = torch.tensor([sample[7].cpu().detach().numpy() for sample in dataset]).cuda()
        user_s_ids = torch.LongTensor([sample[8] for sample in dataset])
        mask_s_ids = torch.LongTensor([sample[9] for sample in dataset])

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            logger("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
        if not args.mean_reciprocal_rank:  # 如果不是MRR推荐算法机制
            for i, (
            input_ids_batch, label_ids_batch, mask_ids_batch, user_c_ids_batch,
            mask_c_ids_batch, user_r1_ids_batch, user_r2_ids_batch, user_r3_ids_batch, user_s_ids_batch, mask_s_ids_batch) in enumerate(
                    batch_loader(batch_size, input_ids, label_ids, mask_ids, user_c_ids,
                                 mask_c_ids, user_r1_ids, user_r2_ids, user_r3_ids, user_s_ids, mask_s_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                user_c_ids_batch = user_c_ids_batch.to(device)
                mask_c_ids_batch = mask_c_ids_batch.to(device)
                user_r1_ids_batch = user_r1_ids_batch.to(device)
                user_r2_ids_batch = user_r2_ids_batch.to(device)
                user_r3_ids_batch = user_r3_ids_batch.to(device)
                user_s_ids_batch = user_s_ids_batch.to(device)
                mask_s_ids_batch = mask_s_ids_batch.to(device)

                with torch.no_grad():  # 进行计算图的构建
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, user_c_ids_batch, mask_c_ids_batch,
                                        user_r1_ids_batch, user_r2_ids_batch, user_r3_ids_batch, user_s_ids_batch,
                                         mask_s_ids_batch)
                logits = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                correct += torch.sum(pred == gold).item()

            if is_test:
                logger("Confusion matrix:")
                logger(confusion)
                logger("Report precision, recall, and f1:")
            A=0
            for i in range(confusion.size()[0]):
                p = confusion[i, i].item() / confusion[i, :].sum().item()
                r = confusion[i, i].item() / confusion[:, i].sum().item()
                f1 = 2 * p * r / (p + r)
                if is_test:
                    logger("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
                A += f1
            logger('F : {:.3f}'.format(A / 3))
            logger("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
            return A
        else:  # 如果是MRR推荐算法机制
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch,
                    user_c_ids_batch, mask_c_ids_batch, user_r1_ids_batch, user_r2_ids_batch, user_r3_ids_batch, user_s_ids_batch, mask_s_ids_batch) in enumerate(
                batch_loader(batch_size, input_ids, label_ids, mask_ids, user_c_ids, mask_c_ids,
                            user_r1_ids, user_r2_ids, user_r3_ids, user_s_ids, mask_s_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                user_c_ids_batch = user_c_ids_batch.to(device)
                mask_c_ids_batch = mask_c_ids_batch.to(device)
                user_r1_ids_batch = user_r1_ids_batch.to(device)
                user_r2_ids_batch = user_r2_ids_batch.to(device)
                user_r3_ids_batch = user_r3_ids_batch.to(device)
                user_s_ids_batch = user_s_ids_batch.to(device)
                mask_s_ids_batch = mask_s_ids_batch.to(device)
                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch,
                                          user_c_ids_batch, mask_c_ids_batch,
                                         user_r1_ids_batch, user_r2_ids_batch, user_r3_ids_batch, user_s_ids_batch,
                                         mask_s_ids_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all = logits
                if i >= 1:
                    logits_all = torch.cat((logits_all, logits), 0)

            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][3]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid, j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid, j))

            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order = gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid = int(dataset[i][3])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            for i in range(len(score_list)):
                if len(label_order[i]) == 1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i], reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            logger(MRR)
            return MRR

    # Training phase.
    logger("Start training.")
    trainset = read_dataset(args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    input_ids = torch.LongTensor([example[0] for example in trainset])
    label_ids = torch.LongTensor([example[1] for example in trainset])
    mask_ids = torch.LongTensor([example[2] for example in trainset])
    user_c_ids = torch.LongTensor([example[3] for example in trainset])
    mask_c_ids = torch.LongTensor([example[4] for example in trainset])
    user_r1_ids = torch.tensor([example[5].cpu().detach().numpy() for example in trainset]).cuda()
    user_r2_ids = torch.tensor([example[6].cpu().detach().numpy() for example in trainset]).cuda()
    user_r3_ids = torch.tensor([example[7].cpu().detach().numpy() for example in trainset]).cuda()
    user_s_ids = torch.LongTensor([example[8] for example in trainset])
    mask_s_ids = torch.LongTensor([example[9] for example in trainset])
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    logger("Batch size: ", batch_size)
    logger("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0
    up_epoch = 0

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, user_c_ids_batch,
                mask_c_ids_batch, user_r1_ids_batch, user_r2_ids_batch, user_r3_ids_batch, user_s_ids_batch, mask_s_ids_batch) in enumerate(
                batch_loader(batch_size, input_ids, label_ids, mask_ids, user_c_ids, mask_c_ids,
                             user_r1_ids, user_r2_ids, user_r3_ids, user_s_ids, mask_s_ids)):  # 循环
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            user_c_ids_batch = user_c_ids_batch.to(device)
            mask_c_ids_batch = mask_c_ids_batch.to(device)
            user_r1_ids_batch = user_r1_ids_batch.to(device)
            user_r2_ids_batch = user_r2_ids_batch.to(device)
            user_r3_ids_batch = user_r3_ids_batch.to(device)
            user_s_ids_batch = user_s_ids_batch.to(device)
            mask_s_ids_batch = mask_s_ids_batch.to(device)

            loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, user_c_ids_batch,mask_c_ids_batch,user_r1_ids_batch, user_r2_ids_batch, user_r3_ids_batch,user_s_ids_batch,mask_s_ids_batch)
            # print("loss",loss)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                logger("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,total_loss / args.report_steps))
                total_loss = 0.

            loss.backward()
            optimizer.step()
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
            up_epoch = epoch
        else:
            if epoch - up_epoch >= 5:
                break
            continue

    # Evaluation phase.
    if args.test_path is not None:
        logger("Test set evaluation.")
        model = load_model(model, args.output_model_path)
        evaluate(args, True)
    log_file.close()


if __name__ == "__main__":
    main()
