##  MV-IUP-pytorch

##  Introduction

We designed the MV-IUP model, it contains three parts:  

1. We learn users' implicit preferences from multi-view user-content interaction.

2. We model and represent the basic information of users, historical texts of users and sentimental sentences.
3. We integrate three kinds of user information representation with emotional sentence representation.

## Environment Requirement

`pip install -r requirements.txt`

## Dataset

We use an implicit sentiment dataset D-implicit and a universal sentiment dataset D-general .

## An example to  run the model

Using stacked attention fusion method：

* Modify dataset path
Change `train_path、dev_path、test_path` in `stacked_attention.py`
* Multi-view user implicit preference vector file using corresponding dataset
Change `file1、file2、file3` in `stacked_attention.py` using `user1_embedding.pkl、user2_embedding.pkl、user3_embedding.pkl` we provided.
(*These three files are the vector representations learned from the graph neural network model DMV-GCN mentioned in this paper.*)
 * Run the model
 After making the above modifications, you can execute `python ../stacked_attention.py` through the command line to train the model.
