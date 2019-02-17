# Kaggle-Quora-Insincere-Questions-Classification
这是Kaggle 2018年9月份的一个比赛，比赛任务需要参赛者将Quora上不真诚的提问剔除出来(比如涉及国家分裂，种族歧视，或者那些发表自己观点而不是寻求有用答案的问题...)。这是一个短文本分类问题，将问题分为真诚与不真诚两类。
我从10月中旬开始参加，一直到1月份，最后拿到银牌(Top 5%)，下面是我在比赛中用到的模型以及一些大佬们的方法。
## 环境
## 预处理
## 我的模型
具体参看`modol.py`文件
大致结构如下：
![]
运行 `python run_cnn.py train`，可以开始训练。
## 测试
运行 `python run_cnn.py test` 在测试集上进行测试。
下面是大佬们的模型：
## 1st model
```
0.7*Glove + 0.3*Paragram
Embedding(300)
    Spatial Dropout
Bidirectional CuDNNLSTM
  1D Convolution(64)    Statistical features
Global Max Pooling 1D     Dense(64)
               concatenate
                Dense(128)
                Dropout
            Batch Normalization
                out
```
一个Bi-LSTM 后接一个一维的大小为1的卷积，然后做一个GlobalMaxPooling的池化操作，dropout层，同时将statistical features通过一个全连接层，将两者的结果串接在一起，送到后面的一个全连接层，然后是dropout层和BN层，最后将得到的结果做二分类。没有用到很复杂的结构。作者训练了10个模型，最后将这10个模型的平均值作为预测结果。
作者在预处理中没有将所有序列填充到相同长度，而是在训练中将每一个batch中的序列填充到相同长度，这样能大大减少训练时间，训练出更多的模型。
