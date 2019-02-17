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
一个Bi-LSTM 后接一个一维的大小为1的卷积，然后做一个GlobalMaxPooling的池化操作，dropout层，同时将statistical features通过一个全连接层，将两者的结果串接在一起，送到后面的一个全连接层，然后是dropout层和BN层，最后将得到的结果做二分类。作者训练了10个模型，最后将这10个模型的平均值作为预测结果。
作者的模型并不复杂，我想能拿到第一的原因至少有两点：
1.用10个模型做集成。作者在预处理中没有将所有序列填充到相同长度，而是在训练中将每一个batch中的序列填充到相同长度，这样能大大减少训练时间，训练出更多的模型。而大多数人由于时间的限制都只用了5个模型；
2.不同的预处理。在预处理中，没有将英文全部转为小写，并且通过检查单词的单复数，检查单词的小写形式等方式来找到更多的单词的词向量，对于OOV的单词，用一个随机的词向量表示；
3.精细的调参。在使用词向量时，对不同的词向量赋予不同的权重，加权平均得到最后的词向量，而很多人都使用简单的平均。

