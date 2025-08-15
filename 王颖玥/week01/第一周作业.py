import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer   # 文本转特征
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import linear_model

"""
实现了中文文本分类，训练集含有两列，第一列是句子，第二列是句子所体现的类型，比如视频播放、音乐播放、
天气预报等。预测输入的一句话体现了什么类型

采用了KNN、逻辑回归和决策树三种模型来实现文本分类
具体流程是，读取到文本，用CountVectorizer()来解析词频，但是它是以空格作为分隔符来统计的，中文不会
以空格作为分隔符，所以要提前将分割出来的词语以空格分开。然后要统计词汇表中出现的所有词汇，生成一个词
汇表，这个词汇表中含有词和这个词的索引，再将训练文本转换成特征矩阵，矩阵的每个值是词语出现的次数。之后，
用特征矩阵和标签分别对三种模型进行训练，预测并输出结果

多次测试得到的结果见test，从测试结果我发现，KNN的模型预测结果不如逻辑回归和决策树
"""

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset[1].value_counts())
# print(dataset[0].head(5))
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# input_sentence = dataset[0].apply(lambda x: jieba.lcut(x))
# print(input_sentence)

vector = CountVectorizer()
vector.fit(input_sentence.values)   # 拟合训练文本，学习词汇表
input_feature = vector.transform(input_sentence.values)  # 将文本转换为特征矩阵
# print(vector.vocabulary_)

while True:
    test_query = input("请输入你想预测的文本：")
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])  # 这个地方不能写成list，因为会形成一个单个字符的列表，比如输入的是["abc"]，得到的就是['a', 'b', 'c']

    # KNN模型
    model_knn = KNeighborsClassifier()
    model_knn.fit(input_feature, dataset[1].values)
    # print(model)

    print(f"KNN模型预测结果：{model_knn.predict(test_feature)}")

    # 逻辑回归模型
    model_logic = linear_model.LogisticRegression(max_iter=1000)
    model_logic.fit(input_feature, dataset[1].values)

    print(f"逻辑回归模型预测结果：{model_logic.predict(test_feature)}")

    # 决策树模型
    model_tree = tree.DecisionTreeClassifier()
    model_tree.fit(input_feature, dataset[1].values)

    print(f"决策树的预测结果：{model_tree.predict(test_feature)}")

    flag = input("是否要继续(y/n)：")
    if flag == "y":
        continue
    else:
        break
