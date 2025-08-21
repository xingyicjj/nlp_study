import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier  # knn模型
from sklearn.tree import DecisionTreeClassifier  # 决策树模块


def init_data():
    """
    初始化数据，包括读数据集，提取文本特征
    """
    dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
    # print(dataset.head(5))

    # 提取 文本的特征 tfidf， dataset[0]
    input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # sklearn对中文处理
    vector = CountVectorizer()  # 对文本进行提取特征 默认是使用标点符号分词
    vector.fit(input_sententce.values)
    input_feature = vector.transform(input_sententce.values)
    return [dataset, vector, input_feature]


def knn_model(X, y):
    """
    构建一个knn模型，学习提取的特征和标签的关系
    :param X: 特征
    :param y: 标签
    :return: 模型
    """
    model = KNeighborsClassifier()
    model.fit(X, y)
    return model


def tree_model(X, y):
    """
    构建一个决策树模型，学习提取的特征和标签的关系
    :param X: 特征
    :param y: 标签
    :return: 模型
    """
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


if __name__ == '__main__':
    dataset, vector, input_feature = init_data()
    # 对用户输入的文本进行预测结果
    test_query = input("请输入待预测的文本：")
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])
    print("待预测的文本：", test_query)
    print("KNN模型预测结果：", knn_model(input_feature, dataset[1].values).predict(test_feature))
    print("决策树模型预测结果：", tree_model(input_feature, dataset[1].values).predict(test_feature))
