import pandas as pd
import jieba
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree, linear_model, neighbors
from sklearn.naive_bayes import GaussianNB

dataset: DataFrame = pd.read_csv('../dataset.csv', sep='\t', header=None)
input_sentense = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
print(input_sentense)
print('-' * 50)

vector = CountVectorizer()
vector.fit(input_sentense.values)
input_feature = vector.transform(input_sentense.values)
# print(input_feature)

# todo 模型一:决策树
model = tree.DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)
print('决策树模型：', model)
test_query = '我喜欢周杰伦的歌曲'
# test_query = '河南新闻广播找一下啊是新闻台'
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print('待测试文本：', test_query)
print('决策树预测结果：', model.predict(test_feature))
print('-' * 50)


# todo 模型二：逻辑回归
model = linear_model.LogisticRegression()
model.fit(input_feature, dataset[1].values)
print('逻辑回归模型：', model)
test_data = "上海明天是晴天吗"
test_data_sentence = " ".join(jieba.lcut(test_data))
test_data_feature = vector.transform([test_data_sentence])
print('测试文本：', test_data)
print('线性逻辑回归模型预测结果： ', model.predict(test_data_feature))
print('-' * 50)


# todo 模型三： 高斯朴素贝叶斯
model = GaussianNB()
model.fit(input_feature.toarray(), dataset[1].values)
print('贝叶斯模型：', model)
test_gnb_data = "打开琅琊榜最后一集"
test_gnb_sentence = " ".join(jieba.lcut(test_gnb_data))
test_gnb_feature = vector.transform([test_gnb_sentence])
print('测试文本：', test_gnb_sentence)
print('贝叶斯模型预测结果：', model.predict(test_gnb_feature.toarray()))
print('-' * 50)


# todo 模型四：KNN模型
model = neighbors.KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
print('KNN模型： ', model)
test_knn_data = "播放周星驰的电影"
test_knn_sentence = " ".join(jieba.lcut(test_knn_data))
test_knn_feature = vector.transform([test_knn_sentence])
print('测试文本 ', test_knn_data)
print('KNN模型预测结果：', model.predict(test_knn_feature))

