import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression # 线性模型模块
from sklearn.tree import DecisionTreeClassifier # 决策树模块
from sklearn.neighbors import KNeighborsClassifier # KNN 模型模块

# 读取数据并显示前五行
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.head(5))

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 将文本数据构建成向量
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

# 构建测试数据
test_querys = ["明天上午9九点提醒我带宝宝去体检",
             "帮我播放复仇者联盟的最新一部电影",
             "现在天气这么好，帮我播放一首适合的音乐",
             "想去潮州旅游，帮我看看那边现在的天气"]

test_sentences = pd.Series(test_querys).apply(lambda x: " ".join(jieba.lcut(x)))
print(test_sentences)
test_features = vector.transform(test_sentences.values)


model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

print("待预测的文本", test_querys)
print("KNN模型预测结果: ", model.predict(test_features))

model = LogisticRegression(max_iter=1000)
model.fit(input_feature, dataset[1].values)

print("待预测的文本", test_querys)
print("逻辑回归模型预测结果: ", model.predict(test_features))

model = DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)

print("待预测的文本", test_querys)
print("决策树模型预测结果: ", model.predict(test_features))
