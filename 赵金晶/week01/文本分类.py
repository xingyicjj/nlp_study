import jieba
import pandas as pd
from sklearn import linear_model # 线性模型模块
from sklearn.model_selection import train_test_split # 数据集划分
from sklearn import tree # 决策树模块
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 读取文件
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset.head(5))

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理、
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
X, y = input_feature, dataset[1]

# 训练集：调整模型的参数 （练习题、知道答案）
# 测试集：验证模型的精度 （摸底考试，不知道答案）
train_x, test_x, train_y, test_y = train_test_split(X, y)

# LogisticRegression
model = linear_model.LogisticRegression(max_iter=1000) # 模型初始化， 人工设置的参数叫做超参数， 模型参数可以从训练集学习到的
model.fit(train_x, train_y)
prediction = model.predict(test_x)
# print("待预测的文本", test_y)
# print("预测结果", prediction)

print(model)
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("模型预测结果: ", model.predict(test_feature))

# DecisionTreeClassifier
model = tree.DecisionTreeClassifier()
model.fit(train_x, train_y)
prediction = model.predict(test_x)
# print("待预测的文本", test_y)
# print("预测结果", prediction)

print(model)
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("模型预测结果: ", model.predict(test_feature))

# KNeighborsClassifier
model = model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_x, train_y)
prediction = model.predict(test_x)
# print("待预测的文本", test_y)
# print("预测结果", prediction)

print(model)
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("模型预测结果: ", model.predict(test_feature))
