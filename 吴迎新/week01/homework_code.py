import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = TfidfVectorizer(tokenizer=str.split, token_pattern=None)
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
##第一个模型
model = LinearSVC()
model.fit(input_feature, dataset[1].values)
print(model)

test_query = "广州明天气温多少度"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("模型预测结果: ", model.predict(test_feature))

##第二种模型
model = LogisticRegression()
model.fit(input_feature, dataset[1].values)
print(model)

test_query = "广州明天气温多少度"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("模型预测结果: ", model.predict(test_feature))
