import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# 导入数据
dataset = pd.read_csv("dataset.csv",sep='\t',header=None)
data,tag= dataset[0].apply(lambda x :" ".join(jieba.lcut(x))),dataset[1]
# 划分数据
train_data,test_data,train_tag,test_tag =train_test_split(data,tag,test_size=0.2,random_state=42)
#提取tfidf
vector = TfidfVectorizer()
train_input = vector.fit_transform(train_data.values)
test_input = vector.transform(test_data.values)
# 构建测试句子
test_query = "帮我订明早的机票"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
#建立KNN模型
knn =KNeighborsClassifier()
knn.fit(train_input,train_tag.values)
knn_pred = knn.predict(test_input)
print("knn 准确率:", accuracy_score(test_tag, knn_pred))
print("待预测的文本", test_query)
print("knn预测输出:",knn.predict(test_feature))
#建立logistic模型
logistic = LogisticRegression(max_iter=200)
logistic.fit(train_input,train_tag.values)
logistic_pred = logistic.predict(test_input)
print("logistic 准确率:", accuracy_score(test_tag, logistic_pred))
print("待预测的文本", test_query)
print("Logistic模型预测输出:",logistic.predict(test_feature))
#建立朴素贝叶斯模型
bayes =MultinomialNB(alpha=0.7)
bayes.fit(train_input,train_tag.values)
bayes_pred = bayes.predict(test_input)
print("朴素贝叶斯准确率:", accuracy_score(test_tag, bayes_pred))
print("待预测的文本", test_query)
print("朴素贝叶斯模型预测输出:",bayes.predict(test_feature))
