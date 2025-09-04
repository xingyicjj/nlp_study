import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# 读取数据
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)

# 文本分词
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 切分数据
date_train, data_test, label_train, label_test = train_test_split(
    input_sentence.values, dataset[1].values, test_size=0.2, random_state=42, stratify=dataset[1].values)

# 拟合向量器
vector = CountVectorizer()
data_train_vec = vector.fit_transform(date_train)
data_test_vec = vector.transform(data_test)

# 使用 逻辑回归 对文本进行分类
model_lr = LogisticRegression()
model_lr.fit(data_train_vec, label_train)
print("逻辑回归模型验证集: ", classification_report(label_test, model_lr.predict(data_test_vec)))

# 使用 决策树 对文本进行分类
model_dt = DecisionTreeClassifier()
model_dt.fit(data_train_vec, label_train)
print("决策树模型验证集: ", classification_report(label_test, model_dt.predict(data_test_vec)))

# 使用 KNN 对文本进行分类
model_knn = KNeighborsClassifier()
model_knn.fit(data_train_vec, label_train)
print("KNN模型验证集: ", classification_report(label_test, model_knn.predict(data_test_vec)))

test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("逻辑回归模型预测结果: ", model_lr.predict(test_feature))
print("决策树模型预测结果: ", model_dt.predict(test_feature))
print("KNN模型预测结果: ", model_knn.predict(test_feature))
