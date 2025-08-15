from typing import List

import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # 文本特征提取
from sklearn.tree import DecisionTreeClassifier      # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier   # KNN

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, recall_score


data = pd.read_csv("./dataset.csv", sep="\t", header=None) # header=None 处理csv 数据没有title情况
# print(data.head(5))


# 提取 文本特征
# 对文本进行分词
data[0] = data[0].apply(lambda x: " ".join(jieba.lcut(x)))  # sklearn对中文处理



vector = CountVectorizer()
# 对文本进行特征提取
input_feature = vector.fit_transform(data[0])

# 划分数据集
train_data, test_data, train_label, test_label = train_test_split(input_feature, data[1], test_size=0.2, random_state=42)

# print(input_feature)


# 模型评估
def model_evaluate(model, test_data, test_label, model_name=None):
    print(f"============{model_name}===============")
    # 准确率
    accuracy = model.score(test_data, test_label)
    print(f"模型准确率: {accuracy:.4f}")
    
    # 预测概率
    test_prob = model.predict_proba(test_data)
    # test_prob = model.predict_proba(test_data)[:, 1]  # 获取正类的概率  predict_proba(test_data)[:, 1] 是二分类模型用法
    # print("模型AUC值: ", roc_auc_score(test_label, test_pred)) auc 值只适用于二分类问题，并且需要 predict_proba 方法
    # LogLoss
    loss = log_loss(test_label, test_prob)
    print(f"模型LogLoss值: {loss:.4f}")
    
    # F1值
    test_pred = model.predict(test_data)
    f1 = f1_score(test_label, test_pred, average="weighted")
    print(f"模型F1值: {f1:.4f}")
    
    # 召回率
    recall = recall_score(test_label, test_pred, average="weighted")
    print(f"模型召回率: {recall:.4f}")
    
    return accuracy, loss, f1, recall




## 使用 knn 算法对文本进行分类训练
for i in range(1, 10, 2):
    print(f"KNN n_neighbors {i}")
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(input_feature, data[1])
    # 模型评估
    model_evaluate(model, test_data, test_label, model_name="KNN")



## 使用 决策树 算法对文本进行分类训练
model = DecisionTreeClassifier()
model.fit(input_feature, data[1])
# 模型评估
model_evaluate(model, test_data, test_label, model_name="决策树")


## 使用 随机森林 算法对文本进行分类训练
model = RandomForestClassifier()
model.fit(input_feature, data[1])
# 模型评估
model_evaluate(model, test_data, test_label, model_name="随机森林")


## 使用 逻辑回归 算法对文本进行分类训练
model = LogisticRegression()
model.fit(input_feature, data[1])
# 模型评估
model_evaluate(model, test_data, test_label, model_name="逻辑回归")
