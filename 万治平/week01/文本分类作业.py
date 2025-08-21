import pandas as pd
# 导入jieba分词
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# KNN模型
from sklearn.neighbors import KNeighborsClassifier
# 逻辑回滚模型
from sklearn.linear_model import LogisticRegression
# 支持向量机模型
from sklearn.svm import SVC
# 决策数模型
from sklearn.tree import DecisionTreeClassifier
# 随机森林模型
from sklearn.ensemble import RandomForestClassifier

import os

here = os.path.dirname(os.path.abspath(__file__))

# 读取当前目录下的dataset.csv文件, 指定分割符为\t, 不指定header
ds = pd.read_csv(os.path.join(here, 'dataset.csv'), sep='\t', header=None)
print(f"dataset文件内容：\n {ds}")

# ds[0]的数据的每一行使用jieba进行分词
fc_rs = ds[0].apply(lambda fc_row_str: " ".join(jieba.lcut(fc_row_str)))
print(f"文本分词后的结果：\n {fc_rs}")

# 对文本进行提取特征，使用CountVectorizer，使用TfidfVectorizer也可
cv = CountVectorizer()
cv.fit(fc_rs.values)

cv_rs = cv.transform(fc_rs.values)
print(f"文本特征提取后的结果：\n {type(cv_rs)}")

# 将特征数据和标签数据放入KNN模型进行训练
knn_model = KNeighborsClassifier()
knn_model.fit(cv_rs, ds[1].values)


# 将特征数据和标签数据放入逻辑回归模型进行训练
lr_model = LogisticRegression()
lr_model.fit(cv_rs, ds[1].values)

# 将特征数据和标签数据放入支持向量机模型进行训练
svc_model = SVC()
svc_model.fit(cv_rs, ds[1].values)

# 将特征数据和标签数据放入决策数模型进行训练
dt_model = DecisionTreeClassifier()
dt_model.fit(cv_rs, ds[1].values)

# 将特征数据和标签数据放入随机森林模型进行训练
rf_model = RandomForestClassifier()
rf_model.fit(cv_rs, ds[1].values)

# 使用提问的问题，让模型把语句做分类
query = "给我播放海阔天空"
# 使用jieba进行分词
jb_rs =  " ".join(jieba.lcut(query))
# 将分词后的结果转换为特征
query_feature = cv.transform([jb_rs])

# 使用Knn模型进行预测
knn_pred_rs = knn_model.predict(query_feature)
print(f"KNN模型预测后的结果：\n {knn_pred_rs}")
# 使用逻辑回归模型进行预测
lr_pred_rs = lr_model.predict(query_feature)
print(f"逻辑回归模型预测后的结果：\n {lr_pred_rs}")
# 使用支持向量机模型进行预测
svc_pred_rs = svc_model.predict(query_feature)
print(f"支持向量机模型预测后的结果：\n {svc_pred_rs}")
# 使用决策数模型进行预测
dt_pred_rs = dt_model.predict(query_feature)
print(f"决策数模型预测后的结果：\n {dt_pred_rs}")
# 使用随机森林模型进行预测
rf_pred_rs = rf_model.predict(query_feature)
print(f"随机森林模型预测后的结果：\n {rf_pred_rs}")