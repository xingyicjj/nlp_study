import jieba
import pandas as pd
import sklearn
from sklearn import linear_model # 线性模型模块
from sklearn import tree # 决策树模块
from sklearn import datasets # 加载数据集
from sklearn.model_selection import train_test_split # 数据集划分
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 获取数据
db = pd.read_csv("dataset.csv", sep="\t", header=None)

# 数据处理
data = db[0].apply(lambda x: " ".join(jieba.lcut(x))) 

# 训练模型
print('开始训练统计模型')
vector = CountVectorizer()
vector.fit(data.values)
input_feature = vector.transform(data.values)
model = KNeighborsClassifier()
model.fit(input_feature, db[1].values)
print('模型训练完毕')

# 测试
test = input("请输入一段内容可进行测试：")
test_sentence = " ".join(jieba.lcut(test))
test_feature = vector.transform([test_sentence])
print(test,'属于',model.predict(test_feature)[0],'类型')

# /////////////////////////////////////////////////////////////////////////

# 获取数据
data = datasets.load_iris()
X, y = data.data, data.target

# 拆分数据train和test
train_x, test_x, train_y, test_y = train_test_split(X, y)

# 训练train数据的模型
print('开始训练线性模型')
model = linear_model.LogisticRegression(max_iter=1000)
model.fit(train_x, train_y)
print('模型训练完毕')

prediction = model.predict(test_x)
print("逻辑回归的预测结果", ((test_y == prediction).sum()/ len(test_x))*100, '%')