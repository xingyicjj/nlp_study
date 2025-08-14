import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import tree

dataset = pd.read_csv ("dataset.csv", sep='\t', header=None)

input_sentence = dataset[0].apply (lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

test_query = "天气预报"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

for k in [1,3,5,7,9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(input_feature, dataset[1].values)
    print("待预测的文本", test_query)
    print(f"{k}-KNN的预测结果", model.predict(test_feature))

model = linear_model.LogisticRegression (max_iter=1000)
model.fit(input_feature, dataset[1].values)
prediction = model.predict(test_feature)
print ("-" * 20)
print("逻辑回归的预测结果", prediction)


model = tree.DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)
prediction = model.predict(test_feature)
print ("-" * 20)
print("决策树的预测结果", prediction)
