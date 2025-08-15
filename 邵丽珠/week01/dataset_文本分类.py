import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset.head(5))

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# print(input_sentence)

vector = CountVectorizer()
vector.fit(input_sentence.values)
# print("词汇表：", vector.get_feature_names_out())
input_feature = vector.transform(input_sentence.values)
# print("文档-词频矩阵：\n", input_feature.toarray())

# 方法1
print("-" * 20 + "方法1" + "-" * 20)
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
# print(model)

test = "播放一首周杰伦的歌"
test_sentence = " ".join(jieba.lcut(test))
test_feature = vector.transform([test_sentence])
print(f"待预测的文本：{test}")
print(f"KNN模型预测结果：{model.predict(test_feature)}")

# 方法2
print("-" * 20 + "方法2" + "-" * 20)
model = linear_model.LogisticRegression(max_iter=1000)
model.fit(input_feature, dataset[1])
# print(model)

test = "播放一首周杰伦的歌"
test_sentence = " ".join(jieba.lcut(test))
test_feature = vector.transform([test_sentence])
print(f"待预测的文本：{test}")
print(f"逻辑回归预测结果：{model.predict(test_feature)}")
