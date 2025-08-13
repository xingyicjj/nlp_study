import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import RadiusNeighborsClassifier


data = pd.read_csv("dataset.csv", sep="\t", header=None)

data_sentence = data[0].apply(lambda x: " ".join(jieba.lcut(x)))

tfidVector = TfidfVectorizer()
input_feature = tfidVector.fit_transform(data_sentence)


model = RadiusNeighborsClassifier()
model.fit(input_feature,data[1].values)

test_string_sentence = " ".join(jieba.lcut("我想听相声"))
test_result1 = model.predict(tfidVector.transform([test_string_sentence]))
print("RN模型预测结果:",test_result1)