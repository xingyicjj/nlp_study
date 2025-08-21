import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("llm_v\week1\documents\Week01\Week01\work\dataset.csv", sep='\t', header=None)
print(dataset[1].value_counts())

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# input_sentence = dataset[0].apply(lambda x: jieba.lcut(x))
# print(input_senentce)
# print(input_sentence.values)
vector = CountVectorizer()
vector.fit(input_sentence.values)
# print(vector.get_feature_names_out())
input_futures = vector.transform(input_sentence.values)

model1 = KNeighborsClassifier()
model1.fit(input_futures, dataset[1].values)

model2 = RandomForestClassifier()
model2.fit(input_futures, dataset[1].values)

test_query = "今天天气怎么样"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print(model1.predict(test_feature))
print(model2.predict(test_feature))
