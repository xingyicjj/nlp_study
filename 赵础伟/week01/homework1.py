import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

data =pd.read_csv("dataset.csv", sep="\t", header=None)

#输入文本分词
data_sentence = data[0].apply(lambda x: " ".join(jieba.lcut(x)))

#使用计数向量化器提取输入文本特征
vectorizer = CountVectorizer()
vectorizer.fit(data_sentence.values)
input_feature = vectorizer.transform(data_sentence.values)

model = KNeighborsClassifier()
model.fit(input_feature,data[1].values)

test_string_sentence = " ".join(jieba.lcut("明天天气怎么样"))
test_result = model.predict(vectorizer.transform([test_string_sentence]))
print("KNN模型预测结果: ", test_result)



