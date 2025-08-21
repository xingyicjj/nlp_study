from ModelFactory import ModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":

    dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
    input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
    vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
    vector.fit(input_sententce.values)
    input_feature = vector.transform(input_sententce.values)
    train_x, test_x, train_y, test_y = train_test_split(input_feature, dataset[1].values, random_state=2025) # 数据切分 25% 样本划分为测试集

    models = [
        ModelFactory.create_model('svc', kernel='linear', C=1),
        ModelFactory.create_model('knn', n_neighbors=5),
        ModelFactory.create_model('forest', n_estimators=500)
    ]

    for model in models:
        model.fit(train_x, train_y)
        preds = model.predict(test_x)
        acc = accuracy_score(test_y, preds)
        print(f"{type(model).__name__} accuracy: {acc:.4f}")