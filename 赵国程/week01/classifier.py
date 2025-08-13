import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import tree

def create_vectorizer(name):
    if name == "count":
        return CountVectorizer()
    elif name == "hash":
        return HashingVectorizer(n_features=10000)
    elif name == "tfidf":
        return TfidfVectorizer()
    else:
        raise ValueError("不支持的vectorizer: {name}")

def create_model(name):
    if name == "logistic":
        return linear_model.LogisticRegression()
    elif name == "knn":
        return KNeighborsClassifier()
    elif name == "tree":
        return tree.DecisionTreeClassifier()
    else:
        raise ValueError("不支持的模型: {name}")

def main():
    dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
    input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
    for vector_name in ["count", "hash", "tfidf"]:
        vector = create_vectorizer(vector_name)
        input_feature = vector.fit_transform(input_sentence.values)
        for model_name in ["logistic", "knn", "tree"]:
            model = create_model(model_name)
            model.fit(input_feature, dataset[1].values)
            test_query = "今天周几"
            test_feature = vector.transform([" ".join(jieba.lcut(test_query))])
            print(f"vectorizer: {vector_name}, model: {model_name}, 预测结果: " + model.predict(test_feature)[0])

if __name__ == "__main__":
    main()

