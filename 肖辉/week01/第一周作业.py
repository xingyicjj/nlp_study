import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. 加载数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
dataset.columns = ["text", "label"]  # 添加列名便于理解

# 2. 中文分词处理
dataset["segmented"] = dataset["text"].apply(lambda x: " ".join(jieba.lcut(x)))

# 3. 使用TF-IDF提取文本特征
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset["segmented"])
y = dataset["label"]

# 4. 训练朴素贝叶斯模型
model = MultinomialNB()
model.fit(X, y)

# 5. 预测新文本
test_query = "帮我买点止泻的药"
test_seg = " ".join(jieba.lcut(test_query))
test_vec = vectorizer.transform([test_seg])
prediction = model.predict(test_vec)[0]

print(f"文本: '{test_query}'")  # 文本: '帮我买点止泻的药'
print(f"朴素贝叶斯预测结果: {prediction}")  # 朴素贝叶斯预测结果: FilmTele-Play

# 方法二
# 2. 创建处理管道
pipeline = make_pipeline(
    TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None),  # 直接使用jieba分词
    StandardScaler(with_mean=False),  # SVM需要特征缩放
    SVC(kernel="linear")  # 线性核SVM
)

# 3. 训练模型
pipeline.fit(dataset["text"], dataset["label"])

# 4. 预测新文本
test_query = "帮我买点止泻的药"
prediction = pipeline.predict([test_query])[0]

print(f"文本: '{test_query}'")  # 文本: '帮我买点止泻的药'
print(f"SVM预测结果: {prediction}")  # SVM预测结果: Alarm-Update

