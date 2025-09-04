import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# 读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print("数据集前5行：")
print(dataset.head(5))

# 文本预处理：使用jieba进行中文分词
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
print("\n分词后的前3个句子：")
print(input_sentence.head(3))

# 特征提取：使用词袋模型
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)
print(f"\n特征矩阵形状: {input_feature.shape}")

# 使用随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(input_feature, dataset[1].values)
print(f"\n随机森林模型: {model}")

# 测试预测
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print(f"\n待预测的文本: {test_query}")
print(f"随机森林模型预测结果: {model.predict(test_feature)}")

# 显示预测概率
probabilities = model.predict_proba(test_feature)
print(f"各类别预测概率: {probabilities}")