import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression  # 改为线性模型
from config import Config

dataset = pd.read_csv(Config['dataset_path'], sep="\t", header=None)

# 中文分词
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
print(input_sententce[0])

# 分词结果转词向量
cv = CountVectorizer()
cv.fit(input_sententce.values)
input_feature = cv.transform(input_sententce.values)

# 验证模型
# 使用逻辑回归（线性模型）
model = LogisticRegression(
    max_iter=1000,  # 增加迭代次数确保收敛
    random_state=42,
    solver='lbfgs'  # 适用于小到中型数据集
)

# 训练模型
model.fit(input_feature, dataset[1].values)

test_sententce = '我要去看一下郭德纲的小品'
test_sentententce = " ".join(jieba.lcut(test_sententce))  # 对测试句子进行分词
pred_sententce = model.predict(cv.transform([test_sententce]))
print(f"模型: Logistic Regression, 待预测的文本: {test_sententce}, 预测结果: {pred_sententce}")