import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from config import Config


dataset = pd.read_csv(Config['dataset_path'], sep="\t", header=None)
# print(dataset[0].__len__())
# 以下为中文分词，在词与词中间添加空格
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
print(input_sententce[0])

# 分词结果转词向量
cv = CountVectorizer()
'''
学习词汇表：分析所有文本，构建词汇表（词典）
统计词频：计算每个词在整个语料库中的出现频率
建立映射关系：为每个唯一单词分配一个整数索引
'''
cv.fit(input_sententce.values)
input_feature = cv.transform(input_sententce.values)



# 验证模型
for n in range(1,5):
    # 训练模型
    model = KNeighborsClassifier(n_neighbors=n)
    '''
    训练模型：使用特征向量和标签学习分类规则
    建立决策边界：在特征空间中划分不同类别区域
    存储模型参数：如 KNN 中的训练样本和距离度量方式
    '''
    model.fit(input_feature, dataset[1].values)

    test_sententce = '我要去看一下郭德纲的小品'
    test_sententce = " ".join(jieba.lcut(test_sententce))  # 对测试句子进行分词
    pred_sententce = model.predict(cv.transform([test_sententce]))
    print(f"模型: KNN(n_neighbors={n}), 待预测的文本: {test_sententce}, 预测结果: {pred_sententce}")
