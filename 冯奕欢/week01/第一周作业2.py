import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 使用pandas加载数据 以\t为分隔符 无标题
data = pd.read_csv('dataset.csv', sep='\t', header=None)
print(data.head())
print(type(data))

# 第一列为参数
# 1.文本进行分词
# jieba分词后是一个列表 使用空格连接为新字符串 目的是满足下面词频统计
data_0 = data[0].apply(lambda text: " ".join(jieba.lcut(text)))
print(data_0.head())
print(type(data_0))

# 2.词频统计作为文本特征
countVectorizer = CountVectorizer()
# 构建词汇表
# countVectorizer.fit(data_0)
# 测试数据转换词频矩阵
# X = countVectorizer.transform(data_0)
# 或者一步到位 = fit + transform
X = countVectorizer.fit_transform(data_0)
print(X.toarray())
print(type(X))
print(X.shape)

# 测试数据标签
y = data[1]
print(y.head())
print(type(y))
print(y.shape)

# 拆分训练集和测试集
train_x, test_x, train_y, text_y = train_test_split(
    X, y, test_size=0.01, random_state=100
)

# 使用逻辑回归模型
logistic_model = LogisticRegression()
logistic_model.fit(train_x, train_y)
predict_y = logistic_model.predict(test_x)
score = accuracy_score(text_y, predict_y)
print(f'逻辑回归模型分数: {score}')

# 使用KNN模型 效果一般
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(train_x, train_y)
predict_y = knn_model.predict(test_x)
score = accuracy_score(text_y, predict_y)
print(f'KNN模型分数: {score}')

# 使用朴素贝叶斯模型（多项式比较适合离散特征）
nb_model = MultinomialNB()
nb_model.fit(train_x, train_y)
predict_y = nb_model.predict(test_x)
score = accuracy_score(text_y, predict_y)
print(f'朴素贝叶斯模型分数: {score}')


# 测试新数据
content_1 = '请帮我导航到北京天安门'
content_2 = '播放周杰伦的晴天歌曲'
content_3 = '小爱小爱，帮我打开电视机'
content_4 = '我想看成龙的电影'


def predict_by_model(content, model):
    """
    预测内容的类型
    :param content: 内容
    :param model: 模型
    :return: 分类结果
    """
    new_data_x = [" ".join(jieba.lcut(content))]
    # print(new_data_x)
    # 这里调用转换词矩阵的方法就行 不能再次fit
    new_x = countVectorizer.transform(new_data_x)
    # print(new_x.toarray())
    predict_new_y = model.predict(new_x)
    # print(predict_new_y)
    if len(predict_new_y) > 0:
        print(f'{content} -> {predict_new_y[0]}')
        return predict_new_y[0]
    else:
        print(f'{content} -> None')
        return None


# 使用逻辑回归模型测试新数据
print("使用逻辑回归测试新数据：")
predict_by_model(content_1, logistic_model)
predict_by_model(content_2, logistic_model)
predict_by_model(content_3, logistic_model)
predict_by_model(content_4, logistic_model)

# 使用朴素贝叶斯模型测试新数据
print("使用朴素贝叶斯测试新数据：")
predict_by_model(content_1, nb_model)
predict_by_model(content_2, nb_model)
predict_by_model(content_3, nb_model)
predict_by_model(content_4, nb_model)
