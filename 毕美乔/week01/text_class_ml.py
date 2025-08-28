from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from word2vec import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer


# 读入数据
df = pd.read_csv("dataset.csv",  sep='\t', header=None)
df.columns = ['Input Text', 'Label']
# 对输入文本进行预处理
df['Processed Text'] = df['Input Text'].apply(preprocess_text)

# 3. 使用TF-IDF生成词向量
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(df['Processed Text'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['Label'], test_size=0.3, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
# 评估模型
print("准确率:", accuracy_score(y_test, y_pred))


# 支持向量机 (SVM)：
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 评估模型
print("准确率:", accuracy_score(y_test, y_pred))

# 随机森林：
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 评估模型
print("准确率:", accuracy_score(y_test, y_pred))
