import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import jieba, re

# 1. 数据加载与预处理
def prep(file):
    df = pd.read_csv(file)
    stop = {'的','了','在','是','我'}  # 中文停用词
    # 文本清洗+分词：保留中英文，去停用词
    df['c'] = df.text.apply(lambda x: ' '.join([w for w in jieba.cut(re.sub(r'[^\u4e00-\u9fa5a-zA-Z]',
             ' ',str(x).lower())) if w.strip() and w not in stop]))
    df.label, _ = pd.factorize(df.label)  # 标签数值化
    return df.c, df.label

# 2. 模型定义
class LSTM(nn.Module):
    def __init__(self, vocab, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, 16, padding_idx=0)
        self.lstm = nn.LSTM(16, 8, batch_first=True)
        self.fc = nn.Linear(8, num_classes)
    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

# 3. 主流程
X, y = prep(wenben.txt')  # 加载预处理
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
n_class = len(np.unique(y))

# 模型1
tfidf = TfidfVectorizer(max_features=500)
nb_acc = accuracy_score(yv, MultinomialNB().fit(tfidf.fit_transform(Xt), yt).predict(tfidf.transform(Xv)))
print(f"朴素贝叶斯准确率: {nb_acc:.4f}")

# 模型2
w2i = {'<pad>':0, '<unk>':1, **{w:i+2 for i,w in enumerate(set(' '.join(Xt).split()))}}  # 词表
# 文本转序列
seq = lambda t: [torch.tensor([w2i.get(w,1) for w in s.split()[:30]] + [0]*(30-len(s.split()[:30]))) for s in t]
train_data = list(zip(seq(Xt), yt))
# 训练
model = LSTM(len(w2i), n_class)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
for _ in range(2):  # 2轮训练
    for x,y in train_data:
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(x.unsqueeze(0)), torch.tensor([y])).backward()
        opt.step()
# 评估
lstm_acc = accuracy_score(yv, [model(x.unsqueeze(0)).argmax().item() for x in seq(Xv)])
print(f"LSTM准确率: {lstm_acc:.4f}")
