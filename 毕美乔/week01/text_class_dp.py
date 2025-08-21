import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import jieba  # 如果是中文分词

# 读取数据
df = pd.read_csv("dataset.csv", sep='\t', header=None)
df.columns = ['Input Text', 'Label']

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('/home/bmq/huggingface')
model = BertForSequenceClassification.from_pretrained(
    '/home/bmq/huggingface',
    num_labels=len(df['Label'].unique())
)

# 中文分词
def preprocess_text(text):
    return " ".join(jieba.cut(text))

# 数据预处理
def preprocess_data(df, tokenizer, max_length=128):
    # 标签映射
    unique_labels = df['Label'].unique().tolist()
    text2label = {text: idx for idx, text in enumerate(unique_labels)}
    label2text = {idx: text for text, idx in text2label.items()}

    labels = df['Label'].map(text2label).values
    labels = torch.tensor(labels, dtype=torch.long)

    # 编码
    encodings = tokenizer(
        df['Input Text'].apply(preprocess_text).tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings, labels, text2label, label2text

encodings, labels, text2label, label2text = preprocess_data(df, tokenizer)

# Dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)

dataset = TextDataset(encodings, labels)
print(dataset[0])

# 划分训练/验证集索引
train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=1e-5)

# 训练
def train(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch[0]['input_ids'].to(device)
        attention_mask = batch[0]['attention_mask'].to(device)
        labels = batch[1].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_dataloader)

# 验证
def evaluate(model, val_dataloader, device):
    model.eval()
    total_loss, correct_predictions, total_predictions = 0, 0, 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            labels = batch[1].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    return total_loss / len(val_dataloader), correct_predictions / total_predictions

# 训练循环
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train(model, train_dataloader, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")
    val_loss, val_accuracy = evaluate(model, val_dataloader, device)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
