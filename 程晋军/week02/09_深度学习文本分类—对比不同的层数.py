#作业：1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 加载数据集，数据集包含两列：文本和标签，使用制表符分隔
dataset = pd.read_csv(r"H:\BaiduNetdiskDownload\第一周-课程介绍及大模型基础\课件\Week01\Week01\dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()  # 提取所有文本数据
string_labels = dataset[1].tolist()  # 提取所有标签数据
# '''| 特点         | dataset[0].tolist()       | iterrows()                |
# | ------------ | ------------------------- | ------------------------- |
# | 速度         | 快                        | 慢                        |
# | 用法         | 一次性取出整列数据        | 一行一行处理              |
# | 适合场景     | 只想取出某一列的所有数据  | 需要同时处理多列，并且逐行操作 |'''
# 创建标签到索引的映射，将字符串标签转换为数字索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}#这行代码是创建一个标签到数字的映射字典，给每个独一无二的标签标号，比如1000句话，15个标签
# 解释：这行代码是创建一个标签到数字的映射字典。让我用简单的方式解释：
# 这行代码是创建一个**标签到数字的映射字典**。让我用简单的方式解释：
# label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 1. `string_labels` - 这是我们之前提取的所有标签列表，比如：
# string_labels = ["导航", "天气", "音乐", "电话", "导航", "天气", ...]
# 2. `set(string_labels)` - 去除重复的标签，只保留唯一的标签：
# set(string_labels) = {"导航", "天气", "音乐", "电话"}
# 3. `enumerate(...)` - 给每个唯一标签分配一个数字编号：
# enumerate({"导航", "天气", "音乐", "电话"}) 会产生:（变成了元组）
# (0, "导航")
# (1, "天气")
# (2, "音乐")
# (3, "电话")
# 4. `{label: i for i, label in ...}` - 创建一个字典，把标签作为键，数字作为值：
# label_to_index = {
#    "导航": 0,
#    "天气": 1,
#    "音乐": 2,
#    "电话": 3
# }
# 5.for i, label in enumerate(...) 中：列表推导式（List Comprehension）
# i 是元组的第一个元素（数字）
# label 是元组的第二个元素（文字标签）
# # 假设我们有这些数据
# string_labels = ["导航", "天气", "音乐", "导航"]
# set(string_labels) = {"导航", "天气", "音乐"}
# enumerate(...) 产生:
#   (0, "导航")
#   (1, "天气")
#   (2, "音乐")
# for i, label in ...:
#   第一次循环: i=0, label="导航"
#   第二次循环: i=1, label="天气"
#   第三次循环: i=2, label="音乐"
# label: i 创建字典:
#   "导航": 0
#   "天气": 1
#   "音乐": 2'''
#enumerate(...) - 给每个唯一标签分配一个数字编号，变成了；set(string_labels) - 去除重复的标签，只保留唯一的标签；
numerical_labels = [label_to_index[label] for label in string_labels]  # 将所有的字符串标签转换为数字标签，比如1000句话，1000都按照上面15个标记
#https://fcnbisyf4ls8.feishu.cn/wiki/KSQRwSGtNimS5ckbc09co7M3nWe?from=from_copylink
# 构建字符级词汇表，将每个字符映射到一个唯一的索引（数量等于词汇表）
char_to_index = {'<pad>': 0}  # '<pad>'用于填充，索引为0
for text in texts:
    for char in text:#遍历文本中的每个字符，在Python中，字符串可以直接用for循环遍历每个字符。
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)#在Python中，如果给字典中一个不存在的键赋值，会自动添加这个新的键值对。不会为重复的字分配新的序号。
#https://fcnbisyf4ls8.feishu.cn/wiki/N1WOwLeAZisfOwkMYPmcdqlbnHA?from=from_copylink
# '''char_to_index = {'<pad>': 0}
# 这行代码是创建一个字符到数字的映射字典，并初始化了一个特殊字符。让我用简单的方式解释：
# char_to_index = {'<pad>': 0}  # '<pad>'用于填充，索引为0
# 这行代码创建了一个字典，用来把字符转换成数字：
# - 键（key）：字符
# - 值（value）：对应的数字编号
# 目前字典里只有一个元素：`'<pad>': 0`
# ### 什么是 '<pad>'？
# `'<pad>'` 是一个特殊标记，在自然语言处理中叫做"填充符"（padding）。
# #### 生活中的例子
# 想象你在整理一批不同长度的木条，要把它们放进标准大小的盒子里：
# 长木条：███████████████ (15个单位)
# 中木条：███████ (7个单位)
# 短木条：████ (4个单位)
# 但盒子必须装10个单位长度，怎么办？
# 长木条：██████████ (截断，只保留10个)
# 中木条：███████□□□ (后面加3个填充物)
# 短木条：████□□□□□□ (后面加6个填充物)
# 这里的"□"就是填充符 `<pad>`，让所有木条看起来长度一致。
# ### 为什么需要这样做？
# 在深度学习中，神经网络需要固定大小的输入。但我们的文本长度不一样：
# - "播放音乐" → 4个字符
# - "帮我导航到北京" → 6个字符
# - "查询明天北京的天气" → 8个字符
# 为了让它们长度一致，我们需要：
# 1. 太长的截断
# 2. 太短的在后面补 `<pad>` 字符
# ### 为什么索引是0？
# 在编程中，我们通常从0开始编号。把 `<pad>` 设为0是因为：
# 1. 它是最常用的特殊符号
# 2. 在后续处理中，0通常表示"无意义"或"填充"
# 3. 方便程序识别哪些是真实字符，哪些是填充字符
# ### 后续会发生什么？
# 在接下来的代码中，会遍历所有文本，把出现的字符都加入这个字典：
# for text in texts:
#     for char in text:
#         if char not in char_to_index:
#             char_to_index[char] = len(char_to_index)
# 最终字典可能变成：
# char_to_index = {
#     '<pad>': 0,   # 填充符
#     '帮': 1,
#     '我': 2,
#     '导': 3,
#     '航': 4,
#     '到': 5,
#     # ... 更多字符
# }
index_to_char = {i: char for char, i in char_to_index.items()}
# '''这是一个字典推导式，用来创建一个反向字典。等价于：
# index_to_char = {}
# for char, i in char_to_index.items():
#     index_to_char[i] = char
# ################################################
#    比如：char_to_index = {'<pad>': 0, '播': 1, '放': 2, '音': 3, '乐': 4}
#    char_to_index.items() 会产生:
#    ('<pad>', 0)
#    ('播', 1)
#    ('放', 2)
#    ('音', 3)
#    ('乐', 4)
#    为什么要创建反向字典？
# 在深度学习中，有时我们需要：
# 正向查找：把字符转换成数字（训练时用）
# 反向查找：把数字转换回字符（预测结果解读时用）
# # 原字典 (字符→数字)
# char_to_index = {'<pad>': 0, '播': 1, '放': 2, '音': 3, '乐': 4}
# # 新字典 (数字→字符)
# index_to_char = {0: '<pad>', 1: '播', 2: '放', 3: '音', 4: '乐'}
# ################################################
#  .items() 的基本作用
# `.items()` 方法会返回字典中所有的键值对，以一种特殊的形式，让我们可以遍历它们。
# ### 生活中的例子
# 想象你有一本班级成绩册：
# ```
# 成绩册 = {
#     "小明": 95,
#     "小红": 87,
#     "小刚": 92
# }
# 使用 `.items()` 就像是把这本成绩册的每一页都展示出来：
# 成绩册.items() 会产生:
# ("小明", 95)
# ("小红", 87)
# ("小刚", 92)
# ### 在你的代码中
# char_to_index = {'<pad>': 0, '播': 1, '放': 2}
# char_to_index.items() 会产生:
# ('<pad>', 0)
# ('播', 1)
# ('放', 2)
# ### 和其他方法的对比
# 字典 = {'a': 1, 'b': 2, 'c': 3}
# 字典.keys()    # 只获取键: 'a', 'b', 'c'
# 字典.values()  # 只获取值: 1, 2, 3
# 字典.items()   # 获取键值对: ('a', 1), ('b', 2), ('c', 3)
vocab_size = len(char_to_index)  # 词汇表大小

max_len = 40  # 设置文本最大长度，用于统一输入维度

class CharBoWDataset(Dataset):
#https://fcnbisyf4ls8.feishu.cn/wiki/KiAwwNC8fi7upXkniHBcWbv8n2d?from=from_copylink
#     '''括号中的 Dataset 表示"继承"
# 这叫做继承（Inheritance），是面向对象编程的一个重要概念。
# 用生活例子解释：
# 想象一下家族传承：
# 你爷爷有很多技能和特征（比如会做饭、会木工）
# 你继承了你爷爷的这些技能和特征，同时还可能有自己的新技能
# 在代码中：
# Dataset 是"父类"（就像你爷爷）
# CharBoWDataset 是"子类"（就像你）
# 子类继承了父类的所有功能，并添加了自己特有的功能
# 为什么要继承 Dataset？
# Dataset 是 PyTorch 框架提供的一个基础类，它已经定义了一些标准功能:
# # PyTorch 的 Dataset 类提供了这些基本功能：
# class Dataset:
#     def __len__(self):
#         # 返回数据集大小（需要子类自己实现）
#         raise NotImplementedError
#
#     def __getitem__(self, idx):
#         # 返回第idx个数据样本（需要子类自己实现）
#         raise NotImplementedError
#
# '''
    """
    自定义数据集类，用于处理字符级词袋模型数据
    继承自torch.utils.data.Dataset，实现数据加载功能
    """
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为PyTorch张量
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()  # 创建词袋向量

    def _create_bow_vectors(self):
        """
        创建词袋(Bag of Words)向量表示
        每个文本被表示为一个向量，向量的每个维度对应词汇表中的一个字符，
        值为该字符在文本中出现的次数
        """
        # 将文本转换为索引序列，并进行填充或截断以保持统一长度
        tokenized_texts = []#tokenized_texts的长度 = 原始文本的数量（不是40），tokenized_texts中每个元素（子列表）的长度 = 40（max_len
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))  # 使用0进行填充
            tokenized_texts.append(tokenized)
# 这段代码是在将文本转换为数字序列，这是自然语言处理中的一个重要步骤。让我用简单的方式解释：
# ### 代码功能概述
# 这段代码的作用是将文本数据转换为数字序列，每个字符都用对应的数字表示。
# ### 逐行解释
# tokenized_texts = []创建一个空列表，用来存放处理后的文本数据。
# for text in self.texts:遍历所有的文本，每次处理一条文本。
# tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
# 这是关键的一行，让我们详细分解：
# 1. `text[:self.max_len]` - 取文本的前[max_len]个字符（防止文本太长）
# 2. `for char in ...` - 遍历每个字符
# 3. `self.char_to_index.get(char, 0)` - 查找字符对应的数字，如果找不到就返回0
        ###get(char, 0) 的意思是：
        # 尝试在字典中查找char
        # 如果找到了，返回对应的数字
        # 如果没找到，返回默认值0
# 4. 整体结果是一个数字列表
# tokenized += [0] * (self.max_len - len(tokenized))  # 使用0进行填充
        ####################
         #+= 的基本含义
         #a += b 等价于 a = a + b
        #################
        #举个例子
        # tokenized = [1, 2]  # 已经处理的文本：[1, 2]
        # self.max_len = 6  # 目标长度：6
        # 需要填充的0个数 = 6 - 2 = 4  # 还需要4个0
        # [0] * 4 = [0, 0, 0, 0]  # 创建4个0
        # tokenized + [0, 0, 0, 0] = [1, 2, 0, 0, 0, 0]
        # 如果文本太短，用0填充到统一长度[max_len]。
# tokenized_texts.append(tokenized)
# 将处理好的数字序列添加到结果列表中。
# ### 完整示例
# 假设有这些数据：
# self.texts = ["播放", "导航到北京"]
# self.char_to_index = {'<pad>': 0, '播': 1, '放': 2, '导': 3, '航': 4, '到': 5, '北': 6, '京': 7}
# self.max_len = 6
# 处理过程：
# 1. **处理"播放"**：
#    - 取前6个字符："播放"（只有2个字符）
#    - 转换为数字：[1, 2]
#    - 填充到长度6：[1, 2, 0, 0, 0, 0]
#
# 2. **处理"导航到北京"**：
#    - 取前6个字符："导航到北"（实际有4个字符）
#    - 转换为数字：[3, 4, 5, 6]
#    - 填充到长度6：[3, 4, 5, 6, 0, 0]
# 最终结果：
# tokenized_texts = [
#     [1, 2, 0, 0, 0, 0],    # "播放"的数字表示
#     [3, 4, 5, 6, 0, 0]     # "导航到北"的数字表示
# ]
# ### 为什么要这样做？
# 1. **统一长度**：神经网络需要固定长度的输入
# 2. **数字化**：神经网络只能处理数字，不能处理文字
# 3. **标准化**：所有文本都转换为统一格式，便于批量处理
# 这就是自然语言处理中非常重要的文本预处理步骤！'''
        # 根据索引序列创建词袋向量
        bow_vectors = []
        for text_indices in tokenized_texts:#遍历每一条已经转换为数字序列的文本。
            bow_vector = torch.zeros(self.vocab_size)  # 创建一个全零向量，长度等于词汇表大小。这个向量将用来统计字符出现次数
            for index in text_indices:
                if index != 0: # 忽略填充符，如果索引不是0（0代表填充符<pad>），就处理它
                    bow_vector[index] += 1  # 统计字符出现次数，在对应位置加1，统计该字符出现的次数。
            bow_vectors.append(bow_vector) #将处理好的词袋向量添加到结果列表中。
        return torch.stack(bow_vectors) #将所有词袋向量堆叠成一个张量
# '''这段代码是创建词袋(Bag of Words)向量的核心部分。让我用简单的方式解释：
# ### 代码功能概述
# 这段代码将数字序列转换为词袋向量，统计每个字符在文本中出现的次数。
# bow_vectors = []创建一个空列表，用来存放所有文本的词袋向量。
# for text_indices in tokenized_texts:遍历每一条已经转换为数字序列的文本。
# bow_vector = torch.zeros(self.vocab_size)  # 初始化词袋向量
# 创建一个全零向量，长度等于词汇表大小。这个向量将用来统计字符出现次数。
#
# for index in text_indices:遍历当前文本中的每个数字（字符索引）。
# if index != 0:  # 忽略填充符,如果索引不是0（0代表填充符`<pad>`），就处理它。
# bow_vector[index] += 1  # 统计字符出现次数,在对应位置加1，统计该字符出现的次数。
            #######################列表通过索引可以直接修改元素的值，包括进行加减运算。
            #列表[索引] += 值  # 在指定位置的元素上加一个值
            # 列表[索引] -= 值  # 在指定位置的元素上减一个值
            # 列表[索引] *= 值  # 在指定位置的元素上乘一个值
            # 列表[索引] /= 值  # 在指定位置的元素上除以一个值

# bow_vectors.append(bow_vector)将处理好的词袋向量添加到结果列表中。
# return torch.stack(bow_vectors)  # 将所有词袋向量堆叠成一个张量
# 将所有词袋向量堆叠成一个二维张量返回。
#
# 假设有以下数据：
# 词汇表大小 = 5
# tokenized_texts = [
#     [1, 2, 0, 0, 0, 0],  # "播放" + 填充
#     [3, 4, 3, 4, 0, 0]   # "音乐音乐" + 填充
# ]
# 处理过程：
# 1. **处理第一条文本 [1, 2, 0, 0, 0, 0]**：
#    - 创建向量：[0, 0, 0, 0, 0]
#    - 索引1 → [0, 1, 0, 0, 0]
#    - 索引2 → [0, 1, 1, 0, 0]
#    - 忽略0 → [0, 1, 1, 0, 0]
# 2. **处理第二条文本 [3, 4, 3, 4, 0, 0]**：
#    - 创建向量：[0, 0, 0, 0, 0]
#    - 索引3 → [0, 0, 0, 1, 0]
#    - 索引4 → [0, 0, 0, 1, 1]
#    - 索引3 → [0, 0, 0, 2, 1]
#    - 索引4 → [0, 0, 0, 2, 2]
#    - 忽略0 → [0, 0, 0, 2, 2]
# 最终结果：
# bow_vectors = [
#     [0, 1, 1, 0, 0],  # "播放"的词袋表示
#     [0, 0, 0, 2, 2]   # "音乐音乐"的词袋表示
# ]
# ### 为什么要用词袋模型？
# 1. **固定长度**：无论原文多长，词袋向量长度都固定为词汇表大小
# 2. **数学化**：将文本转换为数学向量，便于神经网络处理
# 3. **语义信息**：保留了字符出现频率信息
# 这就是词袋模型(Bag of Words)的核心思想！'''
    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """根据索引获取单个样本"""
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    #https://fcnbisyf4ls8.feishu.cn/wiki/W2bGwQq21iIuh2klna2ckmJInPd?from=from_copylink
    """
    简单的分类器模型，包含两个全连接层
    继承自torch.nn.Module，实现前馈神经网络
    """
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()#https://fcnbisyf4ls8.feishu.cn/wiki/JBPQwCcpMiGRarknJzfccEMtnhf?from=from_copylink
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一个全连接层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 第二个全连接层

    def forward(self, x):
        """
        前向传播函数，定义数据在网络中的流动过程
        """
        # 手动实现每层的计算
        out = self.fc1(x)  # 第一层线性变换
        out = self.relu(out)  # 应用ReLU激活函数
        out = self.fc2(out)  # 第二层线性变换
        return out


# 创建数据集实例和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

# 定义模型超参数
hidden_dim = 128  # 隐藏层维度
output_dim = len(label_to_index)  # 输出维度等于类别数
# 创建模型实例
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
# 定义损失函数，CrossEntropyLoss内部包含Softmax激活
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
# 定义优化器，使用随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练参数
num_epochs = 10  # 训练轮数

# 开始训练模型
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 记录累计损失
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 累计损失
        if idx % 50 == 0:#每处理 50 个 batch 就打印一次损失信息，避免打印太多信息，让输出更清晰，可以监控训练进度，但不会被过多信息淹没
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")  # 每50个batch打印一次损失


    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")  # 打印每个epoch的平均损失


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    对新文本进行分类预测
    
    参数:
    text: 待分类的文本
    model: 训练好的模型
    char_to_index: 字符到索引的映射
    vocab_size: 词汇表大小
    max_len: 最大文本长度
    index_to_label: 索引到标签的映射
    
    返回:
    predicted_label: 预测的标签
    """
    # 将文本转换为索引序列并进行填充
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    # 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)  # 增加一个维度以匹配模型输入

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        output = model(bow_vector)  # 前向传播

    # 获取预测结果
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建索引到标签的映射，用于预测结果的转换
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试模型对新文本的分类效果
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")