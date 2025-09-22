import random
from typing import List

# 设置随机种子以确保结果可复现
random.seed(42) # 去掉此行，获得随机结果

# 设置生成文本的提示词列表、最大生成token数和训练参数
prompts = ["春江", "往事"]  # 推理时使用的提示词列表
max_new_token = 100  # 推理生成的最大token数量
max_iters = 8000  # 训练的最大迭代次数
batch_size = 32  # 每个批次的大小
block_size = 8  # 每个序列的最大长度

# 读取训练数据
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

class Tokenizer:
    """分词器类，用于字符级别的编码和解码"""
    
    def __init__(self, text: str):
        """初始化分词器
        Args:
            text: 训练文本数据
        """
        self.chars = sorted(list(set(text)))  # 获取所有不重复的字符并排序
        self.vocab_size = len(self.chars)  # 词汇表大小
        # 创建字符到索引和索引到字符的映射字典
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, s: str) -> List[int]:
        """将字符串编码为索引列表
        Args:
            s: 输入字符串
        Returns:
            编码后的索引列表
        """
        return [self.stoi[c] for c in s]
    
    def decode(self, l: List[int]) -> str:
        """将索引列表解码为字符串
        Args:
            l: 索引列表
        Returns:
            解码后的字符串
        """
        return ''.join([self.itos[i] for i in l])

class BigramLanguageModel:
    """二元语言模型类"""
    
    def __init__(self, vocab_size: int):
        """初始化二元语言模型
        Args:
            vocab_size: 词汇表大小
        """
        self.vocab_size = vocab_size
        # 创建转移概率矩阵，记录每个词的下一个词的出现次数
        # 矩阵维度为 vocab_size * vocab_size
        #      a    b    c  ... (vocab_size)
        #   a  0    1    0  ...
        #   b  0    0    3  ...
        #   c  4    0    0  ...
        #  ... ... ... ... ... ... ...
        # (vocab_size)
        self.transition = [[0.0 for _ in range(vocab_size)] 
                          for _ in range(vocab_size)]
        
    def __call__(self, x):
        """使模型实例可以像函数一样被调用
        Args:
            x: 输入数据
        Returns:
            模型前向传播结果
        """
        # 方便直接调用model(x)
        return self.forward(x)
    
    def forward(self, idx: List[List[int]]) -> List[List[List[float]]]:
        """前向传播函数
        Args:
            idx: 输入的token索引列表，二维数组
                  [[1, 2, 3],
                   [4, 5, 6]]
        Returns:
            每个token的下一个token的概率分布，三维数组
                  [[[0.1, 0.2, 0.3, .. (vocab_size)],
                    [0.4, 0.5, 0.6, .. (vocab_size)],
                    [0.7, 0.8, 0.9, .. (vocab_size)]],
                   [[0.2, 0.3, 0.4, .. (vocab_size)],
                    [0.5, 0.6, 0.7, .. (vocab_size)],
                    [0.8, 0.9, 1.0, .. (vocab_size)]]]
        """
        B = len(idx)  # 批次大小
        T = len(idx[0])  # 每一批的序列长度
        
        # 初始化logits矩阵，用于存储每个token的下一个token的概率分布
        logits = [
            [[0.0 for _ in range(self.vocab_size)] 
             for _ in range(T)]
            for _ in range(B)
        ]
        
        # 计算每个token的下一个token的概率分布
        for b in range(B):
            for t in range(T):
                current_token = idx[b][t]
                # 计算了每一个token的下一个token的概率
                for i in range(self.vocab_size):
                    logits[b][t][i] = float(self.transition[current_token][i])
                
        return logits

    def generate(self, idx: List[List[int]], max_new_tokens: int) -> List[List[int]]:
        """生成新文本
        Args:
            idx: 初始token序列
            max_new_tokens: 最大生成token数
        Returns:
            生成的token序列
        """
        # 逐个生成新token
        for _ in range(max_new_tokens):
            logits_batch = self(idx)  # 获取每个序列中每个token的下一个token概率分布
            # 处理每个批次中的序列
            for batch_idx, logits in enumerate(logits_batch):
                # 我们计算了每一个token的下一个token的概率
                # 但实际上我们只需要最后一个token的"下一个token的概率"
                logits = logits[-1]
                # 计算总和，避免除零错误
                total = max(sum(logits), 1.0)
                # 归一化概率分布
                logits = [logit / total for logit in logits]
                # 根据概率分布随机采样下一个token
                next_token = random.choices(
                    range(self.vocab_size),
                    weights=logits,
                    k=1
                )[0]
                # 将新生成的token添加到序列末尾
                idx[batch_idx].append(next_token)
        return idx
    
def get_batch(tokens, batch_size, block_size):
    """随机获取一批数据用于训练
    Args:
        tokens: 全部训练数据的token列表
        batch_size: 批次大小
        block_size: 序列长度
    Returns:
        训练数据x和标签数据y
        x和y都是二维数组，可以用于并行训练
        其中y数组内的每一个值，都是x数组内对应位置的值的下一个值
        格式如下：
        x = [[1, 2, 3],
             [9, 10, 11]]
        y = [[2, 3, 4],
             [10, 11, 12]]
    """
    # 随机选择批次索引
    ix = random.choices(range(len(tokens) - block_size), k=batch_size)
    x, y = [], []
    # 构建训练数据和标签数据
    for i in ix:
        x.append(tokens[i:i+block_size])
        y.append(tokens[i+1:i+block_size+1])
    return x, y

# 初始化分词器和模型
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

tokens = tokenizer.encode(text)  # 将文本编码为token序列

model = BigramLanguageModel(vocab_size)

# 训练模型
for iter in range(max_iters):
    # 获取训练批次数据
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    # 更新转移概率矩阵
    for i in range(len(x_batch)):
        for j in range(len(x_batch[i])):
            x = x_batch[i][j]  # 当前token
            y = y_batch[i][j]  # 下一个token
            model.transition[x][y] += 1.0  # 更新转移计数

# 将提示词编码为token序列
prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]

# 推理生成文本
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印生成结果
for tokens in result:
    print(tokenizer.decode(tokens))
    print('-'*10)