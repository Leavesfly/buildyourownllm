import torch
import torch.nn as nn
import random
from typing import List

# 设置随机种子以确保结果可复现
random.seed(42)
torch.manual_seed(42)

# 设置生成文本的提示词列表、最大生成token数和训练参数
prompts = ["春江", "往事"]  # 推理时使用的提示词列表
max_new_token = 100  # 推理生成的最大token数量
max_iters = 5000  # 训练的最大迭代次数
batch_size = 32  # 每个批次的大小
block_size = 8  # 每个序列的最大长度

# 设置设备
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

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
    """基于PyTorch的二元语言模型类"""
    
    def __init__(self, vocab_size: int):
        """初始化二元语言模型
        Args:
            vocab_size: 词汇表大小
        """
        self.vocab_size = vocab_size
        # 创建转移概率矩阵，使用PyTorch张量存储
        self.transition = torch.zeros((vocab_size, vocab_size), device=device)
    
    def __call__(self, x):
        """使模型实例可以像函数一样被调用
        Args:
            x: 输入数据
        Returns:
            模型前向传播结果
        """
        return self.forward(x)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """前向传播函数
        Args:
            idx: 输入的token索引张量，形状为(B, T)
        Returns:
            每个token的下一个token的概率分布，形状为(B, T, vocab_size)
        """
        # idx shape: (B, T)
        B, T = idx.shape
        # 初始化结果张量
        result = torch.zeros((B, T, self.vocab_size), device=device)
        # 计算每个token的下一个token的概率分布
        for b in range(B):
            for t in range(T):
                result[b][t] = self.transition[idx[b][t]]
        return result  # shape: (B, T, vocab_size)

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """生成新文本
        Args:
            idx: 初始token序列张量
            max_new_tokens: 最大生成token数
        Returns:
            生成的token序列张量
        """
        # 逐个生成新token
        for _ in range(max_new_tokens):
            # 获取最后一个token的预测
            logits = self(idx)[:, -1, :]  # (B, vocab_size)
            # 将计数转换为概率
            probs = logits / torch.clamp(logits.sum(dim=-1, keepdim=True), min=1.0)
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 拼接到序列中
            idx = torch.cat([idx, next_token], dim=1)  # (B, T+1)
        return idx

def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int):
    """随机获取一批数据用于训练
    Args:
        tokens: 全部训练数据的token张量
        batch_size: 批次大小
        block_size: 序列长度
    Returns:
        训练数据x和标签数据y张量
    """
    # 随机选择批次索引
    ix = torch.randint(len(tokens) - block_size, (batch_size,), device=device)
    # 构建训练数据和标签数据
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x, y

# 初始化分词器和模型
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

# 将文本编码为tensor
tokens = torch.tensor(tokenizer.encode(text)).to(device)

model = BigramLanguageModel(vocab_size)

# 训练模型
for iter in range(max_iters):
    # 获取训练批次数据
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    # 更新转移概率矩阵
    for i in range(batch_size):
        for j in range(block_size):
            x = x_batch[i, j]  # 当前token
            y = y_batch[i, j]  # 下一个token
            model.transition[x, y] += 1

# 将提示词转换为tensor并处理
prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)) for p in prompts])

# 推理生成文本
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印生成结果
for tokens in result:
    print(tokenizer.decode(tokens.tolist()))
    print('-'*10)