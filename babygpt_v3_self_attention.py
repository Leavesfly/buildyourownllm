import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time

# 设置随机数种子，确保结果可复现
torch.manual_seed(42)

# 设置生成文本的提示词列表、最大生成token数和训练参数
prompts = ["春江", "往事"]  # 推理的输入prompts
max_new_token = 100  # 推理生成的最大tokens数量

max_iters = 5000  # 训练的最大迭代次数
eval_iters = 100  # 评估的迭代次数
eval_interval = 200  # 评估的间隔
batch_size = 32  # 每个批次的大小
block_size = 8  # 每个序列的最大长度
learning_rate = 1e-2  # 学习率
n_embed = 32  # 嵌入层的维度
tain_data_ratio = 0.9  # 训练数据占数据集的比例，剩下的是验证数据

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

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

class Head(nn.Module):
    """注意力头类"""
    
    def __init__(self, head_size):
        """初始化注意力头
        Args:
            head_size: 注意力头的大小
        """
        super().__init__()
        # 定义查询、键、值的线性变换层
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # 注册缓冲区，存储下三角矩阵用于掩码
        # __init__里的module都会被pytorch自动当作layer来处理，用register_buffer后，这里就是一个普通的变量
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为(batch_size, block_size, n_embed)
        Returns:
            注意力输出张量
        """
        B, T, C = x.shape  # (batch_size, block_size, n_embed)
        k = self.key(x)    # (B, T, head_size) 计算键
        q = self.query(x)  # (B, T, head_size) 计算查询
        v = self.value(x)  # (B, T, head_size) 计算值
        
        # 计算注意力权重
        # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        # 最后缩放避免softmax过于稀疏
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        
        # 应用掩码，确保只能关注到前面的token
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 应用softmax函数
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # 计算输出
        out = wei @ v  # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
    
class BabyGPT(nn.Module):
    """BabyGPT模型v3版本 - 引入自注意力机制"""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        """初始化BabyGPT模型
        Args:
            vocab_size: 词汇表大小
            block_size: 序列长度
            n_embd: 嵌入层维度
        """
        super().__init__()
        self.block_size = block_size
        # 嵌入层，把token映射到n_embd维空间
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 位置嵌入层，把位置信息映射到n_embd维空间
        self.postion_embedding_table = nn.Embedding(block_size, n_embed)
        # 自注意力头
        self.sa_head = Head(n_embed)
        # 线性层，把n_embd维空间映射到vocab_size维空间
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """前向传播函数
        Args:
            idx: 输入的token索引
            targets: 目标标签（用于训练）
        Returns:
            logits: 模型输出
            loss: 损失值（仅在训练时计算）
        """
        B, T = idx.shape  # B是batch size，T是block size
        T = min(T, self.block_size)
        idx = idx[:, -T:]  # 不管输入的序列有多长，我们只取最后的block_size个token
        
        # 获取token的嵌入表示 (B,T,n_embd)
        tok_emb = self.token_embedding_table(idx)
        # 获取位置的嵌入表示 (T,n_embd)
        pos_emb = self.postion_embedding_table(torch.arange(T, device=idx.device))
        # 给token的嵌入表示加上位置的嵌入表示，x有了"位置"信息！
        x = tok_emb + pos_emb
        # 应用自注意力机制
        x = self.sa_head(x)
        # 通过线性层，把embedding结果重新映射回vocab_size维空间 (B,T,vocab_size)
        logits = self.lm_head(x)

        if targets is None:  # 推理场景，不需要计算损失值
            loss = None
        else:
            B, T, C = logits.shape
            # 把(B,T,C)的形状转换为(B*T,C)，因为交叉熵损失函数第一个参数只接受二维输入
            logits = logits.view(B*T, C)
            # 把(B,T)的形状转换为(B*T)，因为交叉熵损失函数第二个参数只接受一维输入
            targets = targets.view(B*T)
            # 计算交叉熵损失
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """生成新文本
        Args:
            idx: 初始token序列
            max_new_tokens: 最大生成token数
        Returns:
            生成的token序列
        """
        # 逐个生成新token
        for _ in range(max_new_tokens):
            # 获取模型输出，logits的形状是(B,T,vocab_size)，每一个token都计算了下一个token的概率
            logits, _ = self(idx)
            # 实际上我们只需要最后一个token算出来的值
            logits = logits[:, -1, :]
            # 使用softmax函数算概率分布，这里dim=-1表示对最后一个维度进行softmax
            probs = F.softmax(logits, dim=-1)
            # 根据概率分布随机采样，这里num_samples=1表示采样一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            # 把采样的token拼接到序列后面
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 初始化分词器
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

# 将文本编码为tensor并划分训练集和验证集
raw_data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device)
n = int(tain_data_ratio*len(raw_data))  # 训练数据长度
data = {'train': raw_data[:n], 'val': raw_data[n:]}  # 划分训练集和验证集

def get_batch(data, batch_size, block_size):
    """随机获取一批数据用于训练或验证
    Args:
        data: 训练或验证数据
        batch_size: 批次大小
        block_size: 序列长度
    Returns:
        训练数据x和标签数据y张量
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))  # 随机选择批次索引
    x = torch.stack([data[i:i+block_size] for i in ix])  # 构建训练数据
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # 构建标签数据
    x, y = x.to(device), y.to(device)  # 移动到指定设备
    return x, y

@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_iters):
    """计算模型在训练集和验证集上的损失
    Args:
        model: 模型实例
        data: 训练和验证数据
        batch_size: 批次大小
        block_size: 序列长度
        eval_iters: 评估迭代次数
    Returns:
        各数据集上的平均损失
    """
    out = {}
    model.eval()  # 切换到评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data[split], batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # 切换回训练模式
    return out

# 创建模型并移动到指定设备
model = BabyGPT(vocab_size, block_size, n_embed).to(device)

# 训练模型
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 使用AdamW优化器

# 记录训练时间和处理的token数
start_time = time.time()
tokens_processed = 0

for iter in range(max_iters):
    # 获取训练批次数据
    x, y = get_batch(data['train'], batch_size, block_size)
    # 前向传播和计算损失
    logits, loss = model(x, y)
    # 清空梯度、反向传播、更新参数
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # 统计处理的token数
    tokens_processed += batch_size * block_size

    # 定期评估模型并打印训练信息
    if iter % eval_interval == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        losses = estimate_loss(model, data, batch_size, block_size, eval_iters)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec")

# 推理生成文本
# 将提示词转换为tensor
prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)).to(device) for p in prompts])

# 生成文本
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印生成结果
for tokens in result:
    print(tokenizer.decode(tokens.tolist()))
    print('-'*10)