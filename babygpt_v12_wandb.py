import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time
import wandb

# 设置随机数种子，确保结果可复现
torch.manual_seed(42)

# 设置生成文本的提示词列表、最大生成token数和训练参数
prompts = ["春江", "往事"]  # 推理的输入prompts
max_new_token = 100  # 推理生成的最大tokens数量
max_iters = 5000  # 训练的最大迭代次数
eval_iters = 100  # 评估的迭代次数
eval_interval = 50  # 评估的间隔
batch_size = 64  # 每个批次的大小
block_size = 256  # 每个序列的最大长度
learning_rate = 3e-4  # 学习率
n_embed = 384  # 嵌入层的维度
n_head = 6  # 多头注意力的头数
n_layer = 6  # block的数量
dropout = 0.2  # dropout的比例
tain_data_ratio = 0.9  # 训练数据占数据集的比例，剩下的是验证数据

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# 初始化Weights & Biases，用于实验跟踪和可视化
wandb.init(
    project="babygpt",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embed": n_embed,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }
)

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

class Block(nn.Module):
    """模型块类，包含多头注意力和前馈网络，使用残差连接和层归一化"""
    
    def __init__(self, n_embed, n_head):
        """初始化模型块
        Args:
            n_embed: 嵌入层维度
            n_head: 注意力头数量
        """
        super().__init__()
        head_size = n_embed // n_head
        # 多头注意力层
        self.sa = MultiHeadAttention(n_head, head_size)
        # 前馈神经网络层
        self.ffwd = FeedFoward(n_embed)
        # 层归一化层1
        self.ln1 = nn.LayerNorm(n_embed)
        # 层归一化层2
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量
        Returns:
            模型块输出张量
        """
        # 使用残差连接，保留原来的x信息，避免梯度消失
        # 先进行层归一化，再进行多头注意力计算
        x = x + self.sa(self.ln1(x))
        # 先进行层归一化，再进行前馈网络计算
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedFoward(nn.Module):
    """前馈神经网络类，包含Dropout正则化"""
    
    def __init__(self, n_embed):
        """初始化前馈神经网络
        Args:
            n_embed: 嵌入层维度
        """
        super().__init__()
        self.net = nn.Sequential(
            # 第一个线性层，将维度从n_embed扩展到n_embed*4
            nn.Linear(n_embed, n_embed * 4),
            # ReLU激活函数，把负值变为0，正值不变
            nn.ReLU(),
            # 第二个线性层，将维度从n_embed*4压缩回n_embed
            nn.Linear(n_embed * 4, n_embed),
            # Dropout层，防止过拟合
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量
        Returns:
            前馈网络输出张量
        """
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """多头注意力类，包含Dropout正则化"""
    
    def __init__(self, num_heads, head_size):
        """初始化多头注意力
        Args:
            num_heads: 注意力头的数量
            head_size: 每个注意力头的大小
        """
        super().__init__()
        # 创建多个注意力头
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 投影层，把多头注意力的输出映射回n_embed维度
        self.proj = nn.Linear(n_embed, n_embed)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量
        Returns:
            多头注意力输出张量
        """
        # 将所有注意力头的输出在最后一个维度上拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 通过投影层将输出映射回n_embed维度
        out = self.proj(out)
        # 应用dropout
        out = self.dropout(out)
        return out

class Head(nn.Module):
    """注意力头类，包含Dropout正则化"""
    
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
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

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
        # 上三角都是-inf，下三角是q和k的点积
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 应用softmax函数
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # 应用dropout
        wei = self.dropout(wei)
        # 计算输出
        out = wei @ v  # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
    
class BabyGPT(nn.Module):
    """BabyGPT模型v12版本 - wandb集成"""

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
        # 创建多个模型块（包含多头注意力和前馈网络）
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        # 最终层归一化层
        self.ln_final = nn.LayerNorm(n_embed)
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
        # 通过多个模型块处理
        x = self.blocks(x)
        # 最终层归一化
        x = self.ln_final(x)
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

# 使用AdamW优化器，学习率设置为3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
        elapsed_mins = elapsed // 60
        elapsed_secs = elapsed % 60
        # 使用wandb记录训练指标
        wandb.log({
            "train_loss": losses['train'],
            "val_loss": losses['val'],
            "tokens_per_sec": tokens_per_sec,
            "iteration": iter
        })
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec, time: {int(elapsed_mins)}m {elapsed_secs:.1f}s")

# 推理生成文本
# 将提示词转换为tensor
prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)).to(device) for p in prompts])

# 生成文本
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印生成结果
for tokens in result:
    print(tokenizer.decode(tokens.tolist()))
    print('-'*10)

# 保存模型权重
save_path = 'model.pth'
torch.save(model.state_dict(), save_path)
print(f"模型已保存到{save_path}")

# 示例输出（注释掉，避免在实际运行时执行）
"""
simpx@ThePC:~/buildyourownllm$ python babygpt_v12_wandb.py
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: simpxx (simpxx-zhejiang-university). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /home/simpx/buildyourownllm/wandb/run-20250309_235239-ysgr3tei
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run absurd-frog-1
wandb: ⭐️ View project at https://wandb.ai/simpxx-zhejiang-university/babygpt
wandb: 🚀 View run at https://wandb.ai/simpxx-zhejiang-university/babygpt/runs/ysgr3tei
step 0: train loss 8.0529, val loss 8.0512, speed: 55304.00 tokens/sec, time: 0m 0.3s
step 50: train loss 5.9337, val loss 6.0072, speed: 102707.49 tokens/sec, time: 0m 8.1s
step 100: train loss 5.7529, val loss 5.8782, speed: 104971.14 tokens/sec, time: 0m 15.8s
step 150: train loss 5.4843, val loss 5.6578, speed: 105831.56 tokens/sec, time: 0m 23.4s
step 200: train loss 5.2826, val loss 5.4927, speed: 106169.27 tokens/sec, time: 0m 31.0s
step 250: train loss 5.1371, val loss 5.3766, speed: 105984.49 tokens/sec, time: 0m 38.8s
step 300: train loss 5.0116, val loss 5.2703, speed: 105957.57 tokens/sec, time: 0m 46.5s
step 350: train loss 4.9237, val loss 5.1528, speed: 106056.41 tokens/sec, time: 0m 54.2s
step 400: train loss 4.8080, val loss 5.0865, speed: 105914.43 tokens/sec, time: 1m 2.0s
step 450: train loss 4.7279, val loss 4.9910, speed: 105835.10 tokens/sec, time: 1m 9.8s
step 500: train loss 4.6646, val loss 4.9363, speed: 105850.83 tokens/sec, time: 1m 17.5s
step 550: train loss 4.6004, val loss 4.8573, speed: 105825.23 tokens/sec, time: 1m 25.3s
step 600: train loss 4.5383, val loss 4.8317, speed: 105717.62 tokens/sec, time: 1m 33.1s
step 650: train loss 4.4883, val loss 4.7752, speed: 105713.79 tokens/sec, time: 1m 40.9s
step 700: train loss 4.4415, val loss 4.7334, speed: 105657.30 tokens/sec, time: 1m 48.7s
step 750: train loss 4.4077, val loss 4.7024, speed: 105533.37 tokens/sec, time: 1m 56.6s
step 800: train loss 4.3546, val loss 4.6546, speed: 105446.23 tokens/sec, time: 2m 4.5s
step 850: train loss 4.3154, val loss 4.6399, speed: 105418.73 tokens/sec, time: 2m 12.3s
step 900: train loss 4.2720, val loss 4.5936, speed: 105405.89 tokens/sec, time: 2m 20.0s
step 950: train loss 4.2308, val loss 4.5587, speed: 105314.04 tokens/sec, time: 2m 27.9s
step 1000: train loss 4.1714, val loss 4.5141, speed: 105286.65 tokens/sec, time: 2m 35.8s
step 1050: train loss 4.1327, val loss 4.4774, speed: 105286.05 tokens/sec, time: 2m 43.6s
step 1100: train loss 4.1021, val loss 4.4610, speed: 105221.62 tokens/sec, time: 2m 51.4s
step 1150: train loss 4.0632, val loss 4.4143, speed: 105132.34 tokens/sec, time: 2m 59.4s
step 1200: train loss 4.0170, val loss 4.3883, speed: 105118.14 tokens/sec, time: 3m 7.2s
step 1250: train loss 3.9844, val loss 4.3670, speed: 105053.55 tokens/sec, time: 3m 15.1s
step 1300: train loss 3.9601, val loss 4.3501, speed: 105018.94 tokens/sec, time: 3m 23.0s
step 1350: train loss 3.9226, val loss 4.3310, speed: 105021.13 tokens/sec, time: 3m 30.8s
step 1400: train loss 3.9077, val loss 4.3136, speed: 105022.69 tokens/sec, time: 3m 38.6s
step 1450: train loss 3.8786, val loss 4.2988, speed: 104984.37 tokens/sec, time: 3m 46.4s
step 1500: train loss 3.8503, val loss 4.2784, speed: 104971.83 tokens/sec, time: 3m 54.3s
step 1550: train loss 3.8237, val loss 4.2614, speed: 105000.23 tokens/sec, time: 4m 2.0s
step 1600: train loss 3.8005, val loss 4.2503, speed: 104940.92 tokens/sec, time: 4m 10.0s
step 1650: train loss 3.7833, val loss 4.2264, speed: 104912.89 tokens/sec, time: 4m 17.8s
step 1700: train loss 3.7564, val loss 4.2210, speed: 104901.07 tokens/sec, time: 4m 25.7s
step 1750: train loss 3.7411, val loss 4.2056, speed: 104898.69 tokens/sec, time: 4m 33.5s
step 1800: train loss 3.7157, val loss 4.1930, speed: 104873.22 tokens/sec, time: 4m 41.4s
step 1850: train loss 3.7006, val loss 4.1794, speed: 104863.20 tokens/sec, time: 4m 49.2s
step 1900: train loss 3.6843, val loss 4.1722, speed: 104882.90 tokens/sec, time: 4m 57.0s
"""