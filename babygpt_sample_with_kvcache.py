import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time

# 设置随机数种子，确保实验结果可复现
torch.manual_seed(42)

# 超参数配置
max_new_token = 100  # 推理时最大生成的token数量
max_iters = 5000     # 训练最大迭代次数
eval_iters = 100     # 评估时的迭代次数
eval_interval = 50   # 评估间隔步数
batch_size = 64      # 批次大小
block_size = 256     # 序列最大长度（上下文窗口大小）
learning_rate = 3e-4 # 学习率
n_embed = 384        # 嵌入层维度
n_head = 6           # 多头注意力机制中的头数
n_layer = 6          # Transformer块的数量
dropout = 0.2        # Dropout比例
tain_data_ratio = 0.9  # 训练数据占比，剩余部分作为验证数据

# 设备配置：优先使用CUDA，其次MPS（Mac GPU），最后回退到CPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# 读取训练数据文件
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

class Tokenizer:
    """字符级分词器，用于将文本转换为token序列以及反向转换"""
    
    def __init__(self, text: str):
        """初始化分词器，构建词汇表
        
        Args:
            text: 用于构建词汇表的训练文本数据
        """
        # 提取文本中所有不重复的字符并排序，构建词汇表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)  # 词汇表大小
        
        # 构建字符到索引和索引到字符的双向映射字典
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string to index
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # index to string
    
    def encode(self, s: str) -> List[int]:
        """将字符串编码为token索引序列
        
        Args:
            s: 待编码的输入字符串
            
        Returns:
            编码后的token索引列表
        """
        return [self.stoi[c] for c in s]
    
    def decode(self, l: List[int]) -> str:
        """将token索引序列解码为字符串
        
        Args:
            l: 待解码的token索引列表
            
        Returns:
            解码后的字符串
        """
        return ''.join([self.itos[i] for i in l])

class Block(nn.Module):
    """Transformer中的基本块，包含多头注意力和前馈神经网络
    
    采用残差连接和层归一化结构，遵循"Pre-LN" Transformer架构
    """
    
    def __init__(self, n_embed, n_head):
        """初始化Transformer块
        
        Args:
            n_embed: 嵌入维度
            n_head: 注意力头的数量
        """
        super().__init__()
        head_size = n_embed // n_head  # 每个注意力头的维度
        
        # 多头自注意力机制层
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # 前馈神经网络层
        self.ffwd = FeedFoward(n_embed)
        
        # 层归一化层1（用于注意力子层之前）
        self.ln1 = nn.LayerNorm(n_embed)
        
        # 层归一化层2（用于前馈网络子层之前）
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, kv_cache=None):
        """前向传播过程
        
        Args:
            x: 输入张量，形状为(batch_size, sequence_length, n_embed)
            kv_cache: KV缓存，用于加速推理过程，None表示不使用缓存
            
        Returns:
            tuple: (输出张量, 更新后的KV缓存)
                - 输出张量形状为(batch_size, sequence_length, n_embed)
                - KV缓存用于下一次推理
        """
        # 注意力子层：层归一化 -> 多头注意力 -> 残差连接
        sa_out, new_kv_cache = self.sa(self.ln1(x), kv_cache)
        x = x + sa_out
        
        # 前馈网络子层：层归一化 -> 前馈网络 -> 残差连接
        x = x + self.ffwd(self.ln2(x))
        
        return x, new_kv_cache

class FeedFoward(nn.Module):
    """前馈神经网络层，包含两个线性变换和激活函数
    
    结构：Linear -> ReLU -> Linear -> Dropout
    """
    
    def __init__(self, n_embed):
        """初始化前馈网络
        
        Args:
            n_embed: 输入和输出的嵌入维度
        """
        super().__init__()
        # 序贯网络结构：
        # 1. 扩展维度：从n_embed到4*n_embed
        # 2. 激活函数：ReLU非线性变换
        # 3. 压缩维度：从4*n_embed回到n_embed
        # 4. 正则化：Dropout防止过拟合
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为(..., n_embed)
            
        Returns:
            输出张量，形状与输入相同
        """
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """多头注意力机制实现，支持KV缓存优化
    
    将注意力计算分解为多个"头"并行处理，然后合并结果
    """
    
    def __init__(self, num_heads, head_size):
        """初始化多头注意力
        
        Args:
            num_heads: 注意力头的数量
            head_size: 每个注意力头的维度
        """
        super().__init__()
        # 创建多个并行的注意力头
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # 输出投影层：将多头拼接后的结果映射回嵌入维度
        self.proj = nn.Linear(n_embed, n_embed)
        
        # Dropout正则化层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        """前向传播过程
        
        Args:
            x: 输入张量，形状为(batch_size, sequence_length, n_embed)
            kv_cache: KV缓存列表，每个头对应一个缓存元素
            
        Returns:
            tuple: (输出张量, 更新后的KV缓存列表)
        """
        outputs = []       # 存储每个头的输出
        new_kv_caches = [] # 存储每个头更新后的KV缓存
        
        # 并行处理每个注意力头
        for i, head in enumerate(self.heads):
            # 获取当前头对应的KV缓存
            head_kv_cache = None if kv_cache is None else kv_cache[i]
            
            # 计算当前头的输出和更新后的KV缓存
            out, new_head_kv_cache = head(x, head_kv_cache)
            outputs.append(out)
            new_kv_caches.append(new_head_kv_cache)
        
        # 拼接所有头的输出并在投影回嵌入维度
        out = torch.cat(outputs, dim=-1)  # 在特征维度拼接
        out = self.proj(out)              # 线性投影
        out = self.dropout(out)           # Dropout正则化
        
        return out, new_kv_caches

class Head(nn.Module):
    """单个注意力头的实现，支持KV缓存优化
    
    实现缩放点积注意力机制，包含因果掩码确保自回归特性
    """
    
    def __init__(self, head_size):
        """初始化注意力头
        
        Args:
            head_size: 注意力头的维度
        """
        super().__init__()
        # 线性变换层：将输入映射到查询(Q)、键(K)、值(V)空间
        self.key = nn.Linear(n_embed, head_size, bias=False)    # K矩阵变换
        self.query = nn.Linear(n_embed, head_size, bias=False)  # Q矩阵变换
        self.value = nn.Linear(n_embed, head_size, bias=False)  # V矩阵变换
        
        # 因果掩码：注册为缓冲区，确保只关注前面的token（下三角矩阵）
        # 使用register_buffer使该变量能够被PyTorch正确处理但不作为模型参数
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # Dropout正则化层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        """前向传播过程，实现缩放点积注意力
        
        Args:
            x: 输入张量，形状为(batch_size, sequence_length, n_embed)
            kv_cache: KV缓存元组(k_cache, v_cache)，用于加速推理
            
        Returns:
            tuple: (注意力输出, 更新后的KV缓存)
                - 注意力输出形状为(batch_size, sequence_length, head_size)
                - KV缓存包含键和值的缓存张量
        """
        B, T, C = x.shape  # B:批次大小, T:序列长度, C:嵌入维度
        
        # 提取当前处理的token（序列中的最后一个token）
        current_token = x[:, -1:, :]  # 形状: (B, 1, C)
        
        # 计算查询向量（只计算当前token的查询）
        q = self.query(current_token)  # 形状: (B, 1, head_size)

        # 根据是否使用缓存来计算键和值向量
        if kv_cache is None:
            # 训练或无缓存推理：计算整个序列的K和V
            k = self.key(x)    # 形状: (B, T, head_size)
            v = self.value(x)  # 形状: (B, T, head_size)
        else:
            # 推理优化：使用缓存的K和V，并只计算当前token的新K和V
            k_cache, v_cache = kv_cache  # 从缓存中获取历史K和V
            k = self.key(current_token)  # 计算当前token的K
            v = self.value(current_token)  # 计算当前token的V
            
            # 将历史K、V与当前K、V拼接，形成完整的K、V序列
            k = torch.cat([k_cache, k], dim=1)  # 在时间维度拼接
            v = torch.cat([v_cache, v], dim=1)  # 在时间维度拼接
        
        # 更新KV缓存（用于下一次推理）
        new_kv_cache = (k, v)
        
        # 计算注意力分数：Q @ K^T / sqrt(d_k)
        # q: (B, 1, head_size), k: (B, T_total, head_size) -> (B, 1, T_total)
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        
        # 获取序列长度信息
        T_total = k.size(1)      # 总的序列长度（包括历史缓存）
        T_current = q.size(1)    # 当前处理的序列长度（通常为1）
        
        # 创建因果掩码确保自回归特性（当前token只能关注历史token）
        # 注意：掩码大小需要匹配wei的最后两个维度 (T_current, T_total)
        mask = torch.tril(torch.ones(T_total, T_total, device=x.device))
        mask = mask[-T_current:, :]  # 只取与当前查询匹配的部分
        
        # 应用因果掩码：将被掩码位置的注意力分数设为负无穷
        wei = wei.masked_fill(mask == 0, float('-inf'))
        
        # 对注意力分数进行softmax归一化
        wei = F.softmax(wei, dim=-1)  # 形状: (B, T_current, T_total)
        
        # 应用dropout正则化
        wei = self.dropout(wei)
        
        # 计算加权值：Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) @ V
        # wei: (B, T_current, T_total) @ v: (B, T_total, head_size) -> (B, T_current, head_size)
        out = wei @ v
        
        return out, new_kv_cache
    
class BabyGPT(nn.Module):
    """简化版GPT模型，支持KV缓存优化的推理加速"""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        """初始化BabyGPT模型
        
        Args:
            vocab_size: 词汇表大小
            block_size: 序列最大长度（上下文窗口）
            n_embd: 嵌入维度
        """
        super().__init__()
        self.block_size = block_size  # 存储序列长度限制
        
        # Token嵌入层：将token索引映射到嵌入向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # 位置嵌入层：将位置信息映射到嵌入向量
        self.postion_embedding_table = nn.Embedding(block_size, n_embed)
        
        # 多个Transformer块组成的网络主体
        # 使用ModuleList便于单独处理每个block的KV缓存
        self.blocks = nn.ModuleList([Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        
        # 最终层归一化：在输出前进行归一化
        self.ln_final = nn.LayerNorm(n_embed)
        
        # 语言模型头部：将嵌入向量映射回词汇表概率分布
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, kv_cache=None):
        """前向传播过程
        
        Args:
            idx: 输入token索引张量，形状为(batch_size, sequence_length)
            targets: 目标token索引张量，用于训练时计算损失，形状为(batch_size, sequence_length)
            kv_cache: KV缓存列表，用于推理加速，每个block对应一个缓存元素
            
        Returns:
            tuple: (logits, loss, new_kv_cache)
                - logits: 未归一化的词汇表概率分布，形状为(batch_size, sequence_length, vocab_size)
                - loss: 交叉熵损失值（仅在提供targets时计算）
                - new_kv_cache: 更新后的KV缓存列表
        """
        B, T = idx.shape  # B:批次大小, T:序列长度
        
        # 根据是否使用KV缓存来处理输入序列
        if kv_cache is not None:
            # 推理优化模式：只处理最新的token
            # 计算位置偏移量（历史token数量）
            pos_offset = kv_cache[0][0][0].size(1) if kv_cache and kv_cache[0] else 0
            idx = idx[:, -1:]  # 只保留最后一个token
            T = 1
        else:
            # 训练或无缓存推理模式：处理完整序列
            pos_offset = 0
            T = min(T, self.block_size)  # 限制序列长度不超过block_size
            idx = idx[:, -T:]  # 截取最后的T个token
            
        # 计算token嵌入和位置嵌入
        tok_emb = self.token_embedding_table(idx)  # Token嵌入: (B, T, n_embd)
        
        # 位置嵌入: (T, n_embd)，考虑位置偏移量
        pos_emb = self.postion_embedding_table(
            torch.arange(pos_offset, pos_offset + T, device=idx.device)
        )
        
        # 将token嵌入和位置嵌入相加，使模型具有位置感知能力
        x = tok_emb + pos_emb
        
        # 依次通过所有Transformer块，并维护KV缓存
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            # 获取当前block对应的KV缓存
            block_kv_cache = None if kv_cache is None else kv_cache[i]
            
            # 通过当前block并更新KV缓存
            x, block_new_kv_cache = block(x, block_kv_cache)
            new_kv_cache.append(block_new_kv_cache)
            
        # 最终层归一化
        x = self.ln_final(x)
        
        # 通过语言模型头部得到词汇表上的概率分布
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 根据是否提供目标来计算损失
        if targets is None:
            # 推理模式：不计算损失
            loss = None
        else:
            # 训练模式：计算交叉熵损失
            B, T, C = logits.shape
            # 重塑张量以适应交叉熵损失函数的要求
            logits = logits.view(B*T, C)      # (B*T, C)
            targets = targets.view(B*T)       # (B*T)
            loss = F.cross_entropy(logits, targets)  # 计算交叉熵损失
            
        return logits, loss, new_kv_cache

    def generate(self, idx, max_new_tokens):
        """文本生成函数，使用KV缓存优化推理速度
        
        Args:
            idx: 初始token序列，形状为(batch_size, sequence_length)
            max_new_tokens: 最大生成token数量
            
        Returns:
            生成的完整token序列，形状为(batch_size, sequence_length + max_new_tokens)
        """
        # 初始化KV缓存为空
        kv_cache = None
        
        # 逐个生成新token
        for _ in range(max_new_tokens):
            # 前向传播，传入KV缓存以利用历史计算结果
            logits, _, kv_cache = self(idx, kv_cache=kv_cache)
            
            # 只关注最后一个token的预测结果
            logits = logits[:, -1, :]        # (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # 转换为概率分布
            # 从概率分布中采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 将新生成的token添加到序列中
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
# 初始化分词器并获取词汇表大小
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

# 创建模型实例并加载预训练权重
model = BabyGPT(vocab_size, block_size, n_embed).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 设置为评估模式

# 交互式文本生成主循环
while True:
    # 获取用户输入的提示词
    prompt = input("请输入文字: ")
    
    # 将提示词编码为token序列并添加批次维度
    prompt_tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    # 记录生成开始时间
    start_time = time.time()
    
    # 生成文本
    result = model.generate(prompt_tokens, max_new_token)
    
    # 记录生成结束时间
    end_time = time.time()

    # 计算生成性能指标
    elapsed_time = end_time - start_time                    # 总耗时
    tokens_per_second = max_new_token / elapsed_time        # 生成速度(tokens/s)

    # 解码生成结果并打印
    print(tokenizer.decode(result[0].tolist()))
    print(f"> 生成速度: {tokens_per_second:.2f} tokens/s")  # 显示生成速度
    print('-'*10)  # 分隔线