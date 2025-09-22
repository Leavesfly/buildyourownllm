import random

# 设置随机种子以确保结果可复现
random.seed(42) # 去掉此行，获得随机结果

# 设置生成文本的提示词和最大生成token数
prompt = "春江"
max_new_token = 100

# 读取训练数据
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 构建词汇表
chars = sorted(list(set(text)))  # 获取所有不重复的字符并排序
vocab_size = len(chars)  # 词汇表大小
stoi = { ch:i for i,ch in enumerate(chars) }  # 字符到索引的映射
itos = { i:ch for i,ch in enumerate(chars) }  # 索引到字符的映射
encode = lambda s: [stoi[c] for c in s]  # 编码函数：将字符串转换为索引列表
decode = lambda l: ''.join([itos[i] for i in l])  # 解码函数：将索引列表转换为字符串

# 创建转移概率矩阵，记录每个词的下一个词的出现次数
# 矩阵维度为 vocab_size * vocab_size
transition = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]

# 统计文本中字符转移次数
for i in range(len(text) - 1):
    current_token_id = encode(text[i])[0]  # 当前字符的索引
    next_token_id = encode(text[i + 1])[0]  # 下一个字符的索引
    transition[current_token_id][next_token_id] += 1  # 更新转移计数

# 初始化生成序列
generated_token = encode(prompt)  # 将提示词编码为索引序列
current_token_id = generated_token[-1]  # 获取最后一个字符作为当前字符

# 生成新文本
for i in range(max_new_token):
    # 获取当前字符到所有其他字符的转移计数
    logits = transition[current_token_id]
    # 计算总和，避免除零错误
    total = max(sum(logits),1)
    # 将计数转换为概率（归一化）
    logits = [logit / total for logit in logits]
    # 根据概率分布随机采样下一个字符
    next_token_id = random.choices(range(vocab_size), weights=logits, k=1)[0]
    # 将新生成的字符添加到序列中
    generated_token.append(next_token_id)
    # 更新当前字符
    current_token_id = next_token_id

# 解码并打印生成的文本
print(decode(generated_token))