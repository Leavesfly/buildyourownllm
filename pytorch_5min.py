import torch
from torch import nn
from torch.nn import functional as F

# 设置随机数种子，方便复现结果
torch.manual_seed(42)

# 判断环境中是否有GPU可用
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(f"Using {device} device")

# 1. 创建tensor演示
x = torch.tensor([1.0, 2.0, 3.0])  # 创建一维张量
y = torch.tensor([2.0, 4.0, 6.0])  # 创建一维张量

# 2. 基本运算演示
print(x + y)                # 加法: tensor([3., 6., 9.])
print(x * y)                # 点乘: tensor([2., 8., 18.])
print(torch.matmul(x, y))   # 矩阵乘法: tensor(28.)
print(x @ y)                # 另一种矩阵乘写法: tensor(28.)
print(x.shape)              # tensor的形状: torch.Size([3])

# 3. 定义模型
class SimpleNet(nn.Module):
    """简单神经网络模型"""
    
    def __init__(self):
        """初始化模型"""
        super().__init__()
        # 线性层，输入维度=1，输出维度=1
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入数据
        Returns:
            模型输出
        """
        return self.linear(x)

# 4. 生成训练数据
# 真实关系: y = 2x + 1
x_train = torch.rand(100, 1) * 10  # 生成 0-10 之间的随机数
y_train = 2 * x_train + 1 + torch.randn(100, 1) * 0.1  # 真实函数：y = 2x + 1 加上一些噪声
# 将数据移动到指定设备
x_train = x_train.to(device)
y_train = y_train.to(device)

# 5. 创建模型和优化器
model = SimpleNet().to(device)  # 创建模型并移动到指定设备
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器，学习率0.01
criterion = nn.MSELoss()  # 均方误差损失函数

# 6. 训练循环
epochs = 5000  # 训练轮数

print("\n训练开始...")
for epoch in range(epochs):
    # 前向传播，预测结果
    y_pred = model(x_train)
    
    # 计算预测值和真实值之间的损失
    loss = criterion(y_pred, y_train)
    
    # 反向传播，修改模型参数
    optimizer.zero_grad()  # 清除旧的梯度
    loss.backward()  # 计算新的梯度 
    optimizer.step()  # 更新参数：参数 -= 学习率 * 梯度
    
    # 每100轮打印一次训练信息
    if (epoch + 1) % 100 == 0:
        w = model.linear.weight.item()  # 获取权重参数
        b = model.linear.bias.item()  # 获取偏置参数
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, w: {w:.2f}, b: {b:.2f}')

# 7. 打印结果
w = model.linear.weight.item()  # 获取最终的权重参数
b = model.linear.bias.item()  # 获取最终的偏置参数
print(f'\n训练完成！')
print(f'学习到的函数: y = {w:.2f}x + {b:.2f}')
print(f'实际函数: y = 2.00x + 1.00')

# 8. 测试模型
test_x = torch.tensor([[0.0], [5.0], [10.0]]).to(device)  # 测试数据
with torch.no_grad():  # 关闭梯度计算，节省内存
    test_y = model(test_x)  # 模型预测
    print("\n预测结果：")
    for x, y in zip(test_x, test_y):
        print(f'x = {x.item():.1f}, y = {y.item():.2f}')