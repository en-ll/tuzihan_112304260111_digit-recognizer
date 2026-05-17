import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import time

# 数据增强变换
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

print("开始执行脚本...")

# 自定义数据集类
class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        print(f"正在读取 {csv_file}...")
        # 读取整个CSV文件，确保读取所有行
        self.data = pd.read_csv(csv_file)
        self.is_train = is_train
        print(f"加载完成，共 {len(self.data)} 条数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_train:
            # 训练数据包含label列
            label = self.data.iloc[idx, 0]
            pixels = self.data.iloc[idx, 1:].values.astype(np.float32)
            # 归一化像素值到0-1
            pixels = pixels / 255.0
            # 重塑为(1, 28, 28)的张量
            pixels = torch.tensor(pixels).view(1, 28, 28)
            # 应用数据增强
            pixels = train_transform(pixels)
            return pixels, torch.tensor(label, dtype=torch.long)
        else:
            # 测试数据没有label列
            pixels = self.data.iloc[idx].values.astype(np.float32)
            # 归一化像素值到0-1
            pixels = pixels / 255.0
            # 重塑为(1, 28, 28)的张量
            pixels = torch.tensor(pixels).view(1, 28, 28)
            return pixels

# 1. 准备数据
print("\n1. 准备数据...")
start_time = time.time()

# 加载训练数据
train_dataset = MNISTDataset('train.csv/train.csv', is_train=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 加载测试数据
test_dataset = MNISTDataset('test.csv/test.csv', is_train=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

print(f"训练数据大小: {len(train_dataset)}")
print(f"测试数据大小: {len(test_dataset)}")
print(f"数据准备完成，耗时: {time.time() - start_time:.2f}秒")

# 2. 定义模型
print("\n2. 定义模型...")
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # 卷积块1
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 卷积块2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 卷积块1: (batch_size, 1, 28, 28) -> (batch_size, 64, 14, 14)
        x = self.pool(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))
        # 卷积块2: (batch_size, 64, 14, 14) -> (batch_size, 128, 7, 7)
        x = self.pool(self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(x)))))))
        # 展平
        x = x.view(-1, 128 * 7 * 7)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = DeepCNN()
print("模型定义完成")

# 3. 定义损失函数和优化器
print("\n3. 定义损失函数和优化器...")
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
print("损失函数和优化器定义完成")

# 4. 训练模型
print("\n4. 开始训练...", flush=True)
start_time = time.time()
num_epochs = 15  # 充分训练
best_acc = 0.0
patience = 5
early_stop_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch+1}/{num_epochs}, Learning Rate: {optimizer.param_groups[0]['lr']}", flush=True)
    
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if (i+1) % 100 == 0:  # 每100个批次打印一次
            print(f'Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}', flush=True)
            running_loss = 0.0
    
    # 更新学习率
    scheduler.step()
    
    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total >= 2000:  # 验证更多样本
                break
    current_acc = 100 * correct / total
    print(f'当前验证准确率: {current_acc:.2f}%', flush=True)
    
    # 检查是否是最佳模型
    if current_acc > best_acc:
        best_acc = current_acc
        best_model_state = model.state_dict().copy()
        early_stop_counter = 0
        print(f'保存最佳模型，验证准确率: {best_acc:.2f}%', flush=True)
    else:
        early_stop_counter += 1
        print(f'早停计数: {early_stop_counter}/{patience}', flush=True)
        if early_stop_counter >= patience:
            print(f'触发早停！最佳验证准确率: {best_acc:.2f}%', flush=True)
            break

# 加载最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f'加载最佳模型，最终验证准确率: {best_acc:.2f}%', flush=True)

# 保存模型权重文件
torch.save(model.state_dict(), 'model.pth')
print("模型权重文件 model.pth 已保存", flush=True)

print(f"\n训练完成，耗时: {time.time() - start_time:.2f}秒", flush=True)

# 5. 生成sample_submission
print("\n5. 生成sample_submission...")
predictions = []
with torch.no_grad():
    for i, images in enumerate(test_loader):
        print(f"处理批次 {i+1}/{len(test_loader)}")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.numpy())

print(f"总预测数: {len(predictions)}")

# 创建提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})

submission.to_csv('sample_submission.csv', index=False)
print("sample_submission.csv 生成完成")
print(f"提交文件包含 {len(submission)} 条记录")
print("\n脚本执行完成！")