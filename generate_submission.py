import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

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
        x = self.pool(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))
        x = self.pool(self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(x)))))))
        x = x.view(-1, 128 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_train:
            label = self.data.iloc[idx, 0]
            pixels = self.data.iloc[idx, 1:].values.astype(np.float32)
            pixels = pixels / 255.0
            pixels = torch.tensor(pixels).view(1, 28, 28)
            return pixels, torch.tensor(label, dtype=torch.long)
        else:
            pixels = self.data.iloc[idx].values.astype(np.float32)
            pixels = pixels / 255.0
            pixels = torch.tensor(pixels).view(1, 28, 28)
            return pixels

# 加载模型
print("加载模型...")
model = DeepCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 加载测试数据
print("加载测试数据...")
test_dataset = MNISTDataset('test.csv/test.csv', is_train=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 生成预测
print("生成预测...")
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