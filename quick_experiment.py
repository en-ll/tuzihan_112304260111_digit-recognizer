import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_train=True, use_augmentation=False):
        print(f"正在读取 {csv_file}...")
        self.data = pd.read_csv(csv_file)
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train
        print(f"加载完成，共 {len(self.data)} 条数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_train:
            label = self.data.iloc[idx, 0]
            pixels = self.data.iloc[idx, 1:].values.astype(np.float32)
            pixels = pixels / 255.0
            pixels = torch.tensor(pixels).view(1, 28, 28)
            if self.use_augmentation:
                pixels = train_transform(pixels)
            return pixels, torch.tensor(label, dtype=torch.long)
        else:
            pixels = self.data.iloc[idx].values.astype(np.float32)
            pixels = pixels / 255.0
            pixels = torch.tensor(pixels).view(1, 28, 28)
            return pixels

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)
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

def train_model(optimizer_type, lr, batch_size, use_augmentation, use_early_stopping, num_epochs=15):
    print(f"\n{'='*60}")
    print(f"训练配置:")
    print(f"  优化器: {optimizer_type}, 学习率: {lr}, Batch Size: {batch_size}")
    print(f"  数据增强: {'是' if use_augmentation else '否'}, Early Stopping: {'是' if use_early_stopping else '否'}")
    print(f"{'='*60}")
    
    full_dataset = MNISTDataset('train.csv/train.csv', is_train=True, use_augmentation=use_augmentation)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    
    model = DeepCNN()
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    train_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 5
    early_stop_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if use_early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f'Early Stopping at Epoch {epoch+1}')
                    break
    
    train_time = time.time() - start_time
    
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
    train_acc = 100 * train_correct / train_total
    
    result = {
        'optimizer': optimizer_type,
        'lr': lr,
        'batch_size': batch_size,
        'use_augmentation': use_augmentation,
        'use_early_stopping': use_early_stopping,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch if use_early_stopping else num_epochs,
        'train_time': train_time,
        'train_losses': train_losses,
        'final_epoch': epoch + 1
    }
    
    print(f"\n训练完成! 耗时: {train_time:.2f}秒, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    return result

print("="*60)
print("开始4组对比实验")
print("="*60)

result1 = train_model('SGD', 0.01, 64, False, False, num_epochs=15)
result2 = train_model('Adam', 0.001, 64, False, False, num_epochs=15)
result3 = train_model('Adam', 0.001, 128, False, True, num_epochs=15)
result4 = train_model('Adam', 0.001, 64, True, True, num_epochs=15)

results = [result1, result2, result3, result4]
with open('experiment_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*60)
print("所有实验完成！")
print("="*60)

plt.figure(figsize=(12, 8))
epochs_range = range(1, len(result1['train_losses']) + 1)

plt.plot(epochs_range, result1['train_losses'], 'b-', label='Exp1 (SGD)', linewidth=2)
plt.plot(epochs_range, result2['train_losses'], 'g-', label='Exp2 (Adam)', linewidth=2)
plt.plot(epochs_range, result3['train_losses'], 'r-', label='Exp3 (Adam+ES)', linewidth=2)
plt.plot(epochs_range, result4['train_losses'], 'm-', label='Exp4 (Adam+DA+ES)', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss Curves Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
print("Loss曲线图已保存到 loss_curves.png")

print("\n" + "="*60)
print("实验结果汇总表")
print("="*60)
print(f"{'Exp':<6} {'Optimizer':<8} {'LR':<8} {'BS':<6} {'DA':<6} {'ES':<6} {'Train Acc':<12} {'Val Acc':<12} {'Best Loss':<12} {'Epoch':<6}")
print("-" * 100)
for i, r in enumerate(results, 1):
    print(f"Exp{i:<4} {r['optimizer']:<8} {r['lr']:<8} {r['batch_size']:<6} {'Y' if r['use_augmentation'] else 'N':<6} {'Y' if r['use_early_stopping'] else 'N':<6} {r['train_acc']:.2f}%{'':<6} {r['val_acc']:.2f}%{'':<6} {r['best_val_loss']:.4f}{'':<6} {r['final_epoch']}")
print("="*60)
