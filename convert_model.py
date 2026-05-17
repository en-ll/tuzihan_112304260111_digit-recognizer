import torch
import torch.nn as nn
import sys
import os

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

print("步骤1: 加载PyTorch模型...")
try:
    model = DeepCNN()
    state_dict = torch.load('model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    sys.exit(1)

print("\n步骤2: 创建测试输入...")
try:
    dummy_input = torch.randn(1, 1, 28, 28)
    print("✓ 测试输入创建成功")
except Exception as e:
    print(f"✗ 测试输入创建失败: {e}")
    sys.exit(1)

print("\n步骤3: 转换为ONNX格式...")
try:
    torch.onnx.export(
        model,
        dummy_input,
        'mnist_model.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("✓ ONNX模型导出成功")
except Exception as e:
    print(f"✗ ONNX导出失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤4: 验证ONNX文件...")
if os.path.exists('mnist_model.onnx'):
    size = os.path.getsize('mnist_model.onnx')
    print(f"✓ ONNX文件已创建，大小: {size / 1024:.2f} KB")
else:
    print("✗ ONNX文件未创建")
    sys.exit(1)

print("\n✓ 所有步骤完成！")
print("输出文件: mnist_model.onnx")