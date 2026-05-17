"""
手写数字识别 - 命令行工具
使用方法:
  python predict.py                    # 使用随机测试数据
  python predict.py image.png          # 识别指定图片
  python predict.py                    # 然后按提示输入测试
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

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

def load_model():
    model = DeepCNN()
    model.load_state_dict(torch.load('model.pth', map_location='cpu', weights_only=True))
    model.eval()
    return model

def predict(model, image_path=None):
    if image_path:
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
    else:
        image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        image = Image.fromarray(image)
    
    image_np = np.array(image)
    image_np = (255 - image_np) / 255.0
    
    tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
    
    predicted = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    return predicted, confidence, probabilities

if __name__ == '__main__':
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully!")
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nPredicting for: {image_path}")
        pred, conf, probs = predict(model, image_path)
        print(f"\nResult: {pred}")
        print(f"Confidence: {conf:.2%}")
        print("\nAll probabilities:")
        for i in range(10):
            bar = '#' * int(probs[i] * 30)
            print(f"  {i}: {probs[i]:.4f} {bar}")
    else:
        print("\nRunning test with random data...")
        pred, conf, probs = predict(model)
        print(f"\nResult: {pred}")
        print(f"Confidence: {conf:.2%}")
        print("\nAll probabilities:")
        for i in range(10):
            bar = '#' * int(probs[i] * 30)
            print(f"  {i}: {probs[i]:.4f} {bar}")
        
        print("\n" + "="*50)
        print("USAGE:")
        print("  python predict.py image.png  # Predict an image")
        print("  python predict.py            # Run with random data")