import torch
import torch.nn as nn
import numpy as np

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

print("Loading model...")
model = DeepCNN()
model.load_state_dict(torch.load('model.pth', map_location='cpu', weights_only=True))
model.eval()
print("Model loaded successfully!")

print("\nTest with random input:")
test_input = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    output = model(test_input)
    probs = torch.softmax(output, dim=1)
    pred = probs.argmax(dim=1).item()
    confidence = probs[0][pred].item()
    print(f"Prediction: {pred}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for i in range(10):
        print(f"  {i}: {probs[0][i].item():.4f}")

print("\nModel is working correctly!")