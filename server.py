import sys
import torch
import torch.nn as nn
import numpy as np
from base64 import b64decode
from PIL import Image
import io

from flask import Flask, request, jsonify

app = Flask(__name__)

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

print("Loading model...", flush=True)
model = DeepCNN()

try:
    state_dict = torch.load('model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!", flush=True)
except Exception as e:
    print(f"Load error: {e}", flush=True)

model.eval()
print("Ready!", flush=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = image.resize((28, 28))
        image_np = np.array(image)
        image_np = (255 - image_np) / 255.0
        
        tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
        
        result = {
            'prediction': int(np.argmax(probabilities)),
            'confidence': float(np.max(probabilities)),
            'probabilities': {str(i): float(probabilities[i]) for i in range(10)}
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    print("Server started!", flush=True)