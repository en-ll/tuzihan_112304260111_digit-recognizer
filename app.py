import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

import torch
import torch.nn as nn
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import numpy as np
print(f"NumPy version: {np.__version__}")

from PIL import Image
print("PIL imported")

import gradio as gr
print("Gradio imported")

import pandas as pd
from datetime import datetime

# 全局历史记录列表
history_records = []
history_df = pd.DataFrame(columns=["序号", "时间", "输入类型", "识别结果", "置信度"])

# 定义模型结构
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

# 加载模型
print("Loading model...")
model = DeepCNN()
model.load_state_dict(torch.load('model.pth', map_location='cpu', weights_only=True))
model.eval()
print("Model loaded successfully!")

# 推理函数
def predict_digit(image):
    """
    数据处理流程：
    1. Web Canvas (500x500 RGBA) -> 缩放至28x28并灰度化
    2. 必须反色！匹配MNIST数据集的黑底白字特征
    3. 转化为PyTorch张量：[1, 1, 28, 28]，数值归一化(0-1)
    """
    # 处理Gradio Sketchpad返回的字典格式
    if isinstance(image, dict):
        if 'image' in image:
            image = image['image']
        elif 'composite' in image:
            image = image['composite']
    
    # 如果是PIL图像，转换为numpy数组
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 如果是RGBA格式，转换为灰度
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    
    # 如果是彩色图像，转换为灰度
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.mean(image, axis=2)
    
    # 缩放至28x28并灰度化
    image_pil = Image.fromarray(image).resize((28, 28))
    image = np.array(image_pil)
    
    # 必须反色！匹配MNIST数据集的黑底白字特征
    image = 255 - image
    
    # 转化为PyTorch张量：[1, 1, 28, 28]，数值归一化(0-1)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor / 255.0
    
    # 执行推理并应用Softmax
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
    
    # 返回字典格式: {标签: 置信度}
    result = {}
    for i in range(10):
        result[str(i)] = probabilities[0][i].item()
    
    return result

# 手写识别函数
def predict_from_sketch(sketch, history_table):
    """
    从手写画板识别数字
    """
    result = predict_digit(sketch)
    
    # 添加到历史记录
    if result:
        current_time = datetime.now().strftime("%H:%M:%S")
        predicted_digit = max(result, key=result.get)
        confidence = result[predicted_digit]
        
        new_record = {
            "序号": len(history_records) + 1,
            "时间": current_time,
            "输入类型": "✏️ 手写",
            "识别结果": predicted_digit,
            "置信度": f"{confidence:.1%}"
        }
        history_records.append(new_record)
        history_df = pd.DataFrame(history_records)
        
        return result, history_df
    return result, history_table

# 图片上传识别函数
def predict_from_image(image, history_table):
    """
    从上传的图片识别数字
    image: PIL Image 或 numpy array
    """
    if image is None:
        return {}, history_table
    
    # 如果是numpy数组，直接使用
    if isinstance(image, np.ndarray):
        pass  # 已经是numpy数组
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # 转换为灰度并处理
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        if image.shape[2] == 3:  # RGB
            image = np.mean(image, axis=2)
    
    # 缩放至28x28
    image_pil = Image.fromarray(image).resize((28, 28))
    image = np.array(image_pil)
    
    # 反色
    image = 255 - image
    
    # 转为张量
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor / 255.0
    
    # 推理
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
    
    result = {}
    for i in range(10):
        result[str(i)] = probabilities[0][i].item()
    
    # 添加到历史记录
    if result:
        current_time = datetime.now().strftime("%H:%M:%S")
        predicted_digit = max(result, key=result.get)
        confidence = result[predicted_digit]
        
        new_record = {
            "序号": len(history_records) + 1,
            "时间": current_time,
            "输入类型": "📷 图片",
            "识别结果": predicted_digit,
            "置信度": f"{confidence:.1%}"
        }
        history_records.append(new_record)
        history_df = pd.DataFrame(history_records)
        
        return result, history_df
    return result, history_table

# 清空历史记录函数
def clear_history():
    global history_records, history_df
    history_records = []
    history_df = pd.DataFrame(columns=["序号", "时间", "输入类型", "识别结果", "置信度"])
    return history_df

# 创建界面
with gr.Blocks() as demo:
    gr.Markdown("# 🔢 手写数字识别系统")
    gr.Markdown("使用您自己训练的CNN模型进行数字识别")
    
    with gr.Tabs():
        # 手写识别选项卡
        with gr.TabItem("✏️ 手写识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    sketch_input = gr.Sketchpad(label="在下方画布上绘制数字 (0-9)")
                    sketch_button = gr.Button("🔍 识别", variant="primary", size="lg")
                    sketch_output = gr.Label(num_top_classes=3, label="识别结果")
                    
                    # 历史记录
                    with gr.Accordion("📜 历史识别记录（点击展开）", open=False):
                        history_table = gr.DataFrame(
                            headers=["序号", "时间", "输入类型", "识别结果", "置信度"],
                            label="",
                            show_label=False
                        )
                        clear_history_button = gr.Button("🗑️ 清空历史记录", variant="secondary")
                        
                        clear_history_button.click(
                            fn=clear_history,
                            outputs=[history_table]
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 使用说明")
                    gr.Markdown("""
                    1. 在左侧画布上绘制数字（0-9）
                    2. 点击"🔍 识别"按钮获取结果
                    3. 查看识别结果和置信度
                    4. 历史记录会自动保存
                    5. 可点击"📜 历史识别记录"查看历史
                    """)
            
            # 事件绑定
            sketch_button.click(
                fn=predict_from_sketch,
                inputs=[sketch_input, history_table],
                outputs=[sketch_output, history_table]
            )
        
        # 图片识别选项卡
        with gr.TabItem("📷 图片识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="上传包含数字的图片", type="numpy")
                    image_button = gr.Button("🔍 识别", variant="primary", size="lg")
                    image_output = gr.Label(num_top_classes=3, label="识别结果")
                    
                    # 历史记录
                    with gr.Accordion("📜 历史识别记录（点击展开）", open=False):
                        image_history_table = gr.DataFrame(
                            headers=["序号", "时间", "输入类型", "识别结果", "置信度"],
                            label="",
                            show_label=False
                        )
                        image_clear_history_button = gr.Button("🗑️ 清空历史记录", variant="secondary")
                        
                        image_clear_history_button.click(
                            fn=clear_history,
                            outputs=[image_history_table]
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 使用说明")
                    gr.Markdown("""
                    1. 上传包含数字的图片（JPG、PNG等）
                    2. 点击"🔍 识别"按钮获取结果
                    3. 系统会自动处理并识别数字
                    4. 支持多种图片格式
                    5. 可点击"📜 历史识别记录"查看历史
                    """)
            
            # 事件绑定
            image_button.click(
                fn=predict_from_image,
                inputs=[image_input, image_history_table],
                outputs=[image_output, image_history_table]
            )

# 启动应用
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)
