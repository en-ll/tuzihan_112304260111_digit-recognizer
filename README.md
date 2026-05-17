# 手写数字识别 - 使用说明

## 问题诊断

当前环境中存在PyTorch与NumPy的兼容性问题，导致模型无法正常加载。
错误信息：`Failed to initialize NumPy: module 'numpy._globals' has no attribute '_signature_descriptor'`

## 解决方案

请在您的本地终端中运行以下命令：

### 1. 检查Python环境

打开命令提示符（CMD）或PowerShell，运行：

```bash
cd F:\Trae\Digit Recognizer
F:\Trae\python\.venv\Scripts\python.exe --version
```

### 2. 测试模型加载

运行以下命令测试模型是否正常：

```bash
F:\Trae\python\.venv\Scripts\python.exe predict.py
```

如果成功，您应该看到：
```
Loading model...
Model loaded successfully!

Running test with random data...

Result: 随机数字
Confidence: XX.XX%

All probabilities:
  0: X.XXXX ############
  1: X.XXXX ############
  ...
```

### 3. 启动Web服务（可选）

如果命令行工具工作正常，您可以尝试启动Flask服务：

```bash
F:\Trae\python\.venv\Scripts\python.exe server.py
```

服务启动后，打开 `index.html` 文件即可使用网页界面。

### 4. 修复PyTorch环境（推荐）

如果遇到NumPy兼容性问题，您可以尝试：

```bash
# 方案1：降级NumPy
F:\Trae\python\.venv\Scripts\python.exe -m pip install "numpy<2.0"

# 方案2：重新安装PyTorch
F:\Trae\python\.venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio
F:\Trae\python\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 5. 使用命令行工具识别图片

如果您有一张手写数字图片，可以直接识别：

```bash
F:\Trae\python\.venv\Scripts\python.exe predict.py your_image.png
```

支持的图片格式：PNG, JPG, BMP等

## 文件说明

- `model.pth` - 您训练的DeepCNN模型权重
- `predict.py` - 命令行预测工具
- `server.py` - Flask后端服务
- `index.html` - 网页前端界面
- `test_model.py` - 模型测试脚本

## 技术规格

- 模型架构：DeepCNN（4层卷积 + 3层全连接）
- 输入尺寸：28x28灰度图像
- 输出：0-9数字分类
- 验证准确率：99.55%

## 常见问题

### Q: 报错"Failed to initialize NumPy"
A: 这是NumPy 2.x与PyTorch 2.11的兼容性问题，运行：
```bash
F:\Trae\python\.venv\Scripts\python.exe -m pip install "numpy<2.0"
```

### Q: 模型加载失败
A: 确保model.pth文件存在且完整，尝试重新训练模型。

### Q: Flask服务无法启动
A: 检查Flask是否已安装：
```bash
F:\Trae\python\.venv\Scripts\python.exe -m pip install flask
```

## 技术支持

如果问题仍然存在，请提供完整的错误信息以便进一步诊断。