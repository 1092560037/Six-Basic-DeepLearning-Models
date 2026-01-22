# 深度学习模型 Docker 部署项目

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.9-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

六个经典深度学习模型的完整训练和Docker部署方案，用于花卉图像分类任务。

## 项目概述

本项目包含以下六个经典深度学习模型的完整实现:

| 模型 | 年份 | 参数量 | 输入尺寸 | 特点 |
|------|------|--------|---------|------|
| **LeNet-5** | 1998 | ~60K | 32×32 | 最早的CNN，简单高效 |
| **AlexNet** | 2012 | ~60M | 224×224 | ImageNet冠军，深度学习革命 |
| **VGGNet** | 2014 | ~138M | 224×224 | 简洁的架构设计 |
| **GoogLeNet** | 2014 | ~7M | 224×224 | Inception模块 |
| **ResNet** | 2015 | ~21M | 224×224 | 残差连接突破 |
| **DenseNet** | 2017 | ~8M | 224×224 | 密集连接网络 |

## 数据集

使用花卉数据集(Flowers)，包含5个类别:
- 🌼 Daisy (雏菊)
- 🌻 Dandelion (蒲公英)
- 🌹 Rose (玫瑰)
- 🌻 Sunflower (向日葵)
- 🌷 Tulip (郁金香)

## 项目结构

```
Model test/
├── 📁 LeNet5/              LeNet-5模型实现
├── 📁 AlexNet/             AlexNet模型实现
├── 📁 VggNet/              VGGNet模型实现
├── 📁 GoogLeNet/           GoogLeNet模型实现
├── 📁 ResNet/              ResNet模型实现
├── 📁 DenseNet/            DenseNet模型实现
├── 📁 Flowers/             花卉数据集
├── 📁 weights/             模型权重目录
├── 📄 *.pth                训练好的模型权重
├── 📄 class_indices.json   类别索引
│
├── 🐳 Docker部署文件
│   ├── Dockerfile          CPU版本
│   ├── Dockerfile.gpu      GPU版本
│   ├── docker-compose.yml  编排配置
│   ├── .dockerignore       忽略文件
│   └── requirements.txt    Python依赖
│
├── 🚀 服务和脚本
│   ├── app.py              Flask API服务
│   ├── test_api.py         API测试脚本
│   ├── start.sh            Linux/Mac启动脚本
│   └── start.bat           Windows启动脚本
│
└── 📚 文档
    ├── README.md           本文件
    ├── QUICKSTART.md       快速开始
    └── DOCKER_DEPLOYMENT.md 详细部署文档
```

## 快速开始

### ⚡ 5分钟快速部署

#### Windows
```cmd
start.bat
# 选择选项 1 (CPU版本)
```

#### Linux/Mac
```bash
chmod +x start.sh
./start.sh
# 选择选项 1 (CPU版本)
```

#### 使用Docker Compose
```bash
docker-compose up -d model-inference-cpu
```

详细步骤请查看 **[快速开始指南](QUICKSTART.md)**

### 🧪 测试API

```bash
# 健康检查
curl http://localhost:5000/health

# 预测图像
curl -X POST http://localhost:5000/predict \
  -F "model=alexnet" \
  -F "file=@tulip.jpg"

# 运行完整测试
python test_api.py
```

## API文档

### 可用端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 服务健康检查 |
| `/models` | GET | 列出所有可用模型 |
| `/predict` | POST | 单图像预测 |
| `/batch_predict` | POST | 批量图像预测 |
| `/compare_models` | POST | 使用所有模型对比预测 |

### 使用示例

#### Python
```python
import requests

files = {'file': open('tulip.jpg', 'rb')}
data = {'model': 'alexnet'}

response = requests.post('http://localhost:5000/predict',
                        files=files, data=data)
result = response.json()

print(f"预测: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.2%}")
```

#### cURL
```bash
curl -X POST http://localhost:5000/predict \
  -F "model=densenet" \
  -F "file=@flower.jpg"
```

完整API文档请查看 **[部署文档](DOCKER_DEPLOYMENT.md)**

## 功能特性

### ✨ 核心功能

- ✅ **6个经典模型**: LeNet5, AlexNet, VGG, GoogLeNet, ResNet, DenseNet
- ✅ **RESTful API**: 标准HTTP接口，易于集成
- ✅ **批量预测**: 支持一次处理多张图像
- ✅ **模型对比**: 自动使用所有模型进行对比测试
- ✅ **CPU/GPU支持**: 灵活选择运行设备
- ✅ **Docker部署**: 一键部署，环境隔离
- ✅ **完整测试**: 提供测试脚本和示例

### 🎯 技术栈

- **深度学习框架**: PyTorch 2.0
- **Web框架**: Flask
- **容器化**: Docker & Docker Compose
- **图像处理**: Pillow, OpenCV
- **数据处理**: NumPy, scikit-learn

## 部署选项

### CPU版本 (推荐开始)
```bash
docker-compose up -d model-inference-cpu
# 访问: http://localhost:5000
```

### GPU版本 (需要NVIDIA GPU)
```bash
docker-compose --profile gpu up -d model-inference-gpu
# 访问: http://localhost:5001
```

### 生产环境部署

生产环境部署建议请查看 **[部署文档 - 生产环境部署](DOCKER_DEPLOYMENT.md#生产环境部署建议)**

## 模型性能

以下是各模型在验证集上的性能表现:

| 模型 | 准确率 | 推理速度 | 内存占用 |
|------|--------|---------|---------|
| LeNet-5 | ~75% | 最快 | 最低 |
| AlexNet | ~85% | 快 | 中等 |
| VGGNet | ~90% | 慢 | 高 |
| GoogLeNet | ~88% | 中等 | 低 |
| ResNet-34 | ~92% | 中等 | 中等 |
| DenseNet-121 | ~94% | 中等 | 中等 |

*注: 具体性能取决于硬件配置和数据集*

## 训练模型

每个模型目录包含完整的训练代码:

```bash
# 以AlexNet为例
cd AlexNet
python train_v2.py --data-path ../Flowers --epochs 30
```

训练脚本支持的参数:
- `--data-path`: 数据集路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--lr`: 学习率
- `--device`: 设备(cuda/cpu)

## 目录说明

### 各模型目录结构
```
ModelName/
├── model.py        模型定义
├── train_v2.py     训练脚本
├── predict_v2.py   预测脚本
└── utils.py        工具函数 (部分模型)
```

## 常见问题

### Q: 如何切换使用不同的模型?
A: 在API请求中指定 `model` 参数，例如: `model=resnet`

### Q: 支持哪些图像格式?
A: 支持 JPG, PNG, JPEG, GIF, BMP, WEBP

### Q: 如何提高预测速度?
A:
1. 使用GPU版本
2. 预加载常用模型(修改app.py)
3. 使用较小的模型(如MobileNet, 需自行添加)

### Q: 内存不足怎么办?
A:
- 使用CPU版本
- 只加载需要的模型
- 减小batch_size

### Q: 如何添加新的模型?
A: 参考[部署文档 - 添加新模型](DOCKER_DEPLOYMENT.md#添加新模型)

## 文档导航

- 📖 **[快速开始](QUICKSTART.md)** - 5分钟快速部署
- 📖 **[Docker部署文档](DOCKER_DEPLOYMENT.md)** - 完整部署指南
- 📖 **[API测试脚本](test_api.py)** - API使用示例

## 系统要求

### 最低要求
- CPU: 2核
- 内存: 4GB
- 磁盘: 8GB

### 推荐配置
- CPU: 4核+
- 内存: 8GB+
- 磁盘: 20GB+
- GPU: NVIDIA GPU with 4GB+ VRAM (可选)

## 开发和贡献

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 运行开发服务器
python app.py

# 运行测试
python test_api.py
```

## 许可证

请根据您的项目添加适当的许可证。

## 致谢

- PyTorch团队提供的深度学习框架
- 各个模型的原始论文作者
- Flowers数据集提供者

## 更新日志

### v1.0.0 (2026-01-22)
- ✅ 初始发布
- ✅ 支持6个经典模型
- ✅ Docker完整部署方案
- ✅ RESTful API服务
- ✅ 完整的文档和测试

---

**快速链接**:
- [快速开始](QUICKSTART.md) | [完整文档](DOCKER_DEPLOYMENT.md) | [测试脚本](test_api.py)

**如有问题，欢迎提Issue!** 🎉
