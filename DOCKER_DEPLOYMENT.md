# 深度学习模型Docker部署指南

本项目包含6个深度学习模型的Docker部署方案: **LeNet5**, **AlexNet**, **VGGNet**, **GoogLeNet**, **ResNet**, **DenseNet**

## 目录结构

```
Model test/
├── AlexNet/              # AlexNet模型代码
├── LeNet5/               # LeNet5模型代码
├── VggNet/               # VGGNet模型代码
├── GoogLeNet/            # GoogLeNet模型代码
├── ResNet/               # ResNet模型代码
├── DenseNet/             # DenseNet模型代码
├── Flowers/              # 数据集
├── weights/              # 模型权重目录
├── *.pth                 # 各模型权重文件
├── class_indices.json    # 类别索引
├── app.py                # Flask API服务
├── requirements.txt      # Python依赖
├── Dockerfile            # CPU版Docker配置
├── Dockerfile.gpu        # GPU版Docker配置
├── docker-compose.yml    # Docker Compose配置
└── .dockerignore         # Docker忽略文件
```

## 前置要求

### 基础要求
- Docker 20.10+
- Docker Compose 1.29+
- 至少 8GB 可用磁盘空间

### GPU支持（可选）
- NVIDIA GPU
- NVIDIA Driver 470+
- NVIDIA Docker Runtime (nvidia-docker2)

## 快速开始

### 1. CPU版本部署

```bash
# 构建并启动服务
docker-compose up -d model-inference-cpu

# 查看日志
docker-compose logs -f model-inference-cpu

# 检查服务状态
curl http://localhost:5000/health
```

### 2. GPU版本部署

```bash
# 启动GPU服务
docker-compose --profile gpu up -d model-inference-gpu

# 查看日志
docker-compose logs -f model-inference-gpu

# 检查GPU使用情况
docker exec dl-models-gpu nvidia-smi
```

## API使用说明

### 1. 健康检查

```bash
curl http://localhost:5000/health
```

响应示例:
```json
{
  "status": "healthy",
  "device": "cpu",
  "models_loaded": ["alexnet"]
}
```

### 2. 列出所有模型

```bash
curl http://localhost:5000/models
```

响应示例:
```json
{
  "models": ["lenet5", "alexnet", "vggnet", "googlenet", "resnet", "densenet"],
  "loaded": ["alexnet"]
}
```

### 3. 单图像预测

```bash
# 使用文件上传
curl -X POST http://localhost:5000/predict \
  -F "model=alexnet" \
  -F "file=@tulip.jpg"
```

响应示例:
```json
{
  "model": "alexnet",
  "predicted_class": "tulip",
  "predicted_index": 4,
  "confidence": 0.9876,
  "all_probabilities": {
    "daisy": 0.0012,
    "dandelion": 0.0034,
    "rose": 0.0045,
    "sunflower": 0.0033,
    "tulip": 0.9876
  }
}
```

### 4. 批量预测

```bash
curl -X POST http://localhost:5000/batch_predict \
  -F "model=resnet" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### 5. 模型对比

使用所有6个模型对同一张图像进行预测:

```bash
curl -X POST http://localhost:5000/compare_models \
  -F "file=@tulip.jpg"
```

响应示例:
```json
{
  "lenet5": {
    "predicted_class": "tulip",
    "confidence": 0.92,
    "top3": [["tulip", 0.92], ["rose", 0.05], ["sunflower", 0.02]]
  },
  "alexnet": {
    "predicted_class": "tulip",
    "confidence": 0.98,
    "top3": [["tulip", 0.98], ["rose", 0.01], ["daisy", 0.005]]
  },
  ...
}
```

## Python客户端示例

```python
import requests

# 单图像预测
url = "http://localhost:5000/predict"
files = {'file': open('tulip.jpg', 'rb')}
data = {'model': 'alexnet'}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"预测类别: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.4f}")

# 模型对比
url = "http://localhost:5000/compare_models"
files = {'file': open('tulip.jpg', 'rb')}

response = requests.post(url, files=files)
results = response.json()

for model_name, prediction in results.items():
    print(f"{model_name}: {prediction['predicted_class']} ({prediction['confidence']:.4f})")
```

## 支持的模型

| 模型名称 | 模型代码 | 权重文件 | 输入尺寸 |
|---------|---------|---------|---------|
| LeNet5 | `lenet5` | LeNet5_v2.pth | 32x32 |
| AlexNet | `alexnet` | AlexNet_v2.pth | 224x224 |
| VGGNet | `vggnet` | vgg16_transfer_best_v2.pth | 224x224 |
| GoogLeNet | `googlenet` | googleNet_v2.pth | 224x224 |
| ResNet | `resnet` | resNet34_v2.pth | 224x224 |
| DenseNet | `densenet` | densenet121-a639ec97_v2.pth | 224x224 |

## 类别列表

本项目使用花卉数据集，包含5个类别:
- daisy (雏菊)
- dandelion (蒲公英)
- rose (玫瑰)
- sunflower (向日葵)
- tulip (郁金香)

## 常见操作

### 停止服务

```bash
# 停止CPU服务
docker-compose stop model-inference-cpu

# 停止GPU服务
docker-compose --profile gpu stop model-inference-gpu

# 停止所有服务
docker-compose down
```

### 查看日志

```bash
# 实时查看日志
docker-compose logs -f model-inference-cpu

# 查看最近100行日志
docker-compose logs --tail=100 model-inference-cpu
```

### 重新构建镜像

```bash
# 重新构建CPU版本
docker-compose build model-inference-cpu

# 重新构建GPU版本
docker-compose build model-inference-gpu

# 强制重新构建（不使用缓存）
docker-compose build --no-cache model-inference-cpu
```

### 进入容器调试

```bash
# 进入容器
docker exec -it dl-models-cpu /bin/bash

# 在容器内测试Python
python -c "import torch; print(torch.__version__)"

# 测试模型加载
python -c "from AlexNet.model import AlexNet; print('Model loaded')"
```

## 性能优化

### 1. 预加载模型

修改 [app.py](app.py) 底部，取消注释预加载代码:

```python
if __name__ == '__main__':
    # 预加载常用模型
    load_model('alexnet')
    load_model('resnet')
    load_model('densenet')

    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 2. 使用Gunicorn

创建 `gunicorn_config.py`:

```python
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 120
```

修改 Dockerfile 的 CMD:

```dockerfile
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
```

添加到 requirements.txt:
```
gunicorn==21.2.0
```

### 3. 内存优化

如果内存有限，可以使用延迟加载策略（默认配置），只在请求时加载需要的模型。

## 故障排除

### 问题1: 模型权重文件找不到

**症状**: `FileNotFoundError: Weight file not found`

**解决方案**:
```bash
# 确认权重文件存在
ls -lh *.pth

# 检查docker-compose.yml中的volume映射
# 确保所有.pth文件都已映射
```

### 问题2: GPU不可用

**症状**: 容器内无法使用GPU

**解决方案**:
```bash
# 检查NVIDIA Docker运行时
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 检查docker-compose配置
# 确保deploy.resources.reservations.devices配置正确
```

### 问题3: 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小batch_size
- 使用CPU版本
- 只加载需要的模型

### 问题4: 依赖安装失败

**症状**: pip install 超时或失败

**解决方案**:
```dockerfile
# 在Dockerfile中添加国内镜像源
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 生产环境部署建议

### 1. 使用反向代理

使用Nginx作为反向代理:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

### 2. 添加认证

使用API密钥保护接口:

```python
# 在app.py中添加
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... existing code
```

### 3. 日志管理

配置日志输出:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### 4. 监控和告警

使用Prometheus + Grafana进行监控。

## 扩展功能

### 添加新模型

1. 在相应目录添加模型代码
2. 训练并保存权重文件
3. 在 `MODEL_CONFIGS` 中添加配置
4. 在 `load_model()` 函数中添加加载逻辑

### 支持更多图像格式

修改 `predict()` 函数以支持更多格式:

```python
# 支持多种图像格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

## 许可证

请根据您的项目添加适当的许可证信息。

## 联系方式

如有问题，请提交Issue或联系项目维护者。

---

**最后更新**: 2026-01-22
