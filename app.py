"""
深度学习模型推理API服务
支持六个模型: LeNet5, AlexNet, VGGNet, GoogLeNet, ResNet, DenseNet
"""
import os
import json
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms, models
import io
import base64

# 导入各个模型
from LeNet5.net_v2 import LeNet
from AlexNet.model import AlexNet as AlexNetModel
from VggNet.model_v2 import vgg
from GoogLeNet.model_v2 import GoogLeNet
from ResNet.model_v2 import resnet34
from DenseNet.model import densenet121

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 加载类别索引
with open('class_indices.json', 'r', encoding='utf-8') as f:
    class_indict = json.load(f)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型配置
MODEL_CONFIGS = {
    'lenet5': {
        'weight_path': 'LeNet5_v2.pth',
        'num_classes': 5,
        'img_size': 32,
    },
    'alexnet': {
        'weight_path': 'AlexNet_v2.pth',
        'num_classes': 5,
        'img_size': 224,
    },
    'vggnet': {
        'weight_path': 'vgg16_transfer_best_v2.pth',
        'num_classes': 5,
        'img_size': 224,
    },
    'googlenet': {
        'weight_path': 'googleNet_v2.pth',
        'num_classes': 5,
        'img_size': 224,
    },
    'resnet': {
        'weight_path': 'resNet34_v2.pth',
        'num_classes': 5,
        'img_size': 224,
    },
    'densenet': {
        'weight_path': 'densenet121-a639ec97_v2.pth',
        'num_classes': 5,
        'img_size': 224,
    }
}

# 预加载所有模型
models_cache = {}

def load_model(model_name):
    """加载指定模型"""
    if model_name in models_cache:
        return models_cache[model_name]

    config = MODEL_CONFIGS.get(model_name.lower())
    if not config:
        raise ValueError(f"Model {model_name} not supported")

    # 创建模型
    if model_name.lower() == 'lenet5':
        model = LeNet()
    elif model_name.lower() == 'alexnet':
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, config['num_classes'])
    elif model_name.lower() == 'vggnet':
        model = vgg(model_name="vgg16", num_classes=config['num_classes'], init_weights=False)
    elif model_name.lower() == 'googlenet':
        model = GoogLeNet(num_classes=config['num_classes'], aux_logits=False)
    elif model_name.lower() == 'resnet':
        model = resnet34(num_classes=config['num_classes'])
    elif model_name.lower() == 'densenet':
        model = densenet121(num_classes=config['num_classes'])

    # 加载权重
    weight_path = config['weight_path']
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    models_cache[model_name] = model
    print(f"Model {model_name} loaded successfully")
    return model

def get_transform(img_size):
    """获取图像预处理transform"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_image(model, image, img_size):
    """对单张图像进行预测"""
    transform = get_transform(img_size)
    img_tensor = transform(image)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    with torch.no_grad():
        output = torch.squeeze(model(img_tensor)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()

    return predict_cla, predict.numpy().tolist()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'models_loaded': list(models_cache.keys())
    })

@app.route('/models', methods=['GET'])
def list_models():
    """列出所有支持的模型"""
    return jsonify({
        'models': list(MODEL_CONFIGS.keys()),
        'loaded': list(models_cache.keys())
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    预测接口
    POST /predict
    {
        "model": "alexnet",  # 模型名称
        "image": "base64_encoded_image" 或 上传文件
    }
    """
    try:
        # 获取模型名称
        model_name = request.form.get('model', 'alexnet').lower()
        if model_name not in MODEL_CONFIGS:
            return jsonify({
                'error': f'Model not supported. Available: {list(MODEL_CONFIGS.keys())}'
            }), 400

        # 获取图像
        if 'file' in request.files:
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        elif 'image' in request.form:
            # Base64编码的图像
            image_data = base64.b64decode(request.form['image'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

        # 加载模型
        model = load_model(model_name)
        config = MODEL_CONFIGS[model_name]

        # 预测
        predict_cla, probabilities = predict_image(model, image, config['img_size'])

        # 构建结果
        result = {
            'model': model_name,
            'predicted_class': class_indict[str(predict_cla)],
            'predicted_index': predict_cla,
            'confidence': probabilities[predict_cla],
            'all_probabilities': {
                class_indict[str(i)]: prob
                for i, prob in enumerate(probabilities)
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量预测接口
    支持多个图像同时预测
    """
    try:
        model_name = request.form.get('model', 'alexnet').lower()
        if model_name not in MODEL_CONFIGS:
            return jsonify({
                'error': f'Model not supported. Available: {list(MODEL_CONFIGS.keys())}'
            }), 400

        # 获取所有图像文件
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400

        # 加载模型
        model = load_model(model_name)
        config = MODEL_CONFIGS[model_name]

        # 批量预测
        results = []
        for idx, file in enumerate(files):
            image = Image.open(file.stream).convert('RGB')
            predict_cla, probabilities = predict_image(model, image, config['img_size'])

            results.append({
                'file_index': idx,
                'filename': file.filename,
                'predicted_class': class_indict[str(predict_cla)],
                'predicted_index': predict_cla,
                'confidence': probabilities[predict_cla]
            })

        return jsonify({
            'model': model_name,
            'total': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare_models', methods=['POST'])
def compare_models():
    """
    模型对比接口
    对同一张图像使用所有模型进行预测
    """
    try:
        # 获取图像
        if 'file' in request.files:
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

        # 使用所有模型预测
        results = {}
        for model_name in MODEL_CONFIGS.keys():
            try:
                model = load_model(model_name)
                config = MODEL_CONFIGS[model_name]
                predict_cla, probabilities = predict_image(model, image, config['img_size'])

                results[model_name] = {
                    'predicted_class': class_indict[str(predict_cla)],
                    'confidence': probabilities[predict_cla],
                    'top3': sorted(
                        [(class_indict[str(i)], prob) for i, prob in enumerate(probabilities)],
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                }
            except Exception as e:
                results[model_name] = {'error': str(e)}

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Deep Learning Models API Service")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Available models: {list(MODEL_CONFIGS.keys())}")
    print("=" * 60)

    # 预加载默认模型（可选）
    # load_model('alexnet')

    app.run(host='0.0.0.0', port=5000, debug=False)
