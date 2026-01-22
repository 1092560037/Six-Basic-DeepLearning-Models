"""
API测试脚本
用于测试Docker部署的深度学习模型推理服务
"""
import requests
import json
import os
from pathlib import Path

# 配置
API_URL = "http://localhost:5000"
TEST_IMAGE = "tulip.jpg"

def test_health():
    """测试健康检查"""
    print("\n" + "="*60)
    print("测试: 健康检查")
    print("="*60)

    response = requests.get(f"{API_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    return response.status_code == 200

def test_list_models():
    """测试模型列表"""
    print("\n" + "="*60)
    print("测试: 获取模型列表")
    print("="*60)

    response = requests.get(f"{API_URL}/models")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    return response.status_code == 200

def test_single_predict(model_name="alexnet"):
    """测试单图像预测"""
    print("\n" + "="*60)
    print(f"测试: 单图像预测 (模型: {model_name})")
    print("="*60)

    if not os.path.exists(TEST_IMAGE):
        print(f"错误: 测试图像不存在: {TEST_IMAGE}")
        return False

    files = {'file': open(TEST_IMAGE, 'rb')}
    data = {'model': model_name}

    response = requests.post(f"{API_URL}/predict", files=files, data=data)
    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"预测模型: {result['model']}")
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("\n所有类别概率:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    else:
        print(f"错误: {response.text}")

    return response.status_code == 200

def test_batch_predict(model_name="alexnet"):
    """测试批量预测"""
    print("\n" + "="*60)
    print(f"测试: 批量预测 (模型: {model_name})")
    print("="*60)

    if not os.path.exists(TEST_IMAGE):
        print(f"错误: 测试图像不存在: {TEST_IMAGE}")
        return False

    # 使用同一张图片模拟多个文件
    files = [
        ('files', ('image1.jpg', open(TEST_IMAGE, 'rb'), 'image/jpeg')),
        ('files', ('image2.jpg', open(TEST_IMAGE, 'rb'), 'image/jpeg')),
        ('files', ('image3.jpg', open(TEST_IMAGE, 'rb'), 'image/jpeg')),
    ]
    data = {'model': model_name}

    response = requests.post(f"{API_URL}/batch_predict", files=files, data=data)
    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"预测模型: {result['model']}")
        print(f"总数: {result['total']}")
        print("\n预测结果:")
        for item in result['results']:
            print(f"  文件 {item['file_index']}: {item['predicted_class']} (置信度: {item['confidence']:.4f})")
    else:
        print(f"错误: {response.text}")

    return response.status_code == 200

def test_compare_models():
    """测试模型对比"""
    print("\n" + "="*60)
    print("测试: 模型对比（所有6个模型）")
    print("="*60)

    if not os.path.exists(TEST_IMAGE):
        print(f"错误: 测试图像不存在: {TEST_IMAGE}")
        return False

    files = {'file': open(TEST_IMAGE, 'rb')}

    response = requests.post(f"{API_URL}/compare_models", files=files)
    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        results = response.json()
        print("\n所有模型预测结果:")
        print("-" * 80)
        print(f"{'模型':<15} {'预测类别':<15} {'置信度':<10} Top 3 预测")
        print("-" * 80)

        for model_name, prediction in results.items():
            if 'error' in prediction:
                print(f"{model_name:<15} 错误: {prediction['error']}")
            else:
                top3_str = ', '.join([f"{c}({p:.2f})" for c, p in prediction['top3']])
                print(f"{model_name:<15} {prediction['predicted_class']:<15} {prediction['confidence']:<10.4f} {top3_str}")
        print("-" * 80)
    else:
        print(f"错误: {response.text}")

    return response.status_code == 200

def test_all_models():
    """测试所有模型的单图像预测"""
    print("\n" + "="*60)
    print("测试: 所有模型单独预测")
    print("="*60)

    models = ["lenet5", "alexnet", "vggnet", "googlenet", "resnet", "densenet"]
    results = {}

    for model in models:
        print(f"\n测试模型: {model}")
        success = test_single_predict(model)
        results[model] = "成功" if success else "失败"

    print("\n" + "="*60)
    print("所有模型测试结果汇总:")
    print("="*60)
    for model, status in results.items():
        print(f"{model:<15}: {status}")

    return all(s == "成功" for s in results.values())

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("深度学习模型API测试")
    print("="*60)
    print(f"API地址: {API_URL}")
    print(f"测试图像: {TEST_IMAGE}")

    # 运行所有测试
    tests = [
        ("健康检查", test_health),
        ("模型列表", test_list_models),
        ("单图像预测", lambda: test_single_predict("alexnet")),
        ("批量预测", lambda: test_batch_predict("resnet")),
        ("模型对比", test_compare_models),
        # ("所有模型测试", test_all_models),  # 可选：测试所有模型
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✓ 通过" if success else "✗ 失败"
        except Exception as e:
            results[test_name] = f"✗ 错误: {str(e)}"

    # 打印测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, result in results.items():
        print(f"{test_name:<20}: {result}")

    # 统计
    passed = sum(1 for r in results.values() if "✓" in r)
    total = len(results)
    print("-" * 60)
    print(f"总计: {passed}/{total} 个测试通过")
    print("="*60)

if __name__ == "__main__":
    main()
