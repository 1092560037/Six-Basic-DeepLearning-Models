#!/bin/bash

# Docker部署启动脚本

echo "============================================"
echo "深度学习模型Docker部署脚本"
echo "============================================"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 显示菜单
echo ""
echo "请选择部署方式:"
echo "1) CPU版本 (推荐)"
echo "2) GPU版本 (需要NVIDIA GPU)"
echo "3) 停止所有服务"
echo "4) 查看服务日志"
echo "5) 重新构建镜像"
echo "6) 清理所有容器和镜像"
echo "0) 退出"
echo ""

read -p "请输入选项 [0-6]: " choice

case $choice in
    1)
        echo ""
        echo "启动CPU版本服务..."
        docker-compose up -d model-inference-cpu
        echo ""
        echo "服务已启动！"
        echo "API地址: http://localhost:5000"
        echo ""
        echo "测试服务:"
        echo "  curl http://localhost:5000/health"
        echo ""
        echo "查看日志:"
        echo "  docker-compose logs -f model-inference-cpu"
        ;;
    2)
        echo ""
        echo "检查NVIDIA Docker支持..."
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo "错误: GPU支持不可用，请确保已安装nvidia-docker2"
            exit 1
        fi
        echo "启动GPU版本服务..."
        docker-compose --profile gpu up -d model-inference-gpu
        echo ""
        echo "服务已启动！"
        echo "API地址: http://localhost:5001"
        echo ""
        echo "查看GPU使用:"
        echo "  docker exec dl-models-gpu nvidia-smi"
        ;;
    3)
        echo ""
        echo "停止所有服务..."
        docker-compose down
        echo "服务已停止"
        ;;
    4)
        echo ""
        echo "选择要查看的服务:"
        echo "1) CPU版本"
        echo "2) GPU版本"
        read -p "请输入选项 [1-2]: " log_choice

        if [ "$log_choice" == "1" ]; then
            docker-compose logs -f model-inference-cpu
        elif [ "$log_choice" == "2" ]; then
            docker-compose logs -f model-inference-gpu
        else
            echo "无效选项"
        fi
        ;;
    5)
        echo ""
        echo "重新构建镜像..."
        echo "1) CPU版本"
        echo "2) GPU版本"
        echo "3) 全部"
        read -p "请输入选项 [1-3]: " build_choice

        if [ "$build_choice" == "1" ]; then
            docker-compose build --no-cache model-inference-cpu
        elif [ "$build_choice" == "2" ]; then
            docker-compose build --no-cache model-inference-gpu
        elif [ "$build_choice" == "3" ]; then
            docker-compose build --no-cache
        else
            echo "无效选项"
        fi
        echo "构建完成"
        ;;
    6)
        echo ""
        read -p "警告: 这将删除所有容器和镜像，确定吗? (y/N): " confirm
        if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
            echo "停止并删除容器..."
            docker-compose down
            echo "删除镜像..."
            docker rmi $(docker images | grep "model test" | awk '{print $3}') 2>/dev/null
            echo "清理完成"
        else
            echo "已取消"
        fi
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "============================================"
