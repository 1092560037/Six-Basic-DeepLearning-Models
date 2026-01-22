@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================
echo 深度学习模型Docker部署脚本 (Windows)
echo ============================================

REM 检查Docker是否安装
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查Docker是否运行
docker info >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: Docker未运行，请启动Docker Desktop
    pause
    exit /b 1
)

:menu
echo.
echo 请选择部署方式:
echo 1^) CPU版本 (推荐)
echo 2^) GPU版本 (需要NVIDIA GPU)
echo 3^) 停止所有服务
echo 4^) 查看服务日志
echo 5^) 重新构建镜像
echo 6^) 运行API测试
echo 0^) 退出
echo.

set /p choice="请输入选项 [0-6]: "

if "%choice%"=="1" goto cpu
if "%choice%"=="2" goto gpu
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto logs
if "%choice%"=="5" goto build
if "%choice%"=="6" goto test
if "%choice%"=="0" goto end
echo 无效选项
goto menu

:cpu
echo.
echo 启动CPU版本服务...
docker-compose up -d model-inference-cpu
if %errorlevel% equ 0 (
    echo.
    echo 服务已启动！
    echo API地址: http://localhost:5000
    echo.
    echo 测试服务:
    echo   curl http://localhost:5000/health
    echo.
    echo 查看日志:
    echo   docker-compose logs -f model-inference-cpu
    echo.
    echo 运行测试脚本:
    echo   python test_api.py
) else (
    echo 启动失败，请检查日志
)
pause
goto end

:gpu
echo.
echo 启动GPU版本服务...
docker-compose --profile gpu up -d model-inference-gpu
if %errorlevel% equ 0 (
    echo.
    echo 服务已启动！
    echo API地址: http://localhost:5001
    echo.
    echo 查看GPU使用:
    echo   docker exec dl-models-gpu nvidia-smi
) else (
    echo 启动失败，请确保已安装nvidia-docker2
)
pause
goto end

:stop
echo.
echo 停止所有服务...
docker-compose down
echo 服务已停止
pause
goto end

:logs
echo.
echo 查看CPU版本日志 (按Ctrl+C退出)...
docker-compose logs -f model-inference-cpu
pause
goto end

:build
echo.
echo 重新构建CPU版本镜像...
docker-compose build --no-cache model-inference-cpu
echo 构建完成
pause
goto end

:test
echo.
echo 运行API测试...
if exist test_api.py (
    python test_api.py
) else (
    echo 错误: test_api.py 文件不存在
)
pause
goto end

:end
echo.
echo ============================================
endlocal
