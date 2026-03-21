#!/usr/bin/env python3
"""
模型下载脚本 - 国内镜像源优先
自动下载YOLOv8 nano ONNX模型
"""
import os
import json
import urllib.request
import sys
import socket
from typing import Dict, Any

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置文件路径
CONFIG_FILE = os.path.join(MODELS_DIR, "model_sources.json")

# 默认配置（当配置文件不存在时使用）
DEFAULT_MODELS_CONFIG = {
    "yolov8n.onnx": {
        "urls": [
            "https://hf-mirror.com/ultralytics/assets/resolve/main/yolov8n.onnx",
            "https://hf-mirror.com/ultralytics/YOLOv8/resolve/main/yolov8n.onnx",
            "https://mirrors.aliyun.com/huggingface/ultralytics/assets/yolov8n.onnx",
            "https://mirrors.cloud.tencent.com/huggingface/ultralytics/assets/yolov8n.onnx",
            "https://mirrors.tuna.tsinghua.edu.cn/huggingface/ultralytics/assets/yolov8n.onnx",
            "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx",
            "https://media.githubusercontent.com/media/ultralytics/assets/main/models/yolov8n.onnx",
        ],
        "description": "YOLOv8 Nano - 目标检测",
        "size_mb": 12
    }
}

# 网络请求超时设置（秒）
DEFAULT_SOCKET_TIMEOUT = 60

# 最小有效文件大小（字节）
MIN_VALID_FILE_SIZE = 1_000_000


def load_model_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """从配置文件加载模型下载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        模型配置字典
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('models', DEFAULT_MODELS_CONFIG)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 读取配置文件失败，使用默认配置: {e}")
            return DEFAULT_MODELS_CONFIG
    return DEFAULT_MODELS_CONFIG

def download_file(url: str, dest_path: str, description: str, 
                  timeout: int = DEFAULT_SOCKET_TIMEOUT) -> bool:
    """带进度条的文件下载
    
    Args:
        url: 下载地址
        dest_path: 目标文件路径
        description: 下载描述
        timeout: 网络请求超时时间（秒）
        
    Returns:
        下载是否成功
    """
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  下载 {description}: {percent:.1f}%")
        sys.stdout.flush()
    
    try:
        socket.setdefaulttimeout(timeout)
        
        # 添加User-Agent头
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
            ('Accept', 'application/octet-stream,application/*,*/*')
        ]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"  ✓ 完成")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

def verify_downloaded_file(file_path: str, min_size: int = MIN_VALID_FILE_SIZE) -> bool:
    """验证下载的文件是否有效
    
    Args:
        file_path: 文件路径
        min_size: 最小有效文件大小（字节）
        
    Returns:
        文件是否有效
    """
    if not os.path.exists(file_path):
        return False
    file_size = os.path.getsize(file_path)
    if file_size < min_size:
        return False
    return True


def main():
    """主函数"""
    print("=" * 50)
    print("智能节能系统 - 模型下载 (国内镜像)")
    print("=" * 50)
    
    # 加载模型配置
    models_config = load_model_config()
    
    for filename, info in models_config.items():
        dest_path = os.path.join(MODELS_DIR, filename)
        
        # 检查文件是否已存在且有效
        if verify_downloaded_file(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  ✓ {filename} 已存在 ({size_mb:.1f} MB)")
            continue
        
        print(f"\n{info['description']}")
        print(f"  预计大小: {info['size_mb']} MB")
        
        success = False
        urls = info.get('urls', [])
        
        for url in urls:
            print(f"\n  尝试: {url[:50]}...")
            if download_file(url, dest_path, filename):
                # 验证文件大小
                if verify_downloaded_file(dest_path):
                    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                    print(f"  保存至: {dest_path} ({size_mb:.1f} MB)")
                    success = True
                    break
                else:
                    file_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
                    print(f"  文件大小异常 ({file_size} bytes)，尝试下一个源")
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
        
        if not success:
            print(f"\n  ❌ 所有源下载失败")
            print(f"  请手动下载并保存到: {dest_path}")
    
    print("\n" + "=" * 50)
    print("下载完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
