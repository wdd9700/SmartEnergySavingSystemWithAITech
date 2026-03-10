#!/usr/bin/env python3
"""
模型下载脚本 - 国内镜像源优先
自动下载YOLOv8 nano ONNX模型
"""
import os
import urllib.request
import sys
import socket

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# 国内镜像源优先
MODELS = {
    "yolov8n.onnx": {
        "urls": [
            # HF国内镜像
            "https://hf-mirror.com/ultralytics/assets/resolve/main/yolov8n.onnx",
            "https://hf-mirror.com/ultralytics/YOLOv8/resolve/main/yolov8n.onnx",
            # 阿里云镜像
            "https://mirrors.aliyun.com/huggingface/ultralytics/assets/yolov8n.onnx",
            # 腾讯云镜像
            "https://mirrors.cloud.tencent.com/huggingface/ultralytics/assets/yolov8n.onnx",
            # 清华镜像
            "https://mirrors.tuna.tsinghua.edu.cn/huggingface/ultralytics/assets/yolov8n.onnx",
            # 国际源
            "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx",
            "https://media.githubusercontent.com/media/ultralytics/assets/main/models/yolov8n.onnx",
        ],
        "description": "YOLOv8 Nano - 目标检测",
        "size_mb": 12
    }
}

def download_file(url: str, dest_path: str, description: str):
    """带进度条的文件下载"""
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  下载 {description}: {percent:.1f}%")
        sys.stdout.flush()
    
    try:
        socket.setdefaulttimeout(60)
        
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

def main():
    print("=" * 50)
    print("智能节能系统 - 模型下载 (国内镜像)")
    print("=" * 50)
    
    for filename, info in MODELS.items():
        dest_path = os.path.join(MODELS_DIR, filename)
        
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  ✓ {filename} 已存在 ({size_mb:.1f} MB)")
            continue
        
        print(f"\n{info['description']}")
        print(f"  预计大小: {info['size_mb']} MB")
        
        success = False
        for url in info['urls']:
            print(f"\n  尝试: {url[:50]}...")
            if download_file(url, dest_path, filename):
                # 验证文件大小 (模型应 ~12MB)
                if os.path.getsize(dest_path) > 1000000:
                    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                    print(f"  保存至: {dest_path} ({size_mb:.1f} MB)")
                    success = True
                    break
                else:
                    print(f"  文件大小异常 ({os.path.getsize(dest_path)} bytes)，尝试下一个源")
                    os.remove(dest_path)
        
        if not success:
            print(f"\n  ❌ 所有源下载失败")
            print(f"  请手动下载并保存到: {dest_path}")
    
    print("\n" + "=" * 50)
    print("下载完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
