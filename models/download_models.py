#!/usr/bin/env python3
"""
模型下载脚本
自动下载YOLOv8 nano ONNX模型
"""
import os
import urllib.request
import sys

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "yolov8n.onnx": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx",
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
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"  ✓ 完成")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

def main():
    print("=" * 50)
    print("智能节能系统 - 模型下载")
    print("=" * 50)
    
    for filename, info in MODELS.items():
        dest_path = os.path.join(MODELS_DIR, filename)
        
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  ✓ {filename} 已存在 ({size_mb:.1f} MB)")
            continue
        
        print(f"\n{info['description']}")
        print(f"  预计大小: {info['size_mb']} MB")
        
        if download_file(info['url'], dest_path, filename):
            print(f"  保存至: {dest_path}")
        else:
            print(f"  请手动下载: {info['url']}")
            print(f"  保存到: {dest_path}")
    
    print("\n" + "=" * 50)
    print("下载完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
