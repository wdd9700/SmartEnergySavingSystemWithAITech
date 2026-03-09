#!/usr/bin/env python3
"""
测试脚本 - 生成模拟视频用于开发测试
无需摄像头即可测试系统
"""
import cv2
import numpy as np
import random
from pathlib import Path


def create_test_video(output_path: str = "tests/test_corridor.mp4", 
                      duration_sec: int = 30, fps: int = 15):
    """
    创建模拟楼道监控视频
    
    场景：
    - 0-5s: 空画面
    - 5-10s: 一个人走过
    - 10-15s: 空画面
    - 15-20s: 两个人走过
    - 20-30s: 空画面
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    for frame_idx in range(total_frames):
        # 创建深色背景（模拟夜间楼道）
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        
        # 添加一些"墙面"细节
        cv2.rectangle(frame, (0, 0), (width, 100), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, height-50), (width, height), (40, 40, 40), -1)
        
        # 添加"灯"的辉光效果
        light_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(light_mask, (width//2, 50), 80, 100, -1)
        light_mask = cv2.GaussianBlur(light_mask, (51, 51), 0)
        frame[:, :, 0] = np.clip(frame[:, :, 0].astype(int) + light_mask // 3, 0, 255).astype(np.uint8)
        frame[:, :, 1] = np.clip(frame[:, :, 1].astype(int) + light_mask // 4, 0, 255).astype(np.uint8)
        
        sec = frame_idx / fps
        
        # 场景1: 5-10s 一个人走过
        if 5 <= sec < 10:
            progress = (sec - 5) / 5
            x = int(50 + progress * (width - 100))
            y = height // 2
            # 画人形（简化的矩形+圆形）
            body_h = 100
            body_w = 40
            cv2.ellipse(frame, (x, y - body_h//2), (15, 15), 0, 0, 360, (80, 80, 80), -1)  # 头
            cv2.rectangle(frame, (x - body_w//2, y - body_h//2), 
                         (x + body_w//2, y + body_h//2), (60, 70, 80), -1)  # 身体
            cv2.rectangle(frame, (x - 10, y + body_h//2), 
                         (x - 5, y + body_h//2 + 50), (50, 50, 50), -1)  # 左腿
            cv2.rectangle(frame, (x + 5, y + body_h//2), 
                         (x + 10, y + body_h//2 + 50), (50, 50, 50), -1)  # 右腿
        
        # 场景2: 15-20s 两个人走过
        elif 15 <= sec < 20:
            progress = (sec - 15) / 5
            
            # 第一个人
            x1 = int(30 + progress * (width - 100))
            y = height // 2
            cv2.ellipse(frame, (x1, y - 50), (15, 15), 0, 0, 360, (70, 70, 70), -1)
            cv2.rectangle(frame, (x1 - 20, y - 50), (x1 + 20, y + 50), (50, 60, 70), -1)
            
            # 第二个人（稍慢一点）
            x2 = int(80 + progress * (width - 150))
            cv2.ellipse(frame, (x2, y - 50), (15, 15), 0, 0, 360, (75, 75, 75), -1)
            cv2.rectangle(frame, (x2 - 20, y - 50), (x2 + 20, y + 50), (55, 65, 75), -1)
        
        # 添加噪声（模拟低光照噪点）
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 时间戳
        cv2.putText(frame, f"{sec:.1f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        out.write(frame)
    
    out.release()
    print(f"测试视频已保存: {output_path}")
    return output_path


def create_classroom_video(output_path: str = "tests/test_classroom.mp4",
                           duration_sec: int = 60, fps: int = 10):
    """
    创建模拟教室视频
    
    场景：
    - 0-10s: 3个人（前区）
    - 10-20s: 8个人（前后都有）
    - 20-40s: 15个人（满座）
    - 40-50s: 5个人（散座）
    - 50-60s: 0人（下课）
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    random.seed(42)  # 可复现
    
    # 预生成人的位置
    people_positions = []
    for i in range(20):
        people_positions.append({
            'x': random.randint(50, width - 50),
            'y': random.randint(100, height - 100),
            'color': (random.randint(50, 100), random.randint(50, 100), random.randint(50, 100))
        })
    
    for frame_idx in range(total_frames):
        # 教室背景
        frame = np.full((height, width, 3), (200, 180, 160), dtype=np.uint8)
        
        # 画桌椅（简化）
        for row in range(4):
            for col in range(5):
                x = 80 + col * 110
                y = 80 + row * 90
                cv2.rectangle(frame, (x, y), (x + 80, y + 50), (100, 90, 80), -1)
        
        sec = frame_idx / fps
        
        # 根据时间段确定人数
        if sec < 10:
            num_people = 3
        elif sec < 20:
            num_people = 8
        elif sec < 40:
            num_people = 15
        elif sec < 50:
            num_people = 5
        else:
            num_people = 0
        
        # 画人
        for i in range(min(num_people, len(people_positions))):
            person = people_positions[i]
            # 添加微小移动
            dx = int(5 * np.sin(frame_idx * 0.1 + i))
            dy = int(3 * np.cos(frame_idx * 0.15 + i))
            
            x = person['x'] + dx
            y = person['y'] + dy
            
            # 简化的坐姿人形
            cv2.ellipse(frame, (x, y - 30), (12, 12), 0, 0, 360, person['color'], -1)  # 头
            cv2.rectangle(frame, (x - 18, y - 30), (x + 18, y + 20), person['color'], -1)  # 上身
            # 桌子遮挡下半身
        
        # 人数显示
        cv2.putText(frame, f"People: {num_people}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.putText(frame, f"Time: {sec:.0f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        
        out.write(frame)
    
    out.release()
    print(f"教室测试视频已保存: {output_path}")
    return output_path


def main():
    """生成测试视频"""
    Path("tests").mkdir(exist_ok=True)
    
    print("生成测试视频...")
    print("-" * 40)
    
    create_test_video("tests/test_corridor.mp4", duration_sec=30)
    create_classroom_video("tests/test_classroom.mp4", duration_sec=60)
    
    print("-" * 40)
    print("测试视频生成完成!")
    print("\n运行测试:")
    print("  python -m corridor_light.main --source tests/test_corridor.mp4")
    print("  python -m classroom_ac.main --source tests/test_classroom.mp4")


if __name__ == "__main__":
    main()
