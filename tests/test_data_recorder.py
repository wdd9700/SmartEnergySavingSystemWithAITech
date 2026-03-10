#!/usr/bin/env python3
"""
测试数据记录和统一主程序功能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from shared.data_recorder import DataRecorder, HeatmapGenerator, EnergyEstimator


def test_data_recorder():
    """测试数据记录器"""
    print("=" * 60)
    print("测试1: 数据记录器")
    print("=" * 60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        recorder = DataRecorder(log_dir=temp_dir, max_memory_records=1000)
        
        # 模拟记录检测数据
        print("\n记录检测数据...")
        for i in range(10):
            recorder.record_detection(
                timestamp=datetime.now(),
                camera_id='test_cam',
                people_count=i % 3,
                light_states={'light_0': i % 2 == 0},
                brightness=50 + i * 5,
                inference_time_ms=100 + i * 10,
                fps=15.0,
                person_locations=[(100 + i*10, 200)] if i % 3 > 0 else []
            )
            time.sleep(0.01)
        
        # 记录事件
        print("记录事件...")
        recorder.record_event('light_on', '灯光开启', 'test_cam')
        recorder.record_event('person_enter', '人员进入', 'test_cam')
        
        # 获取统计
        print("\n获取统计数据...")
        stats = recorder.get_statistics(hours=1)
        print(f"  记录数: {stats.get('total_records', 0)}")
        print(f"  平均人数: {stats.get('people', {}).get('avg', 0):.2f}")
        print(f"  最大人数: {stats.get('people', {}).get('max', 0)}")
        print(f"  平均推理时间: {stats.get('performance', {}).get('avg_inference_ms', 0):.2f}ms")
        
        # 导出
        export_path = f"{temp_dir}/export.json"
        recorder.export_to_json(export_path, hours=1)
        print(f"\n数据已导出: {export_path}")
        
        # 验证导出文件
        with open(export_path, 'r') as f:
            data = json.load(f)
        print(f"  导出记录数: {len(data.get('records', []))}")
        
        recorder.close()
        print("\n✅ 数据记录器测试通过")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_heatmap_generator():
    """测试热力图生成器"""
    print("\n" + "=" * 60)
    print("测试2: 热力图生成器")
    print("=" * 60)
    
    import numpy as np
    import cv2
    
    # 创建生成器
    heatmap = HeatmapGenerator((480, 640))
    
    # 添加模拟数据
    print("\n添加模拟人员位置数据...")
    
    # 模拟人员在画面左侧频繁出现
    for _ in range(50):
        x = np.random.randint(50, 200)
        y = np.random.randint(150, 350)
        heatmap.add_frame([(x, y)], weight=1.0)
    
    # 模拟人员在画面右侧少量出现
    for _ in range(20):
        x = np.random.randint(400, 600)
        y = np.random.randint(150, 350)
        heatmap.add_frame([(x, y)], weight=0.5)
    
    print(f"  处理了 {heatmap.total_frames} 帧")
    
    # 生成热力图
    result = heatmap.generate()
    print(f"  热力图尺寸: {result.shape}")
    
    # 保存
    temp_path = tempfile.mktemp(suffix='.jpg')
    heatmap.save(temp_path)
    print(f"  热力图已保存: {temp_path}")
    
    # 清理
    Path(temp_path).unlink(missing_ok=True)
    
    print("\n✅ 热力图生成器测试通过")
    return True


def test_energy_estimator():
    """测试能耗估算器"""
    print("\n" + "=" * 60)
    print("测试3: 能耗估算器")
    print("=" * 60)
    
    energy = EnergyEstimator()
    
    # 模拟灯光开关
    print("\n模拟灯光使用场景...")
    
    # light_0 开启10分钟
    energy.update_light_state('light_0', True)
    time.sleep(0.01)  # 加速测试
    energy.update_light_state('light_0', False)
    
    # light_1 开启5分钟
    energy.update_light_state('light_1', True)
    time.sleep(0.01)
    energy.update_light_state('light_1', False)
    
    # light_2 保持开启
    energy.update_light_state('light_2', True)
    
    # 获取统计
    stats = energy.get_statistics()
    print(f"\n能耗统计:")
    print(f"  总能耗: {stats['total_energy_wh']:.4f} Wh")
    print(f"  总能耗: {stats['total_energy_kwh']:.6f} kWh")
    print(f"  各灯开启时长: {stats['light_on_time_hours']}")
    print(f"  当前开启: {stats['current_active']}")
    
    # 节能估算
    savings = energy.estimate_savings(traditional_mode_hours=0.5)
    print(f"\n节能估算 (传统模式0.5小时):")
    print(f"  传统模式能耗: {savings['traditional_mode_kwh']:.6f} kWh")
    print(f"  智能模式能耗: {savings['smart_mode_kwh']:.6f} kWh")
    print(f"  节约: {savings['savings_kwh']:.6f} kWh ({savings['savings_percent']:.1f}%)")
    print(f"  节约成本: ¥{savings['cost_savings_yuan']:.2f}")
    
    print("\n✅ 能耗估算器测试通过")
    return True


def test_hourly_report():
    """测试小时级报告"""
    print("\n" + "=" * 60)
    print("测试4: 小时级报告")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        recorder = DataRecorder(log_dir=temp_dir)
        
        # 模拟一天的数据
        print("\n生成模拟数据...")
        base_time = datetime.now() - timedelta(hours=5)
        
        for hour in range(5):
            for minute in range(0, 60, 5):  # 每5分钟一条记录
                timestamp = base_time + timedelta(hours=hour, minutes=minute)
                people_count = (hour + 1) % 3  # 模拟不同时段人数变化
                
                recorder.record_detection(
                    timestamp=timestamp,
                    camera_id='test_cam',
                    people_count=people_count,
                    light_states={'light_0': people_count > 0},
                    brightness=50,
                    inference_time_ms=100,
                    fps=15.0
                )
        
        # 生成小时级报告
        report = recorder.generate_hourly_report(hours=6)
        
        print("\n小时级报告:")
        for hour, data in sorted(report.items())[:3]:  # 显示前3小时
            print(f"  {hour}: 平均{data['avg_people']:.2f}人, {data['detection_count']}次检测")
        
        recorder.close()
        print("\n✅ 小时级报告测试通过")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    print("\n" + "=" * 60)
    print("数据记录与分析模块 - 测试套件")
    print("=" * 60)
    
    tests = [
        ("数据记录器", test_data_recorder),
        ("热力图生成器", test_heatmap_generator),
        ("能耗估算器", test_energy_estimator),
        ("小时级报告", test_hourly_report),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {name} 测试失败")
                failed += 1
        except Exception as e:
            print(f"\n❌ {name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
