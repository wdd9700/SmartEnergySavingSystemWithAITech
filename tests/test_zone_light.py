#!/usr/bin/env python3
"""
测试基于位置的智能灯光控制
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from corridor_light.light_zones import LightConfig, LightZone, create_default_config
from corridor_light.zone_controller import ZoneLightController


def test_light_zones():
    """测试灯光区域功能"""
    print("=" * 60)
    print("测试1: 灯光区域基础功能")
    print("=" * 60)
    
    # 创建配置
    config = create_default_config(640, 480)
    
    # 测试点包含检测
    test_points = [
        (128, 240),  # 在light_0内
        (320, 240),  # 在light_1内
        (512, 240),  # 在light_2内
        (50, 50),    # 不在任何区域内
    ]
    
    print("\n点包含测试:")
    for point in test_points:
        zone = config.find_zone_by_position(point)
        nearest = config.find_nearest_zone(point)
        if zone:
            print(f"  点{point}: 在区域 [{zone.name}], 最近 [{nearest.name}]")
        else:
            print(f"  点{point}: 不在任何区域, 最近 [{nearest.name}]")
    
    # 测试灯光获取
    print("\n灯光开启测试 (forward 模式):")
    for point in test_points:
        lights = config.get_lights_for_person(point, 'forward')
        zone = config.find_zone_by_position(point)
        location = zone.name if zone else "外部"
        print(f"  人在{location} {point}: 应开启灯 {lights}")
    
    print("\n灯光开启测试 (both 模式):")
    for point in test_points:
        lights = config.get_lights_for_person(point, 'both')
        zone = config.find_zone_by_position(point)
        location = zone.name if zone else "外部"
        print(f"  人在{location} {point}: 应开启灯 {lights}")
    
    return True


def test_zone_controller():
    """测试区域控制器"""
    print("\n" + "=" * 60)
    print("测试2: 区域控制器")
    print("=" * 60)
    
    config = create_default_config(640, 480)
    controller = ZoneLightController(
        light_config=config,
        light_off_delay=0.5,
        facing_direction='forward',
        demo_mode=True
    )
    
    controller.init()
    
    # 模拟检测
    print("\n模拟人形检测场景:")
    
    # 场景1: 人在入口
    print("\n场景1: 人在入口灯区域")
    detections = [
        {'class': 'person', 'foot_point': (128, 240), 'confidence': 0.9}
    ]
    states = controller.update(detections)
    active = [k for k, v in states.items() if v]
    print(f"  开启的灯: {active}")
    
    # 场景2: 人在中间
    print("\n场景2: 人在中间灯区域")
    detections = [
        {'class': 'person', 'foot_point': (320, 240), 'confidence': 0.9}
    ]
    states = controller.update(detections)
    active = [k for k, v in states.items() if v]
    print(f"  开启的灯: {active}")
    
    # 场景3: 多个人在不同区域
    print("\n场景3: 两个人分别在不同区域")
    detections = [
        {'class': 'person', 'foot_point': (128, 240), 'confidence': 0.9},
        {'class': 'person', 'foot_point': (512, 240), 'confidence': 0.85}
    ]
    states = controller.update(detections)
    active = [k for k, v in states.items() if v]
    print(f"  开启的灯: {active}")
    
    # 场景4: 人离开
    print("\n场景4: 人离开 (空检测)")
    import time
    time.sleep(0.6)  # 等待关灯延迟
    states = controller.update([])
    active = [k for k, v in states.items() if v]
    print(f"  开启的灯: {active} (应全部关闭)")
    
    controller.cleanup()
    return True


def test_calibration():
    """测试自动校准"""
    print("\n" + "=" * 60)
    print("测试3: 自动校准功能")
    print("=" * 60)
    
    # 模拟检测结果
    detections = [
        {'class': 'person', 'foot_point': (100, 200), 'confidence': 0.9},
        {'class': 'person', 'foot_point': (300, 200), 'confidence': 0.85},
        {'class': 'person', 'foot_point': (500, 200), 'confidence': 0.88},
        {'class': 'person', 'foot_point': (105, 205), 'confidence': 0.9},  # 太接近，应被合并
    ]
    
    config = LightConfig()
    zones = config.calibrate_from_detections(detections, zone_radius=100)
    
    print(f"\n从 {len(detections)} 个人形位置生成了 {len(zones)} 个灯光区域:")
    for zone in zones:
        print(f"  [{zone.id}] {zone.name}: 位置({zone.x}, {zone.y})")
        print(f"    前方: {zone.forward_zones}, 后方: {zone.backward_zones}")
    
    # 保存和加载测试
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    config.save_to_file(temp_path)
    print(f"\n配置已保存到临时文件")
    
    loaded_config = LightConfig.load_from_file(temp_path)
    print(f"从文件加载配置: {len(loaded_config.zones)} 个区域")
    
    os.unlink(temp_path)
    return True


def test_visualization():
    """可视化测试"""
    print("\n" + "=" * 60)
    print("测试4: 可视化 (生成示意图)")
    print("=" * 60)
    
    config = create_default_config(640, 480)
    controller = ZoneLightController(config, demo_mode=True)
    controller.init()
    
    # 创建空白画面
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 绘制区域
    for zone in config.get_all_zones():
        cv2.circle(frame, (zone.x, zone.y), zone.radius, (64, 64, 64), 1)
        cv2.circle(frame, (zone.x, zone.y), 8, (128, 128, 128), -1)
        cv2.putText(frame, zone.name, (zone.x - 30, zone.y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    # 模拟人在中间区域
    person_pos = (320, 240)
    lights = config.get_lights_for_person(person_pos, 'forward')
    
    # 绘制人
    cv2.circle(frame, person_pos, 5, (0, 0, 255), -1)
    cv2.circle(frame, person_pos, 10, (0, 0, 255), 2)
    cv2.putText(frame, "Person", (person_pos[0] - 30, person_pos[1] + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 高亮激活的灯
    for light_id in lights:
        zone = config.get_zone(light_id)
        if zone:
            cv2.circle(frame, (zone.x, zone.y), zone.radius, (0, 255, 0), 2)
            cv2.circle(frame, (zone.x, zone.y), 10, (0, 255, 0), -1)
            cv2.line(frame, person_pos, (zone.x, zone.y), (0, 255, 255), 2)
    
    # 保存示意图
    output_path = "zone_light_demo.jpg"
    cv2.imwrite(output_path, frame)
    print(f"示意图已保存: {output_path}")
    print(f"  人在位置 {person_pos}")
    print(f"  应开启灯: {lights}")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("基于位置的智能灯光控制系统 - 测试套件")
    print("=" * 60)
    
    tests = [
        ("灯光区域功能", test_light_zones),
        ("区域控制器", test_zone_controller),
        ("自动校准", test_calibration),
        ("可视化", test_visualization),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                print(f"\n✅ {name} 测试通过")
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
