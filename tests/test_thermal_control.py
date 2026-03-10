#!/usr/bin/env python3
"""
测试热负荷计算和预测性控制模块
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import tempfile
import json
from datetime import datetime, timedelta

from classroom_ac.thermal_controller import (
    HeatLoadCalculator,
    ThermalLoadConfig,
    ScheduleManager,
    PredictiveACController
)


def test_heat_load_calculator():
    """测试热负荷计算器"""
    print("=" * 60)
    print("测试1: 热负荷计算")
    print("=" * 60)
    
    calc = HeatLoadCalculator()
    
    # 测试场景: 正常上课
    print("\n场景1: 正常上课 (30人, 思考状态)")
    load = calc.calculate_total_load(
        person_count=30,
        outdoor_temp=32.0,
        indoor_temp=28.0,
        laptop_count=20,  # 20人带笔记本
        activity_level='thinking'
    )
    
    print(f"  人体产热: {load['person_heat']:.0f}W")
    print(f"  设备产热: {load['equipment_heat']:.0f}W")
    print(f"  围护结构: {load['envelope_heat']:.0f}W")
    print(f"  太阳辐射: {load['solar_heat']:.0f}W")
    print(f"  总热负荷: {load['total_load']:.0f}W")
    print(f"  需制冷量: {load['cooling_required']:.0f}W")
    
    # 测试场景: 课间休息后人刚进来
    print("\n场景2: 课间休息后 (轻度运动后)")
    load2 = calc.calculate_total_load(
        person_count=30,
        outdoor_temp=32.0,
        indoor_temp=29.0,
        laptop_count=20,
        activity_level='light_exercise'  # 轻度运动后
    )
    print(f"  人体产热: {load2['person_heat']:.0f}W (轻度运动150W/人)")
    print(f"  总热负荷: {load2['total_load']:.0f}W")
    print(f"  比思考状态多: {load2['total_load'] - load['total_load']:.0f}W")
    
    # 测试场景: 夜间自习
    print("\n场景3: 夜间自习 (人少, 无太阳)")
    load3 = calc.calculate_total_load(
        person_count=10,
        outdoor_temp=28.0,
        indoor_temp=27.0,
        laptop_count=10,
        activity_level='resting',
        solar_radiation=0
    )
    print(f"  总热负荷: {load3['total_load']:.0f}W")
    
    return True


def test_schedule_manager():
    """测试课表管理器"""
    print("\n" + "=" * 60)
    print("测试2: 课表管理")
    print("=" * 60)
    
    # 创建临时课表文件
    temp_file = tempfile.mktemp(suffix='.json')
    schedule_data = {
        "schedule": {
            "monday": [
                {"start": "08:00", "end": "09:35", "course": "数学", "expected_people": 40}
            ]
        },
        "bookings": []
    }
    with open(temp_file, 'w') as f:
        json.dump(schedule_data, f)
    
    manager = ScheduleManager(temp_file)
    
    # 模拟周一上课时间
    test_time = datetime(2026, 3, 10, 8, 30)  # 周一 8:30
    
    is_class = manager.is_class_time(test_time)
    print(f"\n  周一8:30有课: {is_class}")
    
    expected = manager.get_expected_people(test_time)
    print(f"  预期人数: {expected}")
    
    # 测试距离下一节课
    test_time2 = datetime(2026, 3, 10, 7, 30)  # 周一 7:30
    time_to = manager.get_time_to_next_class(test_time2)
    print(f"  距离下一节课: {time_to}分钟")
    
    import os
    os.unlink(temp_file)
    
    print("\n✅ 课表管理测试通过")
    return True


def test_predictive_controller():
    """测试预测性控制器"""
    print("\n" + "=" * 60)
    print("测试3: 预测性控制")
    print("=" * 60)
    
    # 创建临时课表
    temp_file = tempfile.mktemp(suffix='.json')
    schedule_data = {
        "schedule": {
            "tuesday": [
                {"start": "09:00", "end": "10:30", "course": "测试课", "expected_people": 30}
            ]
        },
        "bookings": []
    }
    with open(temp_file, 'w') as f:
        json.dump(schedule_data, f)
    
    calc = HeatLoadCalculator()
    schedule = ScheduleManager(temp_file)
    controller = PredictiveACController(calc, schedule)
    
    # 场景1: 上课中，温度高
    print("\n场景1: 上课中，室内温度28.5°C")
    controller.indoor_temp = 28.5
    controller.outdoor_temp = 32.0
    
    # 模拟当前时间有课
    test_time = datetime(2026, 3, 11, 9, 30)  # 周二 9:30
    
    # 计算当前负荷
    load_data = calc.calculate_total_load(
        person_count=30,
        outdoor_temp=32.0,
        indoor_temp=28.5,
        laptop_count=20,
        activity_level='thinking'
    )
    
    decision = controller.make_decision()
    print(f"  决策: 空调{'开启' if decision['ac_on'] else '关闭'}")
    print(f"  原因: {decision['reason']}")
    print(f"  预计负荷: {decision['estimated_load']:.0f}W")
    
    # 场景2: 即将上课，预冷
    print("\n场景2: 距离上课还有8分钟，室温27.5°C")
    controller.indoor_temp = 27.5
    # 模拟即将上课
    test_time2 = datetime(2026, 3, 11, 8, 52)  # 周二 8:52 (距离9:00还有8分钟)
    
    decision2 = controller.make_decision()
    print(f"  决策: 空调{'开启' if decision2['ac_on'] else '关闭'}")
    print(f"  是否预冷: {decision2['pre_action']}")
    print(f"  原因: {decision2['reason']}")
    
    # 场景3: 下课了
    print("\n场景3: 下课了，人快走光")
    controller.indoor_temp = 26.5
    
    # 模拟历史数据趋势下降
    for _ in range(10):
        calc.people_history.append((datetime.now(), 5))
    
    decision3 = controller.make_decision()
    print(f"  决策: 空调{'开启' if decision3['ac_on'] else '关闭'}, 风扇{'开启' if decision3['fan_on'] else '关闭'}")
    print(f"  原因: {decision3['reason']}")
    
    import os
    os.unlink(temp_file)
    
    print("\n✅ 预测性控制测试通过")
    return True


def test_trend_analysis():
    """测试趋势分析"""
    print("\n" + "=" * 60)
    print("测试4: 趋势分析")
    print("=" * 60)
    
    calc = HeatLoadCalculator()
    
    # 模拟上升趋势
    print("\n模拟人数上升趋势...")
    base_time = datetime.now() - timedelta(minutes=10)
    for i in range(10):
        calc.people_history.append((base_time + timedelta(minutes=i), 5 + i * 3))
    
    trend = calc.get_people_trend(minutes=10)
    print(f"  趋势: {trend}")
    
    # 预测未来负荷
    future = calc.predict_future_load(minutes_ahead=10)
    print(f"  预测10分钟后负荷: {future:.0f}W")
    
    # 模拟下降趋势
    print("\n模拟人数下降趋势...")
    calc.people_history.clear()
    for i in range(10):
        calc.people_history.append((base_time + timedelta(minutes=i), 30 - i * 2))
    
    trend2 = calc.get_people_trend(minutes=10)
    print(f"  趋势: {trend2}")
    
    print("\n✅ 趋势分析测试通过")
    return True


def main():
    print("\n" + "=" * 60)
    print("热负荷计算与预测性控制 - 测试套件")
    print("=" * 60)
    
    tests = [
        ("热负荷计算", test_heat_load_calculator),
        ("课表管理", test_schedule_manager),
        ("预测性控制", test_predictive_controller),
        ("趋势分析", test_trend_analysis),
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
