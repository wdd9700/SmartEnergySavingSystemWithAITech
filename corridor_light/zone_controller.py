#!/usr/bin/env python3
"""
基于人形位置的智能灯光控制器
支持:
- 根据人形脚底位置确定所在灯区域
- 只开启人所在位置和人前方的灯
- 人离开区域后立即关闭对应灯
- 支持多灯独立控制
"""
import time
import threading
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from corridor_light.light_zones import LightConfig, LightZone


class ZoneLightController:
    """
    基于区域的智能灯光控制器
    
    每个灯独立控制，根据人形位置实时决定开启哪些灯
    """
    
    def __init__(self, 
                 light_config: LightConfig,
                 light_off_delay: float = 0.5,  # 人离开后延迟关灯时间
                 facing_direction: str = 'forward',  # 默认朝向
                 demo_mode: bool = True):
        """
        Args:
            light_config: 灯光区域配置
            light_off_delay: 人离开后延迟关灯时间(秒)
            facing_direction: 默认朝向 ('forward', 'backward', 'both')
            demo_mode: True=仅打印状态，False=控制实际硬件
        """
        self.config = light_config
        self.light_off_delay = light_off_delay
        self.facing_direction = facing_direction
        self.demo_mode = demo_mode
        
        # 灯状态管理 {light_id: {'state': bool, 'last_person_time': float, 'last_zone': str}}
        self.light_states: Dict[str, Dict] = {}
        for zone_id in self.config.zones.keys():
            self.light_states[zone_id] = {
                'state': False,
                'last_person_time': 0,
                'last_zone': None
            }
        
        self._lock = threading.Lock()
        self._gpio_available = False
        self._gpio_pins: Dict[str, int] = {}  # light_id -> GPIO pin
        
        # 统计
        self.stats = {
            'zone_entries': {z: 0 for z in self.config.zones.keys()},
            'light_on_count': {z: 0 for z in self.config.zones.keys()},
            'light_on_time': {z: 0.0 for z in self.config.zones.keys()},
            'last_on_time': {z: None for z in self.config.zones.keys()}
        }
    
    def init(self, gpio_mapping: Dict[str, int] = None) -> bool:
        """
        初始化控制器
        
        Args:
            gpio_mapping: 灯ID到GPIO引脚的映射 {light_id: gpio_pin}
        """
        print("=" * 60)
        print("基于位置的智能灯光控制器初始化")
        print("=" * 60)
        
        # 打印配置信息
        print(f"\n灯光区域配置 ({len(self.config.zones)} 个区域):")
        for zone in self.config.get_all_zones():
            print(f"  [{zone.id}] {zone.name}: 位置({zone.x}, {zone.y}), "
                  f"半径{zone.radius}px, 前方{zone.forward_zones}, 后方{zone.backward_zones}")
        
        print(f"\n控制参数:")
        print(f"  关灯延迟: {self.light_off_delay}s")
        print(f"  默认朝向: {self.facing_direction}")
        print(f"  模式: {'Demo' if self.demo_mode else 'Deploy'}")
        
        # GPIO初始化
        if not self.demo_mode and gpio_mapping:
            self._gpio_pins = gpio_mapping
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                for light_id, pin in gpio_mapping.items():
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                self._gpio_available = True
                print(f"\nGPIO初始化成功: {len(gpio_mapping)} 个灯")
            except ImportError:
                print("\n警告: RPi.GPIO未安装，切换到Demo模式")
                self.demo_mode = True
            except Exception as e:
                print(f"\nGPIO初始化失败: {e}，切换到Demo模式")
                self.demo_mode = True
        
        if self.demo_mode:
            print("\nDemo模式: 仅显示灯光状态，不控制硬件")
        
        return True
    
    def _set_light(self, light_id: str, state: bool):
        """设置单个灯的状态"""
        with self._lock:
            if light_id not in self.light_states:
                return
            
            old_state = self.light_states[light_id]['state']
            if old_state == state:
                return
            
            self.light_states[light_id]['state'] = state
            
            # GPIO控制
            if self._gpio_available and light_id in self._gpio_pins:
                try:
                    import RPi.GPIO as GPIO
                    pin = self._gpio_pins[light_id]
                    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
                except Exception as e:
                    print(f"GPIO控制失败 [{light_id}]: {e}")
            
            # 打印状态
            zone = self.config.get_zone(light_id)
            name = zone.name if zone else light_id
            action = "开启" if state else "关闭"
            print(f"[灯光] {name} ({light_id}): {action}")
            
            # 统计
            if state:
                self.stats['light_on_count'][light_id] += 1
                self.stats['last_on_time'][light_id] = time.time()
            else:
                if self.stats['last_on_time'][light_id]:
                    self.stats['light_on_time'][light_id] += time.time() - self.stats['last_on_time'][light_id]
                    self.stats['last_on_time'][light_id] = None
    
    def update(self, detections: List[Dict], facing_direction: str = None) -> Dict[str, bool]:
        """
        更新检测状态并控制灯光
        
        Args:
            detections: 检测结果列表，每项包含 'foot_point': (x, y)
            facing_direction: 当前朝向，None则使用默认值
        
        Returns:
            各灯的当前状态 {light_id: state}
        """
        now = time.time()
        direction = facing_direction or self.facing_direction
        
        # 1. 收集所有需要开启的灯
        lights_should_on = set()
        person_zones = {}  # 记录每个人所在的区域
        
        for det in detections:
            if det.get('class') != 'person':
                continue
            
            foot_point = det.get('foot_point')
            if not foot_point:
                continue
            
            # 获取此人应开启的灯
            person_lights = self.config.get_lights_for_person(foot_point, direction)
            lights_should_on.update(person_lights)
            
            # 记录所在区域
            zone = self.config.find_zone_by_position(foot_point)
            if zone:
                person_zones[det.get('id', id(det))] = zone.id
                self.stats['zone_entries'][zone.id] += 1
        
        # 2. 更新灯状态
        for light_id in self.light_states.keys():
            should_be_on = light_id in lights_should_on
            
            if should_be_on:
                # 有人需要这个灯，立即开启
                self.light_states[light_id]['last_person_time'] = now
                self._set_light(light_id, True)
            else:
                # 检查是否可以关闭
                time_since_last = now - self.light_states[light_id]['last_person_time']
                if time_since_last >= self.light_off_delay:
                    self._set_light(light_id, False)
        
        # 3. 返回当前状态
        return {lid: info['state'] for lid, info in self.light_states.items()}
    
    def get_active_lights(self) -> List[str]:
        """获取当前开启的灯列表"""
        return [lid for lid, info in self.light_states.items() if info['state']]
    
    def get_person_location_info(self, foot_point: Tuple[int, int]) -> Dict:
        """
        获取人形位置的详细信息
        
        Returns:
            {
                'current_zone': 当前所在区域ID或None,
                'nearest_zone': 最近区域ID,
                'lights_should_on': 应开启的灯列表,
                'distance_to_center': 到区域中心的距离
            }
        """
        current = self.config.find_zone_by_position(foot_point)
        nearest = self.config.find_nearest_zone(foot_point)
        lights = self.config.get_lights_for_person(foot_point, self.facing_direction)
        
        return {
            'current_zone': current.id if current else None,
            'nearest_zone': nearest.id if nearest else None,
            'lights_should_on': lights,
            'distance_to_center': nearest.distance_to_point(foot_point) if nearest else float('inf')
        }
    
    def force_light_on(self, light_id: str):
        """强制开启指定灯"""
        self._set_light(light_id, True)
        self.light_states[light_id]['last_person_time'] = time.time()
    
    def force_light_off(self, light_id: str):
        """强制关闭指定灯"""
        self._set_light(light_id, False)
    
    def force_all_off(self):
        """强制关闭所有灯"""
        for light_id in self.light_states.keys():
            self._set_light(light_id, False)
    
    def calibrate_from_frame(self, detections: List[Dict], 
                              zone_radius: int = 100,
                              save_path: str = None) -> bool:
        """
        从当前帧的人形位置校准灯光配置
        
        Args:
            detections: 检测结果列表
            zone_radius: 区域半径
            save_path: 保存配置文件的路径
        """
        if not detections:
            print("校准失败: 未检测到人形")
            return False
        
        person_detections = [d for d in detections if d.get('class') == 'person']
        if not person_detections:
            print("校准失败: 未检测到人形")
            return False
        
        # 清空现有配置
        self.config.zones.clear()
        self.light_states.clear()
        
        # 生成新配置
        zones = self.config.calibrate_from_detections(person_detections, zone_radius)
        
        # 初始化状态
        for zone in zones:
            self.light_states[zone.id] = {
                'state': False,
                'last_person_time': 0,
                'last_zone': None
            }
        
        print(f"\n校准完成! 生成了 {len(zones)} 个灯光区域:")
        for zone in zones:
            print(f"  [{zone.id}] {zone.name}: ({zone.x}, {zone.y}), "
                  f"半径{zone.radius}px")
        
        # 保存配置
        if save_path:
            self.config.save_to_file(save_path)
            print(f"配置已保存到: {save_path}")
        
        return True
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'zone_entries': self.stats['zone_entries'].copy(),
            'light_on_count': self.stats['light_on_count'].copy(),
            'light_on_time': {k: round(v, 2) for k, v in self.stats['light_on_time'].items()},
            'current_active': self.get_active_lights()
        }
        return stats
    
    def cleanup(self):
        """清理资源"""
        print("\n关闭所有灯光...")
        self.force_all_off()
        
        if self._gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
                print("GPIO清理完成")
            except:
                pass
