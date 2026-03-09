#!/usr/bin/env python3
"""
空调控制器模块
支持红外控制和继电器控制
包含智能决策逻辑
"""
import time
import threading
from enum import Enum
from typing import Optional
from datetime import datetime


class ACMode(Enum):
    """空调模式"""
    OFF = 0
    COOL = 1
    HEAT = 2
    FAN = 3
    DRY = 4


class ACController:
    """空调控制器"""
    
    def __init__(self, 
                 demo_mode: bool = True,
                 min_people: int = 3,
                 cooldown_minutes: float = 5.0,
                 gpio_pin: int = 18,
                 ir_gpio_pin: int = 23):
        """
        Args:
            demo_mode: True=仅显示状态，False=实际控制
            min_people: 开空调的最少人数
            cooldown_minutes: 关闭后等待时间（防频繁开关）
            gpio_pin: 继电器控制GPIO
            ir_gpio_pin: 红外发射GPIO
        """
        self.demo_mode = demo_mode
        self.min_people = min_people
        self.cooldown_minutes = cooldown_minutes
        self.gpio_pin = gpio_pin
        self.ir_gpio_pin = ir_gpio_pin
        
        # 状态
        self.is_on = False
        self.target_temp = 26  # 目标温度
        self.current_mode = ACMode.COOL
        self.fan_speed = 2  # 1-3
        
        # 时间记录
        self.last_on_time = None
        self.last_off_time = None
        self.state_change_count = 0
        
        # GPIO
        self._gpio_available = False
        self._lock = threading.Lock()
        
        # 智能策略参数
        self.people_threshold_low = min_people
        self.people_threshold_high = min_people * 2
        self.temperature_threshold_high = 28  # 高温阈值
        self.temperature_threshold_low = 22   # 低温阈值
    
    def init(self) -> bool:
        """初始化控制器"""
        if not self.demo_mode:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.gpio_pin, GPIO.OUT)
                GPIO.output(self.gpio_pin, GPIO.LOW)
                self._gpio_available = True
                print(f"GPIO {self.gpio_pin} 初始化成功")
            except ImportError:
                print("警告: RPi.GPIO未安装，切换到Demo模式")
                self.demo_mode = True
            except Exception as e:
                print(f"GPIO初始化失败: {e}")
                self.demo_mode = True
        
        if self.demo_mode:
            print("Demo模式: 仅显示空调状态")
        
        return True
    
    def _send_ir_command(self, command: str):
        """发送红外命令（需要ir库支持）"""
        # 实际部署时需要安装lirc或使用pigpio
        # 这里预留接口
        print(f"[IR] 发送命令: {command}")
    
    def _set_relay(self, state: bool):
        """设置继电器状态"""
        if self._gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.output(self.gpio_pin, GPIO.HIGH if state else GPIO.LOW)
            except Exception as e:
                print(f"GPIO控制失败: {e}")
    
    def turn_on(self, mode: ACMode = ACMode.COOL, temp: int = 26) -> bool:
        """开空调"""
        with self._lock:
            # 检查冷却时间
            if self.last_off_time:
                elapsed = (time.time() - self.last_off_time) / 60
                if elapsed < self.cooldown_minutes:
                    print(f"[空调] 冷却中，还需等待 {self.cooldown_minutes - elapsed:.1f} 分钟")
                    return False
            
            if not self.is_on:
                self.is_on = True
                self.current_mode = mode
                self.target_temp = temp
                self.last_on_time = time.time()
                self.state_change_count += 1
                
                if not self.demo_mode:
                    self._set_relay(True)
                    self._send_ir_command(f"ON_{mode.name}_{temp}")
                
                print(f"[空调] 开启 - 模式:{mode.name} 温度:{temp}°C")
                return True
            
            return False
    
    def turn_off(self) -> bool:
        """关空调"""
        with self._lock:
            if self.is_on:
                self.is_on = False
                self.last_off_time = time.time()
                self.state_change_count += 1
                
                if not self.demo_mode:
                    self._set_relay(False)
                    self._send_ir_command("OFF")
                
                runtime = (time.time() - self.last_on_time) / 60 if self.last_on_time else 0
                print(f"[空调] 关闭 - 本次运行 {runtime:.1f} 分钟")
                return True
            
            return False
    
    def adjust_power(self, people_count: int):
        """根据人数调节功率/温度"""
        if not self.is_on:
            return
        
        # 简单策略
        if people_count >= self.people_threshold_high:
            # 人多：降低温度，提高风速
            new_temp = 24
            new_fan = 3
        elif people_count >= self.people_threshold_low:
            # 适中
            new_temp = 26
            new_fan = 2
        else:
            # 人少：节能模式
            new_temp = 28
            new_fan = 1
        
        if new_temp != self.target_temp:
            self.target_temp = new_temp
            print(f"[空调] 调节温度至 {new_temp}°C")
            if not self.demo_mode:
                self._send_ir_command(f"TEMP_{new_temp}")
    
    def update(self, avg_people: float) -> bool:
        """
        根据人数更新空调状态
        
        Args:
            avg_people: 平均人数（建议取30秒滑动平均）
        
        Returns:
            当前空调状态
        """
        # 决策逻辑
        if avg_people >= self.min_people:
            # 应该开空调
            if not self.is_on:
                self.turn_on(ACMode.COOL, 26)
            else:
                self.adjust_power(int(avg_people))
        else:
            # 人太少，关空调
            if self.is_on:
                self.turn_off()
        
        return self.is_on
    
    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            'is_on': self.is_on,
            'mode': self.current_mode.name,
            'target_temp': self.target_temp,
            'state_changes': self.state_change_count,
            'demo_mode': self.demo_mode
        }
    
    def cleanup(self):
        """清理资源"""
        self.turn_off()
        
        if self._gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
                print("GPIO清理完成")
            except:
                pass
