#!/usr/bin/env python3
"""
灯光控制器模块
支持Demo模式和实际硬件控制模式
"""
import time
import threading
from typing import Optional


class LightController:
    """灯光控制器"""
    
    def __init__(self, 
                 light_on_delay: float = 0,
                 light_off_delay: float = 5.0,
                 demo_mode: bool = True,
                 gpio_pin: int = 17):
        """
        Args:
            light_on_delay: 开灯延迟（秒）
            light_off_delay: 关灯延迟（秒）
            demo_mode: True=仅打印状态，False=控制GPIO
            gpio_pin: GPIO引脚号（树莓派）
        """
        self.light_on_delay = light_on_delay
        self.light_off_delay = light_off_delay
        self.demo_mode = demo_mode
        self.gpio_pin = gpio_pin
        
        self.light_state = False
        self.last_detection_time = 0
        self.last_empty_time = 0
        self.init_time = time.time()
        
        self._lock = threading.Lock()
        self._gpio_available = False
        
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
                print(f"GPIO初始化失败: {e}，切换到Demo模式")
                self.demo_mode = True
        
        if self.demo_mode:
            print("Demo模式: 仅显示灯光状态，不控制硬件")
        
        return True
    
    def _set_light(self, state: bool):
        """设置灯光状态"""
        with self._lock:
            if self.light_state == state:
                return
            
            self.light_state = state
            
            if self._gpio_available and not self.demo_mode:
                try:
                    import RPi.GPIO as GPIO
                    GPIO.output(self.gpio_pin, GPIO.HIGH if state else GPIO.LOW)
                except Exception as e:
                    print(f"GPIO控制失败: {e}")
            
            action = "开启" if state else "关闭"
            print(f"[灯光] {action} (Demo={self.demo_mode})")
    
    def update(self, person_detected: bool) -> bool:
        """
        更新检测状态并控制灯光
        
        Args:
            person_detected: 是否检测到人
        
        Returns:
            当前灯光状态
        """
        now = time.time()
        
        if person_detected:
            self.last_detection_time = now
            
            # 检测到人就立即开灯（或延迟开灯）
            if not self.light_state:
                if now - self.init_time >= self.light_on_delay:
                    self._set_light(True)
        else:
            self.last_empty_time = now
            
            # 人离开后延迟关灯
            if self.light_state:
                time_since_last_detection = now - self.last_detection_time
                if time_since_last_detection >= self.light_off_delay:
                    self._set_light(False)
        
        return self.light_state
    
    def force_on(self):
        """强制开灯"""
        self._set_light(True)
    
    def force_off(self):
        """强制关灯"""
        self._set_light(False)
    
    def cleanup(self):
        """清理资源"""
        self._set_light(False)
        
        if self._gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
                print("GPIO清理完成")
            except:
                pass
