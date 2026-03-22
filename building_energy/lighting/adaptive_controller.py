#!/usr/bin/env python3
"""
预测性照明控制器

集成昼夜节律和运动预测功能，实现智能预测性照明控制。

主要功能:
- 基于昼夜节律的色温/亮度调节
- 基于运动预测的提前开灯
- 节能与体验平衡优化
- 与现有corridor_light系统集成
"""

from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time

from .circadian_rhythm import CircadianRhythm
from .motion_predictor import MotionPredictor, MotionEvent, ZoneLayout


@dataclass
class PredictiveConfig:
    """预测性控制配置"""
    # 预热阈值
    preheat_threshold: float = 10.0       # 预热提前时间 (秒)
    activate_threshold: float = 5.0       # 完全开启提前时间 (秒)
    
    # 亮度设置
    preheat_brightness: float = 0.3       # 预热亮度 (0-1)
    normal_brightness: float = 0.9        # 正常亮度 (0-1)
    
    # 超时设置
    preheat_timeout: float = 15.0         # 预热超时时间 (秒)
    light_off_delay: float = 5.0          # 关灯延迟 (秒)
    
    # 预测置信度
    min_prediction_confidence: float = 0.5  # 最小预测置信度
    
    # 误预测处理
    false_prediction_penalty: float = 2.0   # 误预测惩罚 (秒)
    max_consecutive_false: int = 3          # 最大连续误预测次数
    
    # 色温控制
    enable_circadian: bool = True         # 启用昼夜节律
    color_temp_transition_time: float = 2.0  # 色温过渡时间 (秒)


@dataclass
class ZoneState:
    """区域状态"""
    zone_id: str
    is_active: bool = False               # 是否完全开启
    is_preheating: bool = False           # 是否预热中
    brightness: float = 0.0               # 当前亮度
    color_temperature: int = 4500         # 当前色温
    last_person_time: float = 0.0         # 最后有人时间
    preheat_start_time: Optional[float] = None  # 预热开始时间
    predicted_arrivals: Dict[str, float] = field(default_factory=dict)  # 预测到达 {track_id: predicted_time}
    consecutive_false_predictions: int = 0  # 连续误预测次数


class PredictiveLightingController:
    """
    预测性照明控制器
    
    集成昼夜节律和运动预测，实现智能照明控制。
    
    Attributes:
        circadian: 昼夜节律控制器
        predictor: 运动预测器
        zone_controller: 现有区域控制器 (corridor_light)
        config: 预测性控制配置
    """
    
    def __init__(self,
                 circadian: CircadianRhythm,
                 predictor: MotionPredictor,
                 zone_controller=None,
                 config: Optional[PredictiveConfig] = None):
        """
        初始化预测性照明控制器
        
        Args:
            circadian: 昼夜节律控制器
            predictor: 运动预测器
            zone_controller: 现有区域控制器 (可选)
            config: 预测性控制配置
        """
        self.circadian = circadian
        self.predictor = predictor
        self.zone_controller = zone_controller
        self.config = config or PredictiveConfig()
        
        # 区域状态
        self.zone_states: Dict[str, ZoneState] = {}
        self._init_zone_states()
        
        # 锁
        self._lock = threading.Lock()
        
        # 回调函数
        self._on_light_change: Optional[Callable[[str, bool, float, int], None]] = None
        
        # 统计
        self.stats = {
            'preheat_count': 0,
            'activation_count': 0,
            'false_predictions': 0,
            'energy_saved': 0.0,
        }
        
        # 运行状态
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
    
    def _init_zone_states(self):
        """初始化区域状态"""
        for zone_id in self.predictor.layout.zones.keys():
            self.zone_states[zone_id] = ZoneState(zone_id=zone_id)
    
    def set_light_change_callback(self, 
                                   callback: Callable[[str, bool, float, int], None]):
        """
        设置灯光变化回调
        
        Args:
            callback: 回调函数(zone_id, is_active, brightness, color_temp)
        """
        self._on_light_change = callback
    
    def on_motion_detected(self, event: MotionEvent):
        """
        处理运动检测事件
        
        Args:
            event: 运动事件
        """
        with self._lock:
            now = time.time()
            zone_id = event.zone_id
            
            # 更新当前区域状态
            if zone_id in self.zone_states:
                state = self.zone_states[zone_id]
                state.last_person_time = now
                
                # 如果区域未激活，立即激活
                if not state.is_active:
                    self._activate_zone(zone_id)
            
            # 预测下一个区域
            predictions = self.predictor.predict_destination(event)
            
            for pred_zone, confidence, arrival_time in predictions:
                if confidence < self.config.min_prediction_confidence:
                    continue
                
                if pred_zone not in self.zone_states:
                    continue
                
                pred_state = self.zone_states[pred_zone]
                
                # 检查是否已在预测列表中
                if event.track_id in pred_state.predicted_arrivals:
                    continue
                
                # 根据到达时间决定操作
                if arrival_time <= self.config.activate_threshold:
                    # 立即完全开启
                    if not pred_state.is_active:
                        self._activate_zone(pred_zone)
                elif arrival_time <= self.config.preheat_threshold:
                    # 预热
                    if not pred_state.is_active and not pred_state.is_preheating:
                        self._preheat_zone(pred_zone)
                
                # 记录预测
                pred_state.predicted_arrivals[event.track_id] = now + arrival_time
    
    def _preheat_zone(self, zone_id: str):
        """
        预热区域 (低亮度)
        
        Args:
            zone_id: 区域ID
        """
        state = self.zone_states[zone_id]
        
        # 获取当前推荐的色温
        if self.config.enable_circadian:
            color_temp, _ = self.circadian.get_lighting_state()
        else:
            color_temp = 4500
        
        state.is_preheating = True
        state.brightness = self.config.preheat_brightness
        state.color_temperature = color_temp
        state.preheat_start_time = time.time()
        
        self.stats['preheat_count'] += 1
        
        # 触发回调
        if self._on_light_change:
            self._on_light_change(zone_id, False, state.brightness, color_temp)
        
        # 更新现有控制器
        if self.zone_controller:
            self._update_zone_controller(zone_id, True)
        
        print(f"[预测照明] 区域 {zone_id} 开始预热 (亮度: {state.brightness:.0%})")
    
    def _activate_zone(self, zone_id: str):
        """
        激活区域 (正常亮度)
        
        Args:
            zone_id: 区域ID
        """
        state = self.zone_states[zone_id]
        
        # 获取当前推荐的色温和亮度
        if self.config.enable_circadian:
            color_temp, brightness = self.circadian.get_lighting_state()
        else:
            color_temp = 4500
            brightness = self.config.normal_brightness
        
        was_preheating = state.is_preheating
        
        state.is_active = True
        state.is_preheating = False
        state.brightness = brightness
        state.color_temperature = color_temp
        state.last_person_time = time.time()
        state.preheat_start_time = None
        
        if not was_preheating:
            self.stats['activation_count'] += 1
        
        # 触发回调
        if self._on_light_change:
            self._on_light_change(zone_id, True, state.brightness, color_temp)
        
        # 更新现有控制器
        if self.zone_controller:
            self._update_zone_controller(zone_id, True)
        
        action = "从预热切换到正常" if was_preheating else "直接开启"
        print(f"[预测照明] 区域 {zone_id} {action} (亮度: {state.brightness:.0%}, 色温: {color_temp}K)")
    
    def _deactivate_zone(self, zone_id: str):
        """
        关闭区域
        
        Args:
            zone_id: 区域ID
        """
        state = self.zone_states[zone_id]
        
        if not state.is_active and not state.is_preheating:
            return
        
        # 计算能耗节省
        if state.is_preheating and state.preheat_start_time:
            preheat_duration = time.time() - state.preheat_start_time
            energy_saved = (self.config.normal_brightness - self.config.preheat_brightness) * preheat_duration
            self.stats['energy_saved'] += energy_saved
        
        state.is_active = False
        state.is_preheating = False
        state.brightness = 0.0
        state.predicted_arrivals.clear()
        
        # 触发回调
        if self._on_light_change:
            self._on_light_change(zone_id, False, 0.0, state.color_temperature)
        
        # 更新现有控制器
        if self.zone_controller:
            self._update_zone_controller(zone_id, False)
        
        print(f"[预测照明] 区域 {zone_id} 关闭")
    
    def _update_zone_controller(self, zone_id: str, state: bool):
        """
        更新现有区域控制器
        
        Args:
            zone_id: 区域ID
            state: 开启/关闭
        """
        if self.zone_controller is None:
            return
        
        try:
            # 尝试调用现有控制器的接口
            if hasattr(self.zone_controller, '_set_light'):
                self.zone_controller._set_light(zone_id, state)
            elif hasattr(self.zone_controller, 'set_light'):
                self.zone_controller.set_light(zone_id, state)
        except Exception as e:
            print(f"[警告] 更新区域控制器失败: {e}")
    
    def update(self):
        """更新控制器状态 (应定期调用)"""
        with self._lock:
            now = time.time()
            
            for zone_id, state in self.zone_states.items():
                # 检查预热超时
                if state.is_preheating and state.preheat_start_time:
                    if now - state.preheat_start_time > self.config.preheat_timeout:
                        # 预热超时，关闭
                        self._deactivate_zone(zone_id)
                        state.consecutive_false_predictions += 1
                        self.stats['false_predictions'] += 1
                        continue
                
                # 检查是否有人
                if state.is_active:
                    time_since_last = now - state.last_person_time
                    if time_since_last > self.config.light_off_delay:
                        # 无人超过延迟时间，关闭
                        self._deactivate_zone(zone_id)
                
                # 检查预测到达
                arrived_tracks = []
                for track_id, predicted_time in list(state.predicted_arrivals.items()):
                    if now >= predicted_time:
                        # 预测到达时间已过
                        arrived_tracks.append(track_id)
                        
                        # 如果区域仍未激活，说明预测错误
                        if not state.is_active and not state.is_preheating:
                            state.consecutive_false_predictions += 1
                            self.stats['false_predictions'] += 1
                            
                            # 如果连续误预测过多，延长预热时间
                            if state.consecutive_false_predictions >= self.config.max_consecutive_false:
                                print(f"[预测照明] 区域 {zone_id} 连续误预测，延长预热时间")
                
                # 清理已到达的预测
                for track_id in arrived_tracks:
                    del state.predicted_arrivals[track_id]
                
                # 如果区域激活，重置误预测计数
                if state.is_active:
                    state.consecutive_false_predictions = 0
    
    def start(self):
        """启动控制器后台线程"""
        if self._running:
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        print("[预测照明] 控制器已启动")
    
    def stop(self):
        """停止控制器"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
        print("[预测照明] 控制器已停止")
    
    def _update_loop(self):
        """后台更新循环"""
        while self._running:
            self.update()
            time.sleep(0.1)  # 100ms 更新间隔
    
    def get_zone_state(self, zone_id: str) -> Optional[ZoneState]:
        """获取区域状态"""
        return self.zone_states.get(zone_id)
    
    def get_all_zone_states(self) -> Dict[str, ZoneState]:
        """获取所有区域状态"""
        return dict(self.zone_states)
    
    def get_active_zones(self) -> List[str]:
        """获取当前激活的区域列表"""
        return [zid for zid, state in self.zone_states.items() if state.is_active]
    
    def get_preheating_zones(self) -> List[str]:
        """获取当前预热的区域列表"""
        return [zid for zid, state in self.zone_states.items() if state.is_preheating]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        prediction_accuracy = self.predictor.get_prediction_accuracy()
        return {
            **self.stats,
            'prediction_accuracy': prediction_accuracy,
            'active_zones': len(self.get_active_zones()),
            'preheating_zones': len(self.get_preheating_zones()),
        }
    
    def force_activate(self, zone_id: str):
        """
        强制激活区域 (用于手动控制)
        
        Args:
            zone_id: 区域ID
        """
        with self._lock:
            if zone_id in self.zone_states:
                self._activate_zone(zone_id)
    
    def force_deactivate(self, zone_id: str):
        """
        强制关闭区域 (用于手动控制)
        
        Args:
            zone_id: 区域ID
        """
        with self._lock:
            if zone_id in self.zone_states:
                self._deactivate_zone(zone_id)
    
    def set_manual_override(self, 
                           color_temp: Optional[int] = None,
                           brightness: Optional[float] = None,
                           duration_minutes: int = 60):
        """
        设置手动覆盖昼夜节律
        
        Args:
            color_temp: 手动色温
            brightness: 手动亮度
            duration_minutes: 持续时间
        """
        self.circadian.set_manual_override(color_temp, brightness, duration_minutes)
        
        # 更新所有激活区域的色温/亮度
        with self._lock:
            for state in self.zone_states.values():
                if state.is_active or state.is_preheating:
                    new_ct, new_br = self.circadian.get_lighting_state()
                    state.color_temperature = color_temp or new_ct
                    if state.is_active:
                        state.brightness = brightness or new_br
                    
                    if self._on_light_change:
                        self._on_light_change(
                            state.zone_id, 
                            state.is_active, 
                            state.brightness, 
                            state.color_temperature
                        )
    
    def clear_manual_override(self):
        """清除手动覆盖"""
        self.circadian.clear_manual_override()
