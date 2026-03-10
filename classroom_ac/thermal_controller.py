#!/usr/bin/env python3
"""
热负荷计算与预测性空调控制模块 v5.0

综合考虑:
- 人体在不同活动状态下的产热 (静止思考/轻度运动)
- 电子设备产热 (笔记本电脑等)
- 外部环境温度
- 历史人数趋势
- 课表和教室预约信息
- 提前预冷/预热策略
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class ThermalLoadConfig:
    """热负荷计算配置"""
    # 人体产热 (瓦特/人)
    HEAT_PERSON_RESTING = 80          # 静坐休息 (基础代谢)
    HEAT_PERSON_THINKING = 100        # 静坐思考 (上课/办公)
    HEAT_PERSON_LIGHT_EXERCISE = 150  # 轻度运动后 (刚进入教室)
    
    # 电子设备产热 (瓦特)
    HEAT_LAPTOP = 20                  # 笔记本电脑
    HEAT_DESKTOP = 80                 # 台式机
    HEAT_PROJECTOR = 200              # 投影仪
    
    # 建筑参数
    ROOM_AREA = 60                    # 教室面积 (平方米)
    CEILING_HEIGHT = 3.0              # 层高 (米)
    WALL_U_VALUE = 0.5                # 墙体传热系数 (W/m²·K)
    WINDOW_AREA = 10                  # 窗户面积 (平方米)
    WINDOW_U_VALUE = 2.5              # 窗户传热系数 (W/m²·K)
    
    # 空调参数
    AC_CAPACITY = 3500                # 空调额定制冷量 (W)
    AC_EER = 3.5                      # 能效比
    FAN_POWER = 50                    # 风扇功率 (W)
    
    # 预测参数
    PRE_COOL_TIME = 10                # 提前预冷时间 (分钟)
    PRE_HEAT_TIME = 5                 # 提前预热时间 (分钟)


class HeatLoadCalculator:
    """热负荷计算器"""
    
    def __init__(self, config: ThermalLoadConfig = None):
        self.config = config or ThermalLoadConfig()
        self.room_volume = self.config.ROOM_AREA * self.config.CEILING_HEIGHT
        
        # 历史数据 (用于趋势分析)
        self.people_history: deque = deque(maxlen=60)  # 60分钟历史
        self.outdoor_temp_history: deque = deque(maxlen=60)
        self.load_history: deque = deque(maxlen=60)
    
    def calculate_person_heat(self, 
                             person_count: int, 
                             activity_level: str = 'thinking') -> float:
        """
        计算人体产热
        
        Args:
            person_count: 人数
            activity_level: 活动级别
                - 'resting': 静坐休息 (80W/人)
                - 'thinking': 静坐思考 (100W/人)
                - 'light_exercise': 轻度运动 (150W/人)
        
        Returns:
            人体总产热 (W)
        """
        heat_rates = {
            'resting': self.config.HEAT_PERSON_RESTING,
            'thinking': self.config.HEAT_PERSON_THINKING,
            'light_exercise': self.config.HEAT_PERSON_LIGHT_EXERCISE
        }
        
        rate = heat_rates.get(activity_level, self.config.HEAT_PERSON_THINKING)
        return person_count * rate
    
    def calculate_equipment_heat(self, 
                                laptop_count: int = 0,
                                desktop_count: int = 0,
                                projector_on: bool = False) -> float:
        """
        计算设备产热
        
        Returns:
            设备总产热 (W)
        """
        heat = 0.0
        heat += laptop_count * self.config.HEAT_LAPTOP
        heat += desktop_count * self.config.HEAT_DESKTOP
        if projector_on:
            heat += self.config.HEAT_PROJECTOR
        return heat
    
    def calculate_envelope_heat(self,
                               outdoor_temp: float,
                               indoor_temp: float) -> float:
        """
        计算围护结构传热
        
        Args:
            outdoor_temp: 室外温度 (°C)
            indoor_temp: 室内温度 (°C)
        
        Returns:
            传热负荷 (W), 正值表示传入热量
        """
        # 简化计算：墙体+窗户
        wall_area = self.config.ROOM_AREA * 2.5  # 估算墙体面积
        wall_heat = wall_area * self.config.WALL_U_VALUE * (outdoor_temp - indoor_temp)
        
        window_heat = self.config.WINDOW_AREA * self.config.WINDOW_U_VALUE * (outdoor_temp - indoor_temp)
        
        return wall_heat + window_heat
    
    def calculate_solar_heat(self, 
                            solar_radiation: float = 500,
                            window_shading: float = 0.5) -> float:
        """
        计算太阳辐射得热
        
        Args:
            solar_radiation: 太阳辐射强度 (W/m²)
            window_shading: 遮阳系数 (0-1)
        
        Returns:
            太阳辐射得热 (W)
        """
        return self.config.WINDOW_AREA * solar_radiation * window_shading * 0.7
    
    def calculate_total_load(self,
                           person_count: int,
                           outdoor_temp: float,
                           indoor_temp: float,
                           laptop_count: int = 0,
                           desktop_count: int = 0,
                           projector_on: bool = False,
                           activity_level: str = 'thinking',
                           solar_radiation: float = 500) -> Dict:
        """
        计算总热负荷
        
        Returns:
            {
                'person_heat': 人体产热 (W),
                'equipment_heat': 设备产热 (W),
                'envelope_heat': 围护结构传热 (W),
                'solar_heat': 太阳辐射得热 (W),
                'total_load': 总热负荷 (W),
                'cooling_required': 需要的制冷量 (W)
            }
        """
        person_heat = self.calculate_person_heat(person_count, activity_level)
        equipment_heat = self.calculate_equipment_heat(laptop_count, desktop_count, projector_on)
        envelope_heat = self.calculate_envelope_heat(outdoor_temp, indoor_temp)
        solar_heat = self.calculate_solar_heat(solar_radiation)
        
        total_load = person_heat + equipment_heat + max(0, envelope_heat) + solar_heat
        
        # 记录历史
        self.people_history.append((datetime.now(), person_count))
        self.outdoor_temp_history.append((datetime.now(), outdoor_temp))
        self.load_history.append((datetime.now(), total_load))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'person_count': person_count,
            'outdoor_temp': outdoor_temp,
            'indoor_temp': indoor_temp,
            'person_heat': round(person_heat, 2),
            'equipment_heat': round(equipment_heat, 2),
            'envelope_heat': round(envelope_heat, 2),
            'solar_heat': round(solar_heat, 2),
            'total_load': round(total_load, 2),
            'cooling_required': round(max(0, total_load), 2),
            'activity_level': activity_level
        }
    
    def get_people_trend(self, minutes: int = 10) -> str:
        """
        分析人数趋势
        
        Returns:
            'increasing': 上升趋势
            'decreasing': 下降趋势
            'stable': 稳定
        """
        if len(self.people_history) < 2:
            return 'stable'
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = [p for t, p in self.people_history if t > cutoff]
        
        if len(recent) < 2:
            return 'stable'
        
        # 线性拟合
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        if slope > 0.5:
            return 'increasing'
        elif slope < -0.5:
            return 'decreasing'
        return 'stable'
    
    def predict_future_load(self, minutes_ahead: int = 10) -> float:
        """
        预测未来热负荷
        
        Args:
            minutes_ahead: 预测多少分钟后
        
        Returns:
            预测热负荷 (W)
        """
        if len(self.load_history) < 5:
            return self.config.AC_CAPACITY * 0.5  # 默认50%负荷
        
        # 简单线性预测
        recent_loads = [l for t, l in list(self.load_history)[-10:]]
        avg_load = np.mean(recent_loads)
        trend = self.get_people_trend()
        
        # 根据趋势调整
        if trend == 'increasing':
            return avg_load * 1.2
        elif trend == 'decreasing':
            return avg_load * 0.8
        return avg_load


class ScheduleManager:
    """课表和预约管理器"""
    
    def __init__(self, schedule_file: str = None):
        self.schedule_file = schedule_file
        self.class_schedule: Dict[str, List[Dict]] = {}  # {day: [{start, end, course, expected_people}]}
        self.bookings: List[Dict] = []  # 教室预约
        
        if schedule_file:
            self.load_schedule(schedule_file)
    
    def load_schedule(self, filepath: str):
        """加载课表"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.class_schedule = data.get('schedule', {})
                self.bookings = data.get('bookings', [])
        except Exception as e:
            print(f"加载课表失败: {e}")
    
    def is_class_time(self, dt: datetime = None) -> bool:
        """检查指定时间是否有课"""
        if dt is None:
            dt = datetime.now()
        
        day = dt.strftime('%A').lower()
        time_str = dt.strftime('%H:%M')
        
        for period in self.class_schedule.get(day, []):
            if period['start'] <= time_str <= period['end']:
                return True
        return False
    
    def get_next_class(self, dt: datetime = None) -> Optional[Dict]:
        """获取下一节课信息"""
        if dt is None:
            dt = datetime.now()
        
        day = dt.strftime('%A').lower()
        time_str = dt.strftime('%H:%M')
        
        classes_today = self.class_schedule.get(day, [])
        
        for period in classes_today:
            if period['start'] > time_str:
                return period
        return None
    
    def get_time_to_next_class(self, dt: datetime = None) -> Optional[int]:
        """距离下一节课还有多少分钟"""
        next_class = self.get_next_class(dt)
        if not next_class:
            return None
        
        if dt is None:
            dt = datetime.now()
        
        next_start = datetime.strptime(next_class['start'], '%H:%M')
        next_start = next_start.replace(year=dt.year, month=dt.month, day=dt.day)
        
        diff = (next_start - dt).total_seconds() / 60
        return int(diff) if diff > 0 else None
    
    def get_expected_people(self, dt: datetime = None) -> int:
        """获取预期人数"""
        if dt is None:
            dt = datetime.now()
        
        day = dt.strftime('%A').lower()
        time_str = dt.strftime('%H:%M')
        
        for period in self.class_schedule.get(day, []):
            if period['start'] <= time_str <= period['end']:
                return period.get('expected_people', 30)
        
        return 0


class PredictiveACController:
    """预测性空调控制器"""
    
    def __init__(self, 
                 heat_calculator: HeatLoadCalculator = None,
                 schedule_manager: ScheduleManager = None):
        self.heat_calc = heat_calculator or HeatLoadCalculator()
        self.schedule = schedule_manager or ScheduleManager()
        
        # 控制参数
        self.target_temp = 26.0
        self.temp_tolerance = 1.0
        self.indoor_temp = 28.0  # 假设初始温度
        self.outdoor_temp = 32.0  # 假设室外温度
        
        # 状态
        self.ac_state = False
        self.fan_state = False
        self.pre_cooling = False
        self.pre_heating = False
        
        # 统计
        self.energy_saved = 0.0  # Wh
    
    def update_environment(self, 
                          indoor_temp: float,
                          outdoor_temp: float,
                          person_count: int,
                          laptop_count: int = 0):
        """更新环境数据"""
        self.indoor_temp = indoor_temp
        self.outdoor_temp = outdoor_temp
        
        # 计算当前热负荷
        load_data = self.heat_calc.calculate_total_load(
            person_count=person_count,
            outdoor_temp=outdoor_temp,
            indoor_temp=indoor_temp,
            laptop_count=laptop_count
        )
        
        return load_data
    
    def make_decision(self) -> Dict:
        """
        做出空调控制决策
        
        Returns:
            {
                'ac_on': 是否开空调,
                'fan_on': 是否开风扇,
                'target_temp': 目标温度,
                'reason': 决策原因,
                'pre_action': 是否预动作,
                'estimated_load': 估计负荷 (W)
            }
        """
        now = datetime.now()
        decision = {
            'ac_on': False,
            'fan_on': False,
            'target_temp': self.target_temp,
            'reason': '',
            'pre_action': False,
            'estimated_load': 0
        }
        
        # 1. 检查是否有课/预约
        has_class = self.schedule.is_class_time(now)
        time_to_next = self.schedule.get_time_to_next_class(now)
        expected_people = self.schedule.get_expected_people(now)
        
        # 2. 分析趋势
        people_trend = self.heat_calc.get_people_trend(minutes=5)
        future_load = self.heat_calc.predict_future_load(minutes_ahead=10)
        
        # 3. 决策逻辑
        
        # 情况1: 正在上课
        if has_class:
            if self.indoor_temp > self.target_temp + self.temp_tolerance:
                decision['ac_on'] = True
                decision['reason'] = f'上课中，室内温度{self.indoor_temp:.1f}°C高于目标'
                
                # 根据人数调节温度
                if expected_people > 40:
                    decision['target_temp'] = 24.0
                elif expected_people > 20:
                    decision['target_temp'] = 25.0
            else:
                decision['fan_on'] = True
                decision['reason'] = '上课中，温度适宜，仅风扇'
        
        # 情况2: 即将上课 (预冷)
        elif time_to_next and time_to_next <= self.heat_calc.config.PRE_COOL_TIME:
            if self.indoor_temp > self.target_temp + 0.5:
                decision['ac_on'] = True
                decision['pre_action'] = True
                decision['target_temp'] = self.target_temp - 1.0  # 预冷时温度设低一点
                decision['reason'] = f'{time_to_next}分钟后上课，提前预冷'
        
        # 情况3: 下课了，人快走光了
        elif not has_class and people_trend == 'decreasing':
            if self.indoor_temp > self.target_temp + 2:
                decision['fan_on'] = True
                decision['reason'] = '下课了，人少，仅风扇维持'
            else:
                decision['reason'] = '下课了，温度适宜，关闭所有设备'
        
        # 情况4: 趋势上升，预判人多
        elif people_trend == 'increasing' and future_load > 2000:
            if self.indoor_temp > self.target_temp:
                decision['ac_on'] = True
                decision['reason'] = '人数快速增加，提前制冷'
        
        # 情况5: 空闲时段
        else:
            if self.indoor_temp > self.target_temp + 3:
                decision['fan_on'] = True
                decision['reason'] = '空闲时段，仅风扇节能'
            else:
                decision['reason'] = '空闲时段，温度适宜，关闭'
        
        decision['estimated_load'] = future_load
        return decision
    
    def get_status(self) -> Dict:
        """获取控制器状态"""
        return {
            'ac_on': self.ac_state,
            'fan_on': self.fan_state,
            'indoor_temp': self.indoor_temp,
            'target_temp': self.target_temp,
            'pre_cooling': self.pre_cooling,
            'energy_saved_wh': round(self.energy_saved, 2)
        }
