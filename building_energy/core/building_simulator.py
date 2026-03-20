"""
建筑能耗模拟器

提供与EnergyPlus的接口，用于建筑能耗仿真和HVAC控制优化。
"""

import os
import logging
import subprocess
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

# 尝试导入eppy
try:
    from eppy import modeleditor
    from eppy.modeleditor import IDF
    EPPY_AVAILABLE = True
except ImportError:
    EPPY_AVAILABLE = False
    logging.warning("eppy not installed. IDF editing functionality will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class BuildingState:
    """建筑状态数据结构"""
    outdoor_temp: float          # 室外温度 (°C)
    indoor_temp: float           # 室内温度 (°C)
    indoor_humidity: float       # 室内湿度 (%)
    solar_radiation: float       # 太阳辐射 (W/m²)
    occupancy: int               # 人员数量
    hvac_power: float           # HVAC功耗 (kW)
    total_energy: float         # 累计能耗 (kWh)
    hour: int                   # 当前小时 (0-23)
    day_of_week: int            # 星期几 (0-6)
    is_holiday: bool            # 是否节假日
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([
            self.outdoor_temp,
            self.indoor_temp,
            self.indoor_humidity,
            self.solar_radiation,
            self.occupancy,
            self.hvac_power,
            self.hour,
            self.day_of_week,
            float(self.is_holiday)
        ], dtype=np.float32)


class BuildingSimulator:
    """
    建筑能耗模拟器
    
    使用EnergyPlus进行建筑能耗仿真，支持HVAC控制优化。
    
    Attributes:
        idf_path: IDF文件路径
        weather_path: 天气文件路径(EPW格式)
        output_dir: 输出目录
        timestep: 时间步长(分钟)
    """
    
    def __init__(
        self,
        idf_path: str,
        weather_path: str,
        output_dir: str = "output/simulation",
        timestep: int = 15,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化建筑模拟器
        
        Args:
            idf_path: EnergyPlus IDF文件路径
            weather_path: 天气文件路径(EPW格式)
            output_dir: 输出目录
            timestep: 时间步长(分钟)
            config: 额外配置参数
        """
        self.idf_path = idf_path
        self.weather_path = weather_path
        self.output_dir = output_dir
        self.timestep = timestep
        self.config = config or {}
        
        # 检查文件是否存在
        if not os.path.exists(idf_path):
            raise FileNotFoundError(f"IDF file not found: {idf_path}")
        if not os.path.exists(weather_path):
            raise FileNotFoundError(f"Weather file not found: {weather_path}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化状态
        self.current_step = 0
        self.current_time = datetime.now().replace(hour=0, minute=0, second=0)
        self.state = self._initialize_state()
        
        # HVAC控制参数
        self.min_setpoint = self.config.get('hvac', {}).get('min_setpoint', 18.0)
        self.max_setpoint = self.config.get('hvac', {}).get('max_setpoint', 26.0)
        self.target_temp = self.config.get('hvac', {}).get('target_temperature', 22.0)
        
        # 加载IDF文件
        self.idf = None
        if EPPY_AVAILABLE:
            try:
                self._load_idf()
            except Exception as e:
                logger.warning(f"Failed to load IDF file: {e}")
        
        logger.info(f"BuildingSimulator initialized with timestep={timestep}min")
    
    def _load_idf(self) -> None:
        """加载IDF文件"""
        # 查找IDD文件
        idd_path = self._find_idd_file()
        if idd_path:
            IDF.setiddname(idd_path)
            self.idf = IDF(self.idf_path)
            logger.info(f"Loaded IDF file: {self.idf_path}")
        else:
            logger.warning("IDD file not found. IDF editing disabled.")
    
    def _find_idd_file(self) -> Optional[str]:
        """查找EnergyPlus IDD文件"""
        # 常见IDD文件路径
        possible_paths = [
            "/usr/local/EnergyPlus-*/Energy+.idd",
            "C:/EnergyPlusV*/Energy+.idd",
            "/Applications/EnergyPlus-*/Energy+.idd",
            "./Energy+.idd"
        ]
        
        import glob
        for path_pattern in possible_paths:
            matches = glob.glob(path_pattern)
            if matches:
                return matches[0]
        
        return None
    
    def _initialize_state(self) -> BuildingState:
        """初始化建筑状态"""
        return BuildingState(
            outdoor_temp=25.0,
            indoor_temp=24.0,
            indoor_humidity=50.0,
            solar_radiation=200.0,
            occupancy=0,
            hvac_power=0.0,
            total_energy=0.0,
            hour=0,
            day_of_week=0,
            is_holiday=False
        )
    
    def reset(self) -> np.ndarray:
        """
        重置模拟器状态
        
        Returns:
            初始状态数组
        """
        self.current_step = 0
        self.current_time = datetime.now().replace(hour=0, minute=0, second=0)
        self.state = self._initialize_state()
        logger.info("Simulator reset")
        return self.state.to_array()
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一个时间步的模拟
        
        Args:
            action: 控制动作(温度设定点，单位°C)
        
        Returns:
            observation: 新的状态数组
            reward: 奖励值
            done: 是否结束
            info: 额外信息字典
        """
        # 限制动作范围
        setpoint = np.clip(action, self.min_setpoint, self.max_setpoint)
        
        # 更新建筑状态(简化模型)
        self._update_state(setpoint)
        
        # 计算奖励
        reward = self._calculate_reward(setpoint)
        
        # 检查是否结束
        self.current_step += 1
        max_steps = 24 * 60 // self.timestep  # 一天的步数
        done = self.current_step >= max_steps
        
        # 准备info
        info = {
            'setpoint': setpoint,
            'hvac_power': self.state.hvac_power,
            'total_energy': self.state.total_energy,
            'comfort_violation': abs(self.state.indoor_temp - self.target_temp) > 1.0
        }
        
        return self.state.to_array(), reward, done, info
    
    def _update_state(self, setpoint: float) -> None:
        """
        更新建筑状态(简化热力学模型)
        
        Args:
            setpoint: 温度设定点
        """
        # 更新时间
        self.current_time += timedelta(minutes=self.timestep)
        
        # 简化的热平衡模型
        # dT/dt = (outdoor_temp - indoor_temp) / tau + solar_gain - hvac_cooling
        
        # 时间常数(建筑热惯性)
        tau = 120.0  # 分钟
        
        # 太阳辐射增益(简化)
        hour = self.current_time.hour
        if 6 <= hour <= 18:
            solar_factor = np.sin((hour - 6) * np.pi / 12)
            self.state.solar_radiation = 800 * solar_factor
        else:
            self.state.solar_radiation = 0.0
        
        # 室外温度变化(简化日变化)
        self.state.outdoor_temp = 25 + 5 * np.sin((hour - 6) * np.pi / 12)
        
        # occupancy变化
        if 8 <= hour <= 18:
            self.state.occupancy = 20
        else:
            self.state.occupancy = 5
        
        # HVAC控制(简化PID)
        temp_error = self.state.indoor_temp - setpoint
        hvac_output = np.clip(temp_error * 0.5, 0, 1)  # 0-1范围
        self.state.hvac_power = hvac_output * 5.0  # 5kW额定功率
        
        # 室内温度更新
        temp_change = (
            (self.state.outdoor_temp - self.state.indoor_temp) / tau +
            self.state.solar_radiation * 0.001 +  # 太阳增益
            self.state.occupancy * 0.01 -  # 人员热量
            hvac_output * 0.1  # HVAC冷却
        ) * self.timestep
        
        self.state.indoor_temp += temp_change
        self.state.indoor_temp = np.clip(self.state.indoor_temp, 10, 40)
        
        # 湿度更新
        self.state.indoor_humidity = 50 + 10 * np.sin(hour * np.pi / 12)
        self.state.indoor_humidity = np.clip(self.state.indoor_humidity, 30, 70)
        
        # 累计能耗
        self.state.total_energy += self.state.hvac_power * self.timestep / 60
        
        # 更新时间特征
        self.state.hour = hour
        self.state.day_of_week = self.current_time.weekday()
        self.state.is_holiday = self.state.day_of_week >= 5
    
    def _calculate_reward(self, setpoint: float) -> float:
        """
        计算奖励函数
        
        Args:
            setpoint: 温度设定点
        
        Returns:
            奖励值
        """
        # 舒适度惩罚
        temp_deviation = abs(self.state.indoor_temp - self.target_temp)
        comfort_penalty = temp_deviation ** 2
        
        # 能耗惩罚
        energy_penalty = self.state.hvac_power * 0.1
        
        # 综合奖励(负值)
        reward = -(comfort_penalty + energy_penalty)
        
        return reward
    
    def get_current_state(self) -> BuildingState:
        """获取当前建筑状态"""
        return self.state
    
    def run_energyplus(self) -> bool:
        """
        运行EnergyPlus模拟(需要安装EnergyPlus)
        
        Returns:
            是否成功运行
        """
        try:
            # 检查energyplus是否可用
            result = subprocess.run(
                ['energyplus', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error("EnergyPlus not found")
                return False
            
            # 运行模拟
            cmd = [
                'energyplus',
                '-w', self.weather_path,
                '-d', self.output_dir,
                self.idf_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            
            if success:
                logger.info("EnergyPlus simulation completed successfully")
            else:
                logger.error(f"EnergyPlus simulation failed: {result.stderr}")
            
            return success
            
        except FileNotFoundError:
            logger.error("EnergyPlus not installed or not in PATH")
            return False
        except Exception as e:
            logger.error(f"Error running EnergyPlus: {e}")
            return False
