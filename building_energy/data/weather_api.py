"""
天气数据API接口

支持多个天气数据提供商：
- OpenWeatherMap: 全球覆盖，免费额度充足
- WeatherAPI.com: 国内访问快，支持中文
"""

import os
import json
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """天气数据结构"""
    timestamp: datetime
    temperature: float           # 温度 (°C)
    humidity: float             # 湿度 (%)
    pressure: float             # 气压 (hPa)
    wind_speed: float           # 风速 (m/s)
    wind_direction: float       # 风向 (度)
    solar_radiation: float      # 太阳辐射 (W/m²)，可能为估算值
    cloud_cover: float          # 云量 (%)
    description: str            # 天气描述
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'temperature': self.temperature,
            'humidity': self.humidity,
            'pressure': self.pressure,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'solar_radiation': self.solar_radiation,
            'cloud_cover': self.cloud_cover,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeatherData':
        """从字典创建"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            temperature=data['temperature'],
            humidity=data['humidity'],
            pressure=data['pressure'],
            wind_speed=data['wind_speed'],
            wind_direction=data['wind_direction'],
            solar_radiation=data['solar_radiation'],
            cloud_cover=data['cloud_cover'],
            description=data['description']
        )


class WeatherAPI(ABC):
    """
    天气API抽象基类
    
    提供统一的天气数据获取接口，支持缓存机制。
    """
    
    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 1  # 缓存有效期（小时）
    ):
        """
        初始化天气API
        
        Args:
            api_key: API密钥
            cache_dir: 缓存目录
            cache_ttl: 缓存有效期（小时）
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        
        # 设置缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.cache' / 'building_energy' / 'weather'
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BuildingEnergyManager/0.1.0'
        })
        
        logger.info(f"{self.__class__.__name__} initialized with cache at {self.cache_dir}")
    
    @abstractmethod
    def get_current_weather(self, lat: float, lon: float) -> WeatherData:
        """获取当前天气"""
        pass
    
    @abstractmethod
    def get_forecast(self, lat: float, lon: float, days: int = 7) -> List[WeatherData]:
        """获取天气预报"""
        pass
    
    def _get_cache_key(self, prefix: str, lat: float, lon: float) -> str:
        """生成缓存键"""
        key = f"{prefix}_{lat:.4f}_{lon:.4f}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        从缓存加载数据
        
        Args:
            cache_key: 缓存键
        
        Returns:
            缓存数据或None
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # 检查缓存是否过期
            cached_time = datetime.fromisoformat(cached['cached_at'])
            if datetime.now() - cached_time > timedelta(hours=self.cache_ttl):
                logger.debug(f"Cache expired for {cache_key}")
                return None
            
            logger.debug(f"Loaded from cache: {cache_key}")
            return cached['data']
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """
        保存数据到缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'data': data
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved to cache: {cache_key}")
            
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _make_request(
        self,
        url: str,
        params: Dict[str, Any],
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        发送HTTP请求（带重试）
        
        Args:
            url: 请求URL
            params: 请求参数
            max_retries: 最大重试次数
        
        Returns:
            JSON响应或None
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"Max retries exceeded for {url}")
                    return None
        
        return None


class OpenWeatherMapAPI(WeatherAPI):
    """
    OpenWeatherMap API实现
    
    文档: https://openweathermap.org/api
    """
    
    BASE_URL = "https://api.openweathermap.org/data/3.0"
    BASE_URL_2_5 = "https://api.openweathermap.org/data/2.5"
    
    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 1,
        api_version: str = "3.0",
        units: str = "metric",
        lang: str = "zh_cn"
    ):
        """
        初始化OpenWeatherMap API
        
        Args:
            api_key: API密钥
            cache_dir: 缓存目录
            cache_ttl: 缓存有效期（小时）
            api_version: API版本 ("2.5" 或 "3.0")
            units: 单位 ("metric", "imperial", "standard")
            lang: 语言代码
        """
        super().__init__(api_key, cache_dir, cache_ttl)
        self.api_version = api_version
        self.units = units
        self.lang = lang
    
    def get_current_weather(self, lat: float, lon: float) -> WeatherData:
        """
        获取当前天气
        
        Args:
            lat: 纬度
            lon: 经度
        
        Returns:
            WeatherData对象
        """
        cache_key = self._get_cache_key("current", lat, lon)
        
        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return WeatherData.from_dict(cached_data)
        
        # 使用One Call API 3.0
        if self.api_version == "3.0":
            url = f"{self.BASE_URL}/onecall"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': self.units,
                'lang': self.lang,
                'exclude': 'minutely,hourly,daily,alerts'
            }
        else:
            # 使用2.5版本API
            url = f"{self.BASE_URL_2_5}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': self.units,
                'lang': self.lang
            }
        
        data = self._make_request(url, params)
        if not data:
            raise ConnectionError("Failed to fetch weather data")
        
        # 解析数据
        if self.api_version == "3.0":
            current = data.get('current', {})
            weather_data = WeatherData(
                timestamp=datetime.now(),
                temperature=current.get('temp', 0),
                humidity=current.get('humidity', 0),
                pressure=current.get('pressure', 0),
                wind_speed=current.get('wind_speed', 0),
                wind_direction=current.get('wind_deg', 0),
                solar_radiation=self._estimate_solar_radiation(current),
                cloud_cover=current.get('clouds', 0),
                description=current.get('weather', [{}])[0].get('description', '')
            )
        else:
            weather_data = WeatherData(
                timestamp=datetime.now(),
                temperature=data.get('main', {}).get('temp', 0),
                humidity=data.get('main', {}).get('humidity', 0),
                pressure=data.get('main', {}).get('pressure', 0),
                wind_speed=data.get('wind', {}).get('speed', 0),
                wind_direction=data.get('wind', {}).get('deg', 0),
                solar_radiation=0,  # 2.5版本没有直接提供
                cloud_cover=data.get('clouds', {}).get('all', 0),
                description=data.get('weather', [{}])[0].get('description', '')
            )
        
        # 保存到缓存
        self._save_to_cache(cache_key, weather_data.to_dict())
        
        return weather_data
    
    def get_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 7
    ) -> List[WeatherData]:
        """
        获取天气预报
        
        Args:
            lat: 纬度
            lon: 经度
            days: 预报天数
        
        Returns:
            WeatherData列表
        """
        cache_key = self._get_cache_key(f"forecast_{days}", lat, lon)
        
        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return [WeatherData.from_dict(d) for d in cached_data]
        
        # One Call API 3.0
        url = f"{self.BASE_URL}/onecall"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': self.units,
            'lang': self.lang,
            'exclude': 'current,minutely,alerts'
        }
        
        data = self._make_request(url, params)
        if not data:
            raise ConnectionError("Failed to fetch forecast data")
        
        # 解析小时预报
        forecasts = []
        hourly_data = data.get('hourly', [])[:days*24]  # 限制天数
        
        for hour_data in hourly_data:
            forecast = WeatherData(
                timestamp=datetime.fromtimestamp(hour_data.get('dt', 0)),
                temperature=hour_data.get('temp', 0),
                humidity=hour_data.get('humidity', 0),
                pressure=hour_data.get('pressure', 0),
                wind_speed=hour_data.get('wind_speed', 0),
                wind_direction=hour_data.get('wind_deg', 0),
                solar_radiation=self._estimate_solar_radiation(hour_data),
                cloud_cover=hour_data.get('clouds', 0),
                description=hour_data.get('weather', [{}])[0].get('description', '')
            )
            forecasts.append(forecast)
        
        # 保存到缓存
        cache_list = [f.to_dict() for f in forecasts]
        self._save_to_cache(cache_key, cache_list)
        
        return forecasts
    
    def _estimate_solar_radiation(self, data: Dict[str, Any]) -> float:
        """
        估算太阳辐射
        
        基于云量和天气条件估算太阳辐射。
        
        Args:
            data: API返回的天气数据
        
        Returns:
            估算的太阳辐射 (W/m²)
        """
        # 简化的估算模型
        cloud_cover = data.get('clouds', 0) / 100.0  # 转换为0-1
        
        # 晴天最大辐射约1000 W/m²
        max_radiation = 1000
        
        # 根据云量调整
        estimated = max_radiation * (1 - 0.75 * cloud_cover)
        
        return max(0, estimated)


class WeatherAPICom(WeatherAPI):
    """
    WeatherAPI.com实现
    
    国内访问速度快，支持中文。
    文档: https://www.weatherapi.com/docs/
    """
    
    BASE_URL = "http://api.weatherapi.com/v1"
    
    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 1,
        lang: str = "zh"
    ):
        """
        初始化WeatherAPI.com
        
        Args:
            api_key: API密钥
            cache_dir: 缓存目录
            cache_ttl: 缓存有效期（小时）
            lang: 语言代码
        """
        super().__init__(api_key, cache_dir, cache_ttl)
        self.lang = lang
    
    def get_current_weather(self, lat: float, lon: float) -> WeatherData:
        """
        获取当前天气
        
        Args:
            lat: 纬度
            lon: 经度
        
        Returns:
            WeatherData对象
        """
        cache_key = self._get_cache_key("current", lat, lon)
        
        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return WeatherData.from_dict(cached_data)
        
        url = f"{self.BASE_URL}/current.json"
        params = {
            'key': self.api_key,
            'q': f"{lat},{lon}",
            'lang': self.lang,
            'aqi': 'yes'
        }
        
        data = self._make_request(url, params)
        if not data:
            raise ConnectionError("Failed to fetch weather data")
        
        current = data.get('current', {})
        
        weather_data = WeatherData(
            timestamp=datetime.now(),
            temperature=current.get('temp_c', 0),
            humidity=current.get('humidity', 0),
            pressure=current.get('pressure_mb', 0),
            wind_speed=current.get('wind_kph', 0) / 3.6,  # 转换为m/s
            wind_direction=current.get('wind_degree', 0),
            solar_radiation=current.get('uv', 0) * 100,  # UV指数估算
            cloud_cover=current.get('cloud', 0),
            description=current.get('condition', {}).get('text', '')
        )
        
        # 保存到缓存
        self._save_to_cache(cache_key, weather_data.to_dict())
        
        return weather_data
    
    def get_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 7
    ) -> List[WeatherData]:
        """
        获取天气预报
        
        Args:
            lat: 纬度
            lon: 经度
            days: 预报天数 (最多14天)
        
        Returns:
            WeatherData列表
        """
        cache_key = self._get_cache_key(f"forecast_{days}", lat, lon)
        
        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return [WeatherData.from_dict(d) for d in cached_data]
        
        url = f"{self.BASE_URL}/forecast.json"
        params = {
            'key': self.api_key,
            'q': f"{lat},{lon}",
            'days': min(days, 14),  # 最多14天
            'lang': self.lang,
            'aqi': 'yes'
        }
        
        data = self._make_request(url, params)
        if not data:
            raise ConnectionError("Failed to fetch forecast data")
        
        # 解析预报数据
        forecasts = []
        forecast_days = data.get('forecast', {}).get('forecastday', [])
        
        for day_data in forecast_days:
            # 获取小时数据
            for hour_data in day_data.get('hour', []):
                forecast = WeatherData(
                    timestamp=datetime.fromtimestamp(hour_data.get('time_epoch', 0)),
                    temperature=hour_data.get('temp_c', 0),
                    humidity=hour_data.get('humidity', 0),
                    pressure=hour_data.get('pressure_mb', 0),
                    wind_speed=hour_data.get('wind_kph', 0) / 3.6,
                    wind_direction=hour_data.get('wind_degree', 0),
                    solar_radiation=hour_data.get('uv', 0) * 100,
                    cloud_cover=hour_data.get('cloud', 0),
                    description=hour_data.get('condition', {}).get('text', '')
                )
                forecasts.append(forecast)
        
        # 保存到缓存
        cache_list = [f.to_dict() for f in forecasts]
        self._save_to_cache(cache_key, cache_list)
        
        return forecasts
