"""
电价API接口

提供电价数据获取、缓存和预测功能。
支持多种电价数据源，包括国家电网和第三方API。
"""

import logging
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


@dataclass
class ElectricityPrice:
    """电价数据点
    
    Attributes:
        timestamp: 时间戳
        price: 电价 (元/kWh)
        period: 时段类型 ("peak" | "shoulder" | "valley" | "critical")
    """
    timestamp: datetime
    price: float
    period: str = "shoulder"
    
    def __post_init__(self):
        """验证数据有效性"""
        if self.price < 0:
            raise ValueError(f"Price cannot be negative: {self.price}")
        valid_periods = ["peak", "shoulder", "valley", "critical", "flat"]
        if self.period not in valid_periods:
            raise ValueError(f"Invalid period: {self.period}. Must be one of {valid_periods}")


@dataclass
class PriceSchedule:
    """电价计划
    
    Attributes:
        prices: 电价列表
        forecast_horizon: 预测时长 (小时)
        source: 数据来源
        cached_at: 缓存时间
    """
    prices: List[ElectricityPrice] = field(default_factory=list)
    forecast_horizon: int = 24
    source: str = "default"
    cached_at: datetime = field(default_factory=datetime.now)
    
    def get_price_at(self, timestamp: datetime) -> Optional[ElectricityPrice]:
        """获取指定时间的电价
        
        Args:
            timestamp: 查询时间
        
        Returns:
            电价数据点，如果未找到则返回None
        """
        for price in self.prices:
            if price.timestamp.hour == timestamp.hour:
                return price
        return None
    
    def get_average_price(self) -> float:
        """计算平均电价
        
        Returns:
            平均电价 (元/kWh)
        """
        if not self.prices:
            return 0.0
        return sum(p.price for p in self.prices) / len(self.prices)
    
    def get_peak_valley_ratio(self) -> float:
        """计算峰谷比
        
        Returns:
            峰谷电价比
        """
        peak_prices = [p.price for p in self.prices if p.period == "peak"]
        valley_prices = [p.price for p in self.prices if p.period == "valley"]
        
        if not peak_prices or not valley_prices:
            return 1.0
        
        avg_peak = sum(peak_prices) / len(peak_prices)
        avg_valley = sum(valley_prices) / len(valley_prices)
        
        return avg_peak / avg_valley if avg_valley > 0 else 1.0


class PriceAPI:
    """电价API接口
    
    提供电价数据获取、缓存和降级方案。
    支持多种数据源，当API不可用时使用默认分时电价。
    
    Attributes:
        provider: 电价提供商 ("state_grid", "third_party", "default")
        cache_dir: 缓存目录
        cache_duration: 缓存有效期 (分钟)
    
    Example:
        >>> api = PriceAPI(provider="default")
        >>> current_price = api.get_current_price()
        >>> print(f"Current price: {current_price.price:.3f} 元/kWh")
        >>> 
        >>> schedule = api.get_price_schedule(hours=24)
        >>> print(f"Peak/Valley ratio: {schedule.get_peak_valley_ratio():.2f}")
    """
    
    # 默认分时电价配置 (中国工商业电价示例)
    DEFAULT_PRICES = {
        "valley": 0.3,    # 谷时电价 (23:00-7:00)
        "flat": 0.6,      # 平时电价
        "peak": 1.0,      # 峰时电价 (8:00-11:00, 13:00-17:00, 19:00-22:00)
        "critical": 1.5,  # 尖峰电价 (夏季部分时段)
    }
    
    # 时段定义 (小时 -> 时段类型)
    TIME_PERIODS = {
        0: "valley", 1: "valley", 2: "valley", 3: "valley",
        4: "valley", 5: "valley", 6: "valley", 7: "valley",
        8: "peak", 9: "peak", 10: "peak", 11: "peak",
        12: "flat", 13: "peak", 14: "peak", 15: "peak",
        16: "peak", 17: "peak", 18: "flat", 19: "peak",
        20: "peak", 21: "peak", 22: "peak", 23: "valley",
    }
    
    def __init__(
        self,
        provider: str = "default",
        cache_dir: Optional[str] = None,
        cache_duration: int = 60,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        region: str = "beijing"
    ):
        """
        初始化电价API
        
        Args:
            provider: 电价提供商 ("state_grid", "third_party", "default")
            cache_dir: 缓存目录路径
            cache_duration: 缓存有效期 (分钟)
            api_key: API密钥 (用于第三方API)
            api_endpoint: API端点URL
            region: 地区代码
        """
        self.provider = provider
        self.cache_duration = cache_duration
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.region = region
        
        # 设置缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".energy_storage" / "price_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self._cache: Optional[PriceSchedule] = None
        self._cache_timestamp: Optional[datetime] = None
        
        logger.info(f"PriceAPI initialized with provider: {provider}, region: {region}")
    
    def get_current_price(self) -> ElectricityPrice:
        """获取当前电价
        
        Returns:
            当前电价数据
        """
        now = datetime.now()
        
        # 尝试从缓存或API获取
        schedule = self.get_price_schedule(hours=1)
        
        if schedule.prices:
            # 找到当前小时的电价
            for price in schedule.prices:
                if price.timestamp.hour == now.hour:
                    return price
        
        # 使用默认电价
        return self._get_default_price(now)
    
    def get_price_schedule(self, hours: int = 24) -> PriceSchedule:
        """获取未来电价计划
        
        Args:
            hours: 预测时长 (小时)
        
        Returns:
            电价计划
        """
        # 检查内存缓存
        if self._is_cache_valid():
            logger.debug("Using memory cache for price schedule")
            return self._cache
        
        # 检查文件缓存
        cached_schedule = self._load_from_file_cache()
        if cached_schedule:
            self._cache = cached_schedule
            self._cache_timestamp = datetime.now()
            return cached_schedule
        
        # 尝试从API获取
        try:
            if self.provider == "state_grid":
                schedule = self._fetch_from_state_grid(hours)
            elif self.provider == "third_party":
                schedule = self._fetch_from_third_party(hours)
            else:
                schedule = self._generate_default_schedule(hours)
            
            # 缓存结果
            self._cache = schedule
            self._cache_timestamp = datetime.now()
            self._save_to_file_cache(schedule)
            
            return schedule
            
        except Exception as e:
            logger.warning(f"Failed to fetch price from API: {e}. Using default schedule.")
            return self._generate_default_schedule(hours)
    
    def _is_cache_valid(self) -> bool:
        """检查内存缓存是否有效
        
        Returns:
            缓存是否有效
        """
        if self._cache is None or self._cache_timestamp is None:
            return False
        
        age = datetime.now() - self._cache_timestamp
        return age < timedelta(minutes=self.cache_duration)
    
    def _load_from_file_cache(self) -> Optional[PriceSchedule]:
        """从文件缓存加载电价数据
        
        Returns:
            缓存的电价计划，如果不存在或过期则返回None
        """
        cache_file = self.cache_dir / f"price_cache_{self.region}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            age = datetime.now() - cached_at
            
            if age >= timedelta(minutes=self.cache_duration):
                logger.debug("File cache expired")
                return None
            
            # 解析电价数据
            prices = []
            for p in data.get("prices", []):
                prices.append(ElectricityPrice(
                    timestamp=datetime.fromisoformat(p["timestamp"]),
                    price=p["price"],
                    period=p["period"]
                ))
            
            schedule = PriceSchedule(
                prices=prices,
                forecast_horizon=data.get("forecast_horizon", 24),
                source=data.get("source", "cache"),
                cached_at=cached_at
            )
            
            logger.debug("Loaded price schedule from file cache")
            return schedule
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_file_cache(self, schedule: PriceSchedule) -> None:
        """保存电价数据到文件缓存
        
        Args:
            schedule: 电价计划
        """
        cache_file = self.cache_dir / f"price_cache_{self.region}.json"
        
        try:
            data = {
                "prices": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "price": p.price,
                        "period": p.period
                    }
                    for p in schedule.prices
                ],
                "forecast_horizon": schedule.forecast_horizon,
                "source": schedule.source,
                "cached_at": datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug("Saved price schedule to file cache")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _fetch_from_state_grid(self, hours: int) -> PriceSchedule:
        """从国家电网API获取电价数据
        
        Note: 这是一个模拟实现，实际使用时需要替换为真实的API调用
        
        Args:
            hours: 预测时长
        
        Returns:
            电价计划
        """
        # TODO: 实现真实的国家电网API调用
        # 目前使用默认分时电价作为降级方案
        logger.info("Using default schedule (State Grid API not implemented)")
        return self._generate_default_schedule(hours)
    
    def _fetch_from_third_party(self, hours: int) -> PriceSchedule:
        """从第三方API获取电价数据
        
        Note: 这是一个模拟实现，实际使用时需要替换为真实的API调用
        
        Args:
            hours: 预测时长
        
        Returns:
            电价计划
        
        Raises:
            requests.RequestException: API调用失败
        """
        if not self.api_endpoint:
            logger.warning("No API endpoint configured, using default schedule")
            return self._generate_default_schedule(hours)
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            params = {
                "region": self.region,
                "hours": hours
            }
            
            response = requests.get(
                self.api_endpoint,
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # 解析API响应
            prices = []
            for item in data.get("prices", []):
                prices.append(ElectricityPrice(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    price=float(item["price"]),
                    period=item.get("period", "flat")
                ))
            
            return PriceSchedule(
                prices=prices,
                forecast_horizon=hours,
                source="third_party_api",
                cached_at=datetime.now()
            )
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def _generate_default_schedule(self, hours: int) -> PriceSchedule:
        """生成默认分时电价计划
        
        基于中国工商业分时电价政策生成。
        
        Args:
            hours: 预测时长
        
        Returns:
            电价计划
        """
        now = datetime.now()
        prices = []
        
        for i in range(hours):
            timestamp = now + timedelta(hours=i)
            hour = timestamp.hour
            period = self.TIME_PERIODS.get(hour, "flat")
            price = self.DEFAULT_PRICES.get(period, self.DEFAULT_PRICES["flat"])
            
            prices.append(ElectricityPrice(
                timestamp=timestamp,
                price=price,
                period=period
            ))
        
        return PriceSchedule(
            prices=prices,
            forecast_horizon=hours,
            source="default",
            cached_at=datetime.now()
        )
    
    def _get_default_price(self, timestamp: datetime) -> ElectricityPrice:
        """获取默认电价
        
        Args:
            timestamp: 时间戳
        
        Returns:
            电价数据点
        """
        hour = timestamp.hour
        period = self.TIME_PERIODS.get(hour, "flat")
        price = self.DEFAULT_PRICES.get(period, self.DEFAULT_PRICES["flat"])
        
        return ElectricityPrice(
            timestamp=timestamp,
            price=price,
            period=period
        )
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._cache = None
        self._cache_timestamp = None
        
        cache_file = self.cache_dir / f"price_cache_{self.region}.json"
        if cache_file.exists():
            cache_file.unlink()
        
        logger.info("Price cache cleared")
    
    def set_custom_prices(self, prices: Dict[str, float]) -> None:
        """设置自定义电价
        
        Args:
            prices: 电价字典，如 {"valley": 0.25, "peak": 1.2}
        """
        self.DEFAULT_PRICES.update(prices)
        logger.info(f"Updated default prices: {prices}")
    
    def get_price_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取电价统计信息
        
        Args:
            hours: 统计时长
        
        Returns:
            统计信息字典
        """
        schedule = self.get_price_schedule(hours)
        
        if not schedule.prices:
            return {}
        
        prices = [p.price for p in schedule.prices]
        
        return {
            "min_price": min(prices),
            "max_price": max(prices),
            "avg_price": sum(prices) / len(prices),
            "peak_valley_ratio": schedule.get_peak_valley_ratio(),
            "valley_hours": len([p for p in schedule.prices if p.period == "valley"]),
            "peak_hours": len([p for p in schedule.prices if p.period == "peak"]),
        }
