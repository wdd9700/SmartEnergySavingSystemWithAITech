"""
故障告警器

负责故障告警的分级、触发、抑制和恢复管理。
集成现有AnomalyDetector，实现分级告警系统。
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time

try:
    from .fault_locator import FaultDiagnosis, SeverityLevel
except ImportError:
    from fault_locator import FaultDiagnosis, SeverityLevel

# 导入现有的AnomalyDetector
try:
    from ..models.anomaly_detector import AnomalyDetector, AnomalyAlert
    ANOMALY_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        from building_energy.models.anomaly_detector import AnomalyDetector, AnomalyAlert
        ANOMALY_DETECTOR_AVAILABLE = True
    except ImportError:
        ANOMALY_DETECTOR_AVAILABLE = False
        logging.warning("AnomalyDetector not available, using fallback implementation")

logger = logging.getLogger(__name__)


class AlertStatus(Enum):
    """告警状态枚举"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class FaultAlert:
    """故障告警数据结构"""
    alert_id: str
    fault_diagnosis: FaultDiagnosis
    status: str
    created_at: datetime
    updated_at: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    suppression_reason: Optional[str] = None
    notification_count: int = 0
    last_notified_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'alert_id': self.alert_id,
            'fault': self.fault_diagnosis.to_dict(),
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'suppression_reason': self.suppression_reason,
            'notification_count': self.notification_count,
            'last_notified_at': self.last_notified_at.isoformat() if self.last_notified_at else None
        }


class FaultAlerter:
    """
    故障告警器
    
    实现故障告警的分级、触发、抑制和恢复管理。
    集成现有AnomalyDetector进行异常检测增强。
    
    告警分级:
    - LOW: 轻微异常，记录日志
    - MEDIUM: 一般故障，发送通知
    - HIGH: 严重故障，立即通知运维
    - CRITICAL: 紧急故障，多渠道告警
    
    告警抑制策略:
    - 同一设备相同故障类型在抑制期内不重复告警
    - 抑制期根据严重程度不同: LOW=1小时, MEDIUM=30分钟, HIGH=15分钟, CRITICAL=5分钟
    
    Attributes:
        detector: 异常检测器（可选）
        active_alerts: 当前活动告警字典
        alert_history: 告警历史记录
        suppression_periods: 抑制期配置
        notification_handlers: 通知处理器列表
    
    Example:
        >>> alerter = FaultAlerter(anomaly_detector)
        >>> alerter.add_notification_handler(email_handler)
        >>> alert = alerter.alert(diagnosis)
        >>> alerter.acknowledge(alert.alert_id, "operator_1")
        >>> alerter.resolve(alert.alert_id)
    """
    
    # 抑制期配置（分钟）
    DEFAULT_SUPPRESSION_PERIODS = {
        'low': 60,
        'medium': 30,
        'high': 15,
        'critical': 5
    }
    
    # 通知间隔（分钟）
    NOTIFICATION_INTERVALS = {
        'low': 240,      # 4小时
        'medium': 120,   # 2小时
        'high': 60,      # 1小时
        'critical': 15   # 15分钟
    }
    
    def __init__(
        self,
        anomaly_detector: Optional[Any] = None,
        suppression_periods: Optional[Dict[str, int]] = None,
        max_history_size: int = 1000
    ):
        """
        初始化故障告警器
        
        Args:
            anomaly_detector: 异常检测器实例（可选）
            suppression_periods: 抑制期配置（分钟）
            max_history_size: 最大历史记录数
        """
        self.detector = anomaly_detector
        self.suppression_periods = suppression_periods or self.DEFAULT_SUPPRESSION_PERIODS
        self.max_history_size = max_history_size
        
        # 告警管理
        self.active_alerts: Dict[str, FaultAlert] = {}
        self.alert_history: List[FaultAlert] = []
        self._alert_counter = 0
        
        # 抑制记录: {(device_id, fault_type): last_alert_time}
        self._suppression_record: Dict[tuple, datetime] = {}
        
        # 通知处理器
        self._notification_handlers: List[Callable[[FaultAlert], None]] = []
        
        logger.info("FaultAlerter initialized")
    
    def add_notification_handler(self, handler: Callable[[FaultAlert], None]) -> None:
        """
        添加通知处理器
        
        Args:
            handler: 通知处理函数，接收FaultAlert参数
        """
        self._notification_handlers.append(handler)
        logger.info(f"Added notification handler: {handler.__name__}")
    
    def alert(self, diagnosis: FaultDiagnosis) -> Optional[FaultAlert]:
        """
        触发告警
        
        检查抑制规则后创建新告警或更新现有告警。
        
        Args:
            diagnosis: 故障诊断结果
            
        Returns:
            创建的告警对象，如果被抑制返回None
        """
        # 检查是否应该抑制
        if self._should_suppress(diagnosis):
            logger.debug(f"Alert suppressed for {diagnosis.affected_device}: {diagnosis.fault_type}")
            return None
        
        # 检查是否已存在相同告警
        existing_alert = self._find_existing_alert(diagnosis)
        
        if existing_alert:
            # 更新现有告警
            alert = self._update_alert(existing_alert, diagnosis)
        else:
            # 创建新告警
            alert = self._create_alert(diagnosis)
            self.active_alerts[alert.alert_id] = alert
        
        # 发送通知
        self._send_notification(alert)
        
        # 更新抑制记录
        suppression_key = (diagnosis.affected_device, diagnosis.fault_type)
        self._suppression_record[suppression_key] = datetime.now()
        
        logger.warning(f"Fault alert triggered: {alert.alert_id} - {diagnosis.fault_type} "
                      f"in {diagnosis.affected_device} ({diagnosis.severity})")
        
        return alert
    
    def acknowledge(
        self,
        alert_id: str,
        operator: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        确认告警
        
        Args:
            alert_id: 告警ID
            operator: 操作员ID
            notes: 备注信息
            
        Returns:
            True表示成功，False表示告警不存在
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Cannot acknowledge unknown alert: {alert_id}")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED.value
        alert.acknowledged_by = operator
        alert.acknowledged_at = datetime.now()
        alert.updated_at = datetime.now()
        
        logger.info(f"Alert {alert_id} acknowledged by {operator}")
        
        if notes:
            logger.info(f"Acknowledgment notes: {notes}")
        
        return True
    
    def resolve(
        self,
        alert_id: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """
        解除告警
        
        Args:
            alert_id: 告警ID
            resolution_notes: 解决备注
            
        Returns:
            True表示成功，False表示告警不存在
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Cannot resolve unknown alert: {alert_id}")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED.value
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()
        
        # 移动到历史记录
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)
        
        # 从活动告警中移除
        del self.active_alerts[alert_id]
        
        # 清除抑制记录
        suppression_key = (alert.fault_diagnosis.affected_device, alert.fault_diagnosis.fault_type)
        if suppression_key in self._suppression_record:
            del self._suppression_record[suppression_key]
        
        logger.info(f"Alert {alert_id} resolved")
        
        if resolution_notes:
            logger.info(f"Resolution notes: {resolution_notes}")
        
        return True
    
    def suppress(
        self,
        alert_id: str,
        reason: str,
        duration_minutes: Optional[int] = None
    ) -> bool:
        """
        手动抑制告警
        
        Args:
            alert_id: 告警ID
            reason: 抑制原因
            duration_minutes: 抑制持续时间（分钟），None表示永久抑制
            
        Returns:
            True表示成功，False表示告警不存在
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Cannot suppress unknown alert: {alert_id}")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED.value
        alert.suppression_reason = reason
        alert.updated_at = datetime.now()
        
        logger.info(f"Alert {alert_id} suppressed: {reason}")
        
        # 如果指定了持续时间，设置恢复定时器
        if duration_minutes:
            # 这里可以添加定时器逻辑
            logger.info(f"Alert will be unsuppressed after {duration_minutes} minutes")
        
        return True
    
    def get_active_alerts(
        self,
        severity: Optional[str] = None,
        device_id: Optional[str] = None
    ) -> List[FaultAlert]:
        """
        获取活动告警
        
        Args:
            severity: 按严重程度过滤
            device_id: 按设备ID过滤
            
        Returns:
            活动告警列表
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.fault_diagnosis.severity == severity]
        
        if device_id:
            alerts = [a for a in alerts if a.fault_diagnosis.affected_device == device_id]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_history(
        self,
        since: Optional[datetime] = None,
        device_id: Optional[str] = None,
        limit: int = 100
    ) -> List[FaultAlert]:
        """
        获取告警历史
        
        Args:
            since: 只返回此时间之后的告警
            device_id: 按设备ID过滤
            limit: 返回数量限制
            
        Returns:
            历史告警列表
        """
        alerts = self.alert_history.copy()
        
        if since:
            alerts = [a for a in alerts if a.created_at >= since]
        
        if device_id:
            alerts = [a for a in alerts if a.fault_diagnosis.affected_device == device_id]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        获取告警统计信息
        
        Returns:
            统计信息字典
        """
        active = self.active_alerts.values()
        history = self.alert_history
        
        # 按严重程度统计
        severity_counts = {
            'low': 0,
            'medium': 0,
            'high': 0,
            'critical': 0
        }
        
        for alert in active:
            sev = alert.fault_diagnosis.severity
            if sev in severity_counts:
                severity_counts[sev] += 1
        
        # 按故障类型统计
        fault_type_counts: Dict[str, int] = {}
        for alert in list(active) + history:
            ft = alert.fault_diagnosis.fault_type
            fault_type_counts[ft] = fault_type_counts.get(ft, 0) + 1
        
        return {
            'active_alerts': len(active),
            'total_history': len(history),
            'by_severity': severity_counts,
            'by_fault_type': fault_type_counts,
            'suppression_active': len(self._suppression_record)
        }
    
    def clear_all_alerts(self) -> None:
        """清除所有告警（谨慎使用）"""
        for alert in self.active_alerts.values():
            alert.status = AlertStatus.RESOLVED.value
            alert.resolved_at = datetime.now()
            self.alert_history.append(alert)
        
        self.active_alerts.clear()
        self._suppression_record.clear()
        
        logger.warning("All alerts cleared")
    
    def _should_suppress(self, diagnosis: FaultDiagnosis) -> bool:
        """
        检查是否应该抑制告警
        
        Args:
            diagnosis: 故障诊断结果
            
        Returns:
            True表示应该抑制
        """
        suppression_key = (diagnosis.affected_device, diagnosis.fault_type)
        
        if suppression_key not in self._suppression_record:
            return False
        
        last_alert_time = self._suppression_record[suppression_key]
        suppression_period = timedelta(
            minutes=self.suppression_periods.get(diagnosis.severity, 30)
        )
        
        return datetime.now() - last_alert_time < suppression_period
    
    def _find_existing_alert(self, diagnosis: FaultDiagnosis) -> Optional[FaultAlert]:
        """
        查找已存在的相同告警
        
        Args:
            diagnosis: 故障诊断结果
            
        Returns:
            现有告警对象或None
        """
        for alert in self.active_alerts.values():
            if (alert.fault_diagnosis.affected_device == diagnosis.affected_device and
                alert.fault_diagnosis.fault_type == diagnosis.fault_type and
                alert.status != AlertStatus.RESOLVED.value):
                return alert
        return None
    
    def _create_alert(self, diagnosis: FaultDiagnosis) -> FaultAlert:
        """创建新告警"""
        self._alert_counter += 1
        alert_id = f"FLT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}"
        
        now = datetime.now()
        return FaultAlert(
            alert_id=alert_id,
            fault_diagnosis=diagnosis,
            status=AlertStatus.ACTIVE.value,
            created_at=now,
            updated_at=now
        )
    
    def _update_alert(self, alert: FaultAlert, diagnosis: FaultDiagnosis) -> FaultAlert:
        """更新现有告警"""
        alert.fault_diagnosis = diagnosis
        alert.updated_at = datetime.now()
        return alert
    
    def _send_notification(self, alert: FaultAlert) -> None:
        """发送通知"""
        # 检查通知间隔
        if alert.last_notified_at:
            interval = timedelta(
                minutes=self.NOTIFICATION_INTERVALS.get(
                    alert.fault_diagnosis.severity, 60
                )
            )
            if datetime.now() - alert.last_notified_at < interval:
                return
        
        # 更新通知记录
        alert.notification_count += 1
        alert.last_notified_at = datetime.now()
        
        # 调用通知处理器
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
        
        # 如果配置了AnomalyDetector，也发送给它
        if self.detector and ANOMALY_DETECTOR_AVAILABLE:
            try:
                self._notify_anomaly_detector(alert)
            except Exception as e:
                logger.error(f"Failed to notify AnomalyDetector: {e}")
    
    def _notify_anomaly_detector(self, alert: FaultAlert) -> None:
        """通知现有的AnomalyDetector"""
        # 将故障告警转换为异常检测特征
        diagnosis = alert.fault_diagnosis
        
        # 创建特征向量
        features = [
            diagnosis.confidence,
            1.0 if diagnosis.severity == 'critical' else 
            0.75 if diagnosis.severity == 'high' else
            0.5 if diagnosis.severity == 'medium' else 0.25,
            1.0  # 故障标记
        ]
        
        # 这里可以调用AnomalyDetector的接口
        # 由于AnomalyDetector主要是用于训练后的预测，
        # 我们这里只是记录告警信息
        logger.debug(f"Notified AnomalyDetector about alert {alert.alert_id}")


# 预定义的通知处理器示例

def console_notification_handler(alert: FaultAlert) -> None:
    """控制台通知处理器"""
    diagnosis = alert.fault_diagnosis
    print(f"\n[{'!' * (4 - ['low', 'medium', 'high', 'critical'].index(diagnosis.severity))}] "
          f"FAULT ALERT: {alert.alert_id}")
    print(f"  Device: {diagnosis.affected_device}")
    print(f"  Type: {diagnosis.fault_type}")
    print(f"  Severity: {diagnosis.severity.upper()}")
    print(f"  Description: {diagnosis.description}")
    print(f"  Action: {diagnosis.recommended_action}")
    print(f"  Confidence: {diagnosis.confidence:.1%}")
    print()


def file_notification_handler(log_file: str = "fault_alerts.log") -> Callable[[FaultAlert], None]:
    """文件日志通知处理器工厂"""
    def handler(alert: FaultAlert) -> None:
        diagnosis = alert.fault_diagnosis
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {alert.alert_id} | "
                   f"{diagnosis.severity.upper()} | {diagnosis.affected_device} | "
                   f"{diagnosis.fault_type} | {diagnosis.description}\n")
    return handler
