#!/usr/bin/env python3
"""
创新用途2: 能耗数据上报与分析系统

功能:
- 收集各设备运行数据
- 生成能耗报表
- 异常检测（如空调非工作时间运行）
- 提供API供其他系统查询

数据可上报至:
- 学校能耗管理平台
- 节能竞赛数据平台
- 自定义Dashboard
"""
import json
import time
import sqlite3
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from collections import defaultdict


@dataclass
class EnergyEvent:
    """能耗事件"""
    timestamp: str
    device_type: str  # 'light' | 'ac'
    device_id: str
    event_type: str   # 'on' | 'off' | 'power_change'
    value: float      # 功率或状态值
    people_count: int = 0
    zone: str = ''
    metadata: Dict = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class EnergyDatabase:
    """能耗数据库"""
    
    def __init__(self, db_path: str = "data/energy.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS energy_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_type TEXT NOT NULL,
                device_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                value REAL,
                people_count INTEGER,
                zone TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                device_type TEXT,
                total_kwh REAL,
                runtime_hours REAL,
                avg_people REAL,
                efficiency_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_event(self, event: EnergyEvent):
        """记录事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO energy_events 
            (timestamp, device_type, device_id, event_type, value, people_count, zone, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp,
            event.device_type,
            event.device_id,
            event.event_type,
            event.value,
            event.people_count,
            event.zone,
            json.dumps(event.metadata) if event.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_events(self, start_time: str, end_time: str, 
                   device_type: Optional[str] = None) -> List[EnergyEvent]:
        """查询事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if device_type:
            cursor.execute('''
                SELECT * FROM energy_events 
                WHERE timestamp BETWEEN ? AND ? AND device_type = ?
                ORDER BY timestamp
            ''', (start_time, end_time, device_type))
        else:
            cursor.execute('''
                SELECT * FROM energy_events 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start_time, end_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            events.append(EnergyEvent(
                timestamp=row[1],
                device_type=row[2],
                device_id=row[3],
                event_type=row[4],
                value=row[5],
                people_count=row[6],
                zone=row[7],
                metadata=json.loads(row[8]) if row[8] else None
            ))
        
        return events
    
    def get_daily_report(self, date: str) -> Dict:
        """获取日报"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 计算各项指标
        start = f"{date} 00:00:00"
        end = f"{date} 23:59:59"
        
        cursor.execute('''
            SELECT device_type, 
                   SUM(CASE WHEN event_type = 'on' THEN 1 ELSE 0 END) as on_count,
                   SUM(CASE WHEN event_type = 'off' THEN 1 ELSE 0 END) as off_count,
                   AVG(people_count) as avg_people
            FROM energy_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY device_type
        ''', (start, end))
        
        results = cursor.fetchall()
        conn.close()
        
        report = {}
        for row in results:
            device_type, on_count, off_count, avg_people = row
            report[device_type] = {
                'power_on_count': on_count or 0,
                'power_off_count': off_count or 0,
                'avg_people': round(avg_people or 0, 2)
            }
        
        return report


class EnergyReporter:
    """能耗报告生成器"""
    
    def __init__(self, db: EnergyDatabase):
        self.db = db
    
    def generate_hourly_report(self, date: str) -> str:
        """生成小时级报告 (Markdown格式)"""
        lines = [
            f"# 能耗报告 - {date}",
            "",
            "## 按小时统计",
            "",
            "| 小时 | 灯开次数 | 灯关次数 | 空调开次数 | 空调关次数 | 平均人数 |",
            "|------|----------|----------|------------|------------|----------|"
        ]
        
        for hour in range(24):
            start = f"{date} {hour:02d}:00:00"
            end = f"{date} {hour:02d}:59:59"
            
            events = self.db.get_events(start, end)
            
            light_on = sum(1 for e in events if e.device_type == 'light' and e.event_type == 'on')
            light_off = sum(1 for e in events if e.device_type == 'light' and e.event_type == 'off')
            ac_on = sum(1 for e in events if e.device_type == 'ac' and e.event_type == 'on')
            ac_off = sum(1 for e in events if e.device_type == 'ac' and e.event_type == 'off')
            avg_people = sum(e.people_count for e in events) / max(len(events), 1)
            
            lines.append(f"| {hour:02d}:00 | {light_on} | {light_off} | {ac_on} | {ac_off} | {avg_people:.1f} |")
        
        return '\n'.join(lines)
    
    def generate_comparison_report(self, date1: str, date2: str) -> str:
        """生成对比报告"""
        report1 = self.db.get_daily_report(date1)
        report2 = self.db.get_daily_report(date2)
        
        lines = [
            f"# 能耗对比报告",
            "",
            f"| 指标 | {date1} | {date2} | 变化 |",
            "|------|----------|----------|------|"
        ]
        
        # 对比各项数据
        for device_type in set(list(report1.keys()) + list(report2.keys())):
            data1 = report1.get(device_type, {})
            data2 = report2.get(device_type, {})
            
            on1 = data1.get('power_on_count', 0)
            on2 = data2.get('power_on_count', 0)
            change = on2 - on1
            change_pct = (change / on1 * 100) if on1 > 0 else 0
            
            lines.append(f"| {device_type} 开启次数 | {on1} | {on2} | {change:+.0f} ({change_pct:+.1f}%) |")
        
        return '\n'.join(lines)
    
    def detect_anomalies(self, date: str) -> List[str]:
        """检测异常"""
        anomalies = []
        
        # 检测非工作时间运行
        off_hours = ['00', '01', '02', '03', '04', '05', '06']
        for hour in off_hours:
            start = f"{date} {hour}:00:00"
            end = f"{date} {hour}:59:59"
            events = self.db.get_events(start, end)
            
            ac_events = [e for e in events if e.device_type == 'ac']
            if ac_events:
                anomalies.append(f"异常: 凌晨{hour}点空调运行")
        
        # 检测频繁开关
        events = self.db.get_events(f"{date} 00:00:00", f"{date} 23:59:59", 'ac')
        cycles = sum(1 for e in events if e.event_type == 'on')
        if cycles > 20:
            anomalies.append(f"警告: 空调频繁启停 ({cycles}次)，建议检查设置")
        
        return anomalies


class CloudUploader:
    """云端数据上传"""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.batch_size = 100
        self.buffer = []
    
    def add_event(self, event: EnergyEvent):
        """添加事件到缓冲区"""
        self.buffer.append(event.to_dict())
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """批量上传"""
        if not self.buffer:
            return
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.api_endpoint,
                json={'events': self.buffer},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"成功上传 {len(self.buffer)} 条数据")
                self.buffer = []
            else:
                print(f"上传失败: {response.status_code}")
        
        except Exception as e:
            print(f"上传错误: {e}")
    
    def close(self):
        """关闭时刷新缓冲区"""
        self.flush()


# 使用示例
def example_usage():
    """使用示例"""
    # 初始化
    db = EnergyDatabase()
    reporter = EnergyReporter(db)
    
    # 记录事件
    event = EnergyEvent(
        timestamp=datetime.now().isoformat(),
        device_type='ac',
        device_id='classroom_101',
        event_type='on',
        value=1500.0,  # 功率W
        people_count=15,
        zone='front',
        metadata={'temperature': 26.5, 'outside_temp': 32.0}
    )
    db.log_event(event)
    
    # 生成报告
    today = datetime.now().strftime('%Y-%m-%d')
    report = reporter.generate_hourly_report(today)
    print(report)
    
    # 检测异常
    anomalies = reporter.detect_anomalies(today)
    for anomaly in anomalies:
        print(anomaly)


if __name__ == "__main__":
    example_usage()
