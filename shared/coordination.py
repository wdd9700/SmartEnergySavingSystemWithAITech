#!/usr/bin/env python3
"""
多终端协调系统 (Multi-Node Coordination System)

解决多摄像头部署时的:
1. 节点发现与注册
2. 空间位置管理（避免重叠/盲区）
3. 数据融合（去重、补全）
4. 负载均衡

架构:
- 协调器 (Coordinator): 中心节点，管理所有边缘节点
- 边缘节点 (EdgeNode): 部署在各位置的检测单元
- 通信: MQTT或HTTP WebSocket
"""
import json
import time
import uuid
import threading
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque


class NodeStatus(Enum):
    """节点状态"""
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class NodePosition:
    """节点空间位置"""
    node_id: str
    x: float  # 米，相对坐标
    y: float
    z: float = 0
    direction: float = 0  # 朝向角度 (0-360度)
    fov_horizontal: float = 90  # 水平视场角
    fov_vertical: float = 60    # 垂直视场角
    detection_range: float = 10  # 检测距离(米)
    
    def get_coverage_area(self) -> Dict:
        """计算覆盖区域（简化扇形/矩形模型）"""
        # 简化：以节点为中心，视场角为扇形的覆盖区域
        return {
            'center': (self.x, self.y),
            'direction': self.direction,
            'fov': self.fov_horizontal,
            'range': self.detection_range
        }
    
    def distance_to(self, other: 'NodePosition') -> float:
        """计算与另一节点的距离"""
        return ((self.x - other.x) ** 2 + 
                (self.y - other.y) ** 2 + 
                (self.z - other.z) ** 2) ** 0.5
    
    def overlaps_with(self, other: 'NodePosition') -> bool:
        """检查覆盖区域是否重叠"""
        distance = self.distance_to(other)
        # 简化判断：距离小于检测范围之和则重叠
        return distance < (self.detection_range + other.detection_range)


@dataclass
class DetectionEvent:
    """检测事件"""
    event_id: str
    node_id: str
    timestamp: str
    object_type: str  # 'person', 'item', etc.
    object_id: str    # 跟踪ID
    position: Optional[tuple]  # 相对位置 (x, y)
    confidence: float
    metadata: Dict
    
    def to_dict(self) -> dict:
        return asdict(self)


class GlobalObjectTracker:
    """全局对象跟踪器
    
    跨摄像头跟踪同一对象，避免重复计数
    """
    
    def __init__(self, merge_threshold_seconds: float = 2.0,
                 merge_distance_meters: float = 2.0):
        self.merge_threshold = merge_threshold_seconds
        self.merge_distance = merge_distance_meters
        
        # 全局对象存储
        self.global_objects: Dict[str, Dict] = {}  # global_id -> object info
        self.object_history: deque = deque(maxlen=1000)  # 最近事件
        
        # 节点到全局ID的映射
        self.node_object_map: Dict[str, str] = {}  # (node_id, local_id) -> global_id
        
        self._lock = threading.Lock()
    
    def process_detection(self, event: DetectionEvent, 
                         node_position: NodePosition) -> str:
        """
        处理检测事件，返回全局对象ID
        
        去重逻辑:
        1. 检查同一节点是否已报告过此对象
        2. 检查相邻节点是否在相近时间报告过相似位置的对象
        3. 如果是同一对象，合并；否则创建新全局对象
        """
        with self._lock:
            local_key = f"{event.node_id}:{event.object_id}"
            
            # 1. 检查已知映射
            if local_key in self.node_object_map:
                global_id = self.node_object_map[local_key]
                self._update_global_object(global_id, event)
                return global_id
            
            # 2. 尝试与现有全局对象匹配
            matched_global_id = self._find_match(event, node_position)
            
            if matched_global_id:
                # 合并到现有对象
                self.node_object_map[local_key] = matched_global_id
                self._update_global_object(matched_global_id, event)
                return matched_global_id
            else:
                # 创建新全局对象
                global_id = f"global_{uuid.uuid4().hex[:8]}"
                self.global_objects[global_id] = {
                    'id': global_id,
                    'first_seen': event.timestamp,
                    'last_seen': event.timestamp,
                    'events': [event],
                    'node_history': [event.node_id],
                    'object_type': event.object_type
                }
                self.node_object_map[local_key] = global_id
                return global_id
    
    def _find_match(self, event: DetectionEvent, 
                   node_position: NodePosition) -> Optional[str]:
        """查找匹配的全局对象"""
        event_time = datetime.fromisoformat(event.timestamp)
        
        # 计算事件的空间位置
        if event.position:
            # 将相对位置转换为全局坐标
            global_x = node_position.x + event.position[0]
            global_y = node_position.y + event.position[1]
        else:
            # 无具体位置，使用节点位置
            global_x, global_y = node_position.x, node_position.y
        
        for global_id, obj in self.global_objects.items():
            # 时间检查
            last_seen = datetime.fromisoformat(obj['last_seen'])
            time_diff = (event_time - last_seen).total_seconds()
            
            if time_diff > self.merge_threshold:
                continue
            
            # 空间检查（检查是否来自相邻节点）
            last_node = obj['node_history'][-1]
            if last_node == event.node_id:
                # 同一节点，通常是同一对象
                if time_diff < 5.0:  # 5秒内认为是同一对象
                    return global_id
            else:
                # 不同节点，检查是否可能是同一对象（穿过区域边界）
                # 简化：如果在时间窗口内且节点相邻，可能是同一对象
                if time_diff < self.merge_threshold:
                    # 可以添加更复杂的空间一致性检查
                    return global_id
        
        return None
    
    def _update_global_object(self, global_id: str, event: DetectionEvent):
        """更新全局对象信息"""
        obj = self.global_objects[global_id]
        obj['last_seen'] = event.timestamp
        obj['events'].append(event)
        if event.node_id not in obj['node_history']:
            obj['node_history'].append(event.node_id)
    
    def get_global_count(self, object_type: str = 'person') -> int:
        """获取当前全局对象数量"""
        with self._lock:
            now = datetime.now()
            active_count = 0
            
            for obj in self.global_objects.values():
                if obj['object_type'] != object_type:
                    continue
                
                last_seen = datetime.fromisoformat(obj['last_seen'])
                if (now - last_seen).seconds < 30:  # 30秒内活跃
                    active_count += 1
            
            return active_count
    
    def cleanup_old_objects(self, max_age_seconds: float = 60):
        """清理过期对象"""
        with self._lock:
            now = datetime.now()
            expired = []
            
            for global_id, obj in self.global_objects.items():
                last_seen = datetime.fromisoformat(obj['last_seen'])
                if (now - last_seen).seconds > max_age_seconds:
                    expired.append(global_id)
            
            for global_id in expired:
                del self.global_objects[global_id]
                # 清理映射
                keys_to_remove = [k for k, v in self.node_object_map.items() 
                                 if v == global_id]
                for key in keys_to_remove:
                    del self.node_object_map[key]


class NodeCoordinator:
    """节点协调器
    
    管理多个边缘节点的注册、位置、状态
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # node_id -> node info
        self.node_positions: Dict[str, NodePosition] = {}
        self.global_tracker = GlobalObjectTracker()
        
        # 空间索引（简化网格）
        self.spatial_grid: Dict[tuple, Set[str]] = defaultdict(set)
        self.grid_size = 5.0  # 米
        
        self._lock = threading.Lock()
    
    def register_node(self, node_id: str, node_type: str,
                     position: NodePosition, capabilities: Dict) -> bool:
        """注册新节点"""
        with self._lock:
            # 检查节点ID冲突
            if node_id in self.nodes:
                return False
            
            # 检查位置冲突
            for existing_id, existing_pos in self.node_positions.items():
                if position.overlaps_with(existing_pos):
                    print(f"警告: 节点 {node_id} 与 {existing_id} 覆盖区域重叠")
            
            self.nodes[node_id] = {
                'id': node_id,
                'type': node_type,
                'status': NodeStatus.ONLINE,
                'registered_at': datetime.now().isoformat(),
                'last_heartbeat': datetime.now().isoformat(),
                'capabilities': capabilities,
                'stats': {
                    'detections_total': 0,
                    'errors': 0
                }
            }
            
            self.node_positions[node_id] = position
            self._add_to_spatial_grid(node_id, position)
            
            print(f"节点注册成功: {node_id} @ ({position.x}, {position.y})")
            return True
    
    def _add_to_spatial_grid(self, node_id: str, position: NodePosition):
        """添加到空间索引"""
        grid_x = int(position.x / self.grid_size)
        grid_y = int(position.y / self.grid_size)
        self.spatial_grid[(grid_x, grid_y)].add(node_id)
    
    def unregister_node(self, node_id: str):
        """注销节点"""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
            if node_id in self.node_positions:
                del self.node_positions[node_id]
            print(f"节点注销: {node_id}")
    
    def update_heartbeat(self, node_id: str, status: NodeStatus = None):
        """更新节点心跳"""
        if node_id in self.nodes:
            self.nodes[node_id]['last_heartbeat'] = datetime.now().isoformat()
            if status:
                self.nodes[node_id]['status'] = status
    
    def process_detection(self, event: DetectionEvent) -> str:
        """
        处理来自节点的检测事件
        
        Returns:
            全局对象ID
        """
        node_id = event.node_id
        
        if node_id not in self.node_positions:
            print(f"警告: 未注册节点 {node_id}")
            return None
        
        node_position = self.node_positions[node_id]
        global_id = self.global_tracker.process_detection(event, node_position)
        
        # 更新统计
        self.nodes[node_id]['stats']['detections_total'] += 1
        
        return global_id
    
    def get_neighbor_nodes(self, node_id: str, max_distance: float = 10.0) -> List[str]:
        """获取相邻节点"""
        if node_id not in self.node_positions:
            return []
        
        pos = self.node_positions[node_id]
        neighbors = []
        
        for other_id, other_pos in self.node_positions.items():
            if other_id != node_id and pos.distance_to(other_pos) <= max_distance:
                neighbors.append(other_id)
        
        return neighbors
    
    def get_coverage_gaps(self) -> List[Dict]:
        """检测覆盖盲区"""
        # 简化实现：检查网格中没有被任何节点覆盖的区域
        gaps = []
        
        # 获取所有有节点的网格
        covered_grids = set(self.spatial_grid.keys())
        
        if not covered_grids:
            return [{'message': '没有活跃节点', 'severity': 'high'}]
        
        # 检查边界区域
        min_x = min(g[0] for g in covered_grids)
        max_x = max(g[0] for g in covered_grids)
        min_y = min(g[1] for g in covered_grids)
        max_y = max(g[1] for g in covered_grids)
        
        # 简单检查：边界外可能存在盲区
        if min_x > 0:
            gaps.append({
                'area': f'x < {min_x * self.grid_size}m',
                'severity': 'medium'
            })
        
        return gaps
    
    def get_global_status(self) -> Dict:
        """获取全局状态"""
        return {
            'total_nodes': len(self.nodes),
            'online_nodes': sum(1 for n in self.nodes.values() 
                               if n['status'] == NodeStatus.ONLINE),
            'global_people_count': self.global_tracker.get_global_count('person'),
            'coverage_gaps': self.get_coverage_gaps()
        }
    
    def rebalance_load(self):
        """负载均衡建议"""
        # 分析各节点负载，建议调整
        recommendations = []
        
        for node_id, node in self.nodes.items():
            stats = node['stats']
            # 简化：如果某节点检测量过高，建议分担
            if stats['detections_total'] > 1000:
                neighbors = self.get_neighbor_nodes(node_id)
                if neighbors:
                    recommendations.append({
                        'node': node_id,
                        'action': 'reduce_sensitivity',
                        'reason': 'high_detection_rate'
                    })
        
        return recommendations


class EdgeNodeClient:
    """边缘节点客户端
    
    部署在各边缘设备上，与协调器通信
    """
    
    def __init__(self, node_id: str, coordinator_url: str,
                 position: NodePosition, capabilities: Dict):
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.position = position
        self.capabilities = capabilities
        
        self.is_registered = False
        self.heartbeat_interval = 10  # 秒
        self._stop_event = threading.Event()
    
    def register(self) -> bool:
        """向协调器注册"""
        # 实际实现中通过HTTP/MQTT发送注册请求
        print(f"节点 {self.node_id} 注册到协调器")
        self.is_registered = True
        return True
    
    def send_detection(self, object_type: str, local_id: str,
                      position: tuple = None, confidence: float = 0.5,
                      metadata: Dict = None) -> str:
        """发送检测事件"""
        if not self.is_registered:
            return None
        
        event = DetectionEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            node_id=self.node_id,
            timestamp=datetime.now().isoformat(),
            object_type=object_type,
            object_id=local_id,
            position=position,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # 实际实现中通过网络发送
        print(f"发送检测: {event.to_dict()}")
        return event.event_id
    
    def start_heartbeat(self):
        """启动心跳线程"""
        def heartbeat_loop():
            while not self._stop_event.is_set():
                if self.is_registered:
                    print(f"心跳: {self.node_id}")
                self._stop_event.wait(self.heartbeat_interval)
        
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
    
    def stop(self):
        """停止客户端"""
        self._stop_event.set()


# 使用示例
def example_usage():
    """使用示例"""
    # 创建协调器
    coordinator = NodeCoordinator()
    
    # 注册节点1（楼道西端）
    pos1 = NodePosition('node_corridor_west', 0, 0, 0, 90, 90, 60, 8)
    coordinator.register_node('node_corridor_west', 'corridor_light', pos1,
                              {'camera': True, 'gpio': True})
    
    # 注册节点2（楼道东端）
    pos2 = NodePosition('node_corridor_east', 15, 0, 0, 270, 90, 60, 8)
    coordinator.register_node('node_corridor_east', 'corridor_light', pos2,
                              {'camera': True, 'gpio': True})
    
    # 模拟检测事件
    event1 = DetectionEvent(
        event_id='e1',
        node_id='node_corridor_west',
        timestamp=datetime.now().isoformat(),
        object_type='person',
        object_id='local_001',
        position=(5, 0),  # 在节点前方5米
        confidence=0.85,
        metadata={}
    )
    
    global_id1 = coordinator.process_detection(event1)
    print(f"全局ID: {global_id1}")
    
    # 相邻节点报告（可能是同一个人穿过边界）
    time.sleep(1)
    event2 = DetectionEvent(
        event_id='e2',
        node_id='node_corridor_east',
        timestamp=datetime.now().isoformat(),
        object_type='person',
        object_id='local_001',
        position=(5, 0),
        confidence=0.82,
        metadata={}
    )
    
    global_id2 = coordinator.process_detection(event2)
    print(f"全局ID: {global_id2}")
    print(f"是否同一对象: {global_id1 == global_id2}")
    
    # 全局统计
    print(f"全局人数: {coordinator.global_tracker.get_global_count('person')}")


if __name__ == "__main__":
    example_usage()
