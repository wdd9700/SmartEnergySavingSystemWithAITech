#!/usr/bin/env python3
"""
智能节能系统 - 综合管理界面 v6.0

展示内容:
- 校准后灯的相对位置 (来自亮度分析)
- 摄像头相对位置 (多摄像头校准)
- 预估人员位置 (实时检测+热力图)
- 教室所需制热/制冷量 (热负荷计算)
- 教室空调功耗 (能耗统计)

技术栈: Flask + WebSocket + Chart.js
"""
import sys
import json
import time
import threading
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError as e:
    FLASK_AVAILABLE = False
    warnings.warn(f"Flask not installed: {e}. Dashboard server will not be available.")
    # 创建占位符类以避免NameError
    class Flask:
        def __init__(self, *args, **kwargs): pass
    class SocketIO:
        def __init__(self, *args, **kwargs): pass
        def on(self, *args, **kwargs): return lambda f: f

try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError as e:
    CV_AVAILABLE = False
    warnings.warn(f"OpenCV or NumPy not installed: {e}")
    np = None

# 导入我们的模块（带错误处理）
try:
    from corridor_light.light_zones import LightConfig
    LIGHT_ZONES_AVAILABLE = True
except ImportError as e:
    LIGHT_ZONES_AVAILABLE = False
    LightConfig = None
    warnings.warn(f"corridor_light.light_zones not available: {e}")

try:
    from corridor_light.multi_camera_calibrator import MultiCameraCalibrator
    CALIBRATOR_AVAILABLE = True
except ImportError as e:
    CALIBRATOR_AVAILABLE = False
    MultiCameraCalibrator = None
    warnings.warn(f"corridor_light.multi_camera_calibrator not available: {e}")

try:
    from corridor_light.brightness_analyzer import BrightnessExtractor
    BRIGHTNESS_AVAILABLE = True
except ImportError as e:
    BRIGHTNESS_AVAILABLE = False
    BrightnessExtractor = None
    warnings.warn(f"corridor_light.brightness_analyzer not available: {e}")

try:
    from classroom_ac.thermal_controller import HeatLoadCalculator, PredictiveACController
    THERMAL_AVAILABLE = True
except ImportError as e:
    THERMAL_AVAILABLE = False
    HeatLoadCalculator = None
    PredictiveACController = None
    warnings.warn(f"classroom_ac.thermal_controller not available: {e}")

try:
    from shared.data_recorder import DataRecorder, HeatmapGenerator, EnergyEstimator
    DATA_RECORDER_AVAILABLE = True
except ImportError as e:
    DATA_RECORDER_AVAILABLE = False
    DataRecorder = None
    HeatmapGenerator = None
    EnergyEstimator = None
    warnings.warn(f"shared.data_recorder not available: {e}")


app = Flask(__name__)
# 从环境变量读取SECRET_KEY，如果没有则使用默认值（仅用于开发）
app.config['SECRET_KEY'] = os.environ.get('DASHBOARD_SECRET_KEY', 'smart-energy-dashboard-dev')
socketio = SocketIO(app, cors_allowed_origins="*")


class DashboardDataManager:
    """仪表盘数据管理器"""
    
    def __init__(self):
        # 灯光校准数据
        self.light_config: LightConfig = None
        self.camera_calibrator: MultiCameraCalibrator = None
        
        # 热负荷数据
        self.heat_calculator = HeatLoadCalculator()
        self.ac_controller = PredictiveACController()
        
        # 能耗统计
        self.energy_estimator = EnergyEstimator()
        
        # 实时数据缓存
        self.person_positions = deque(maxlen=100)
        self.thermal_load_history = deque(maxlen=60)
        self.power_consumption_history = deque(maxlen=60)
        
        # 当前状态
        self.current_status = {
            'timestamp': datetime.now().isoformat(),
            'indoor_temp': 26.0,
            'outdoor_temp': 30.0,
            'people_count': 0,
            'thermal_load': 0,
            'cooling_required': 0,
            'ac_power': 0,
            'fan_power': 0,
            'total_energy_wh': 0,
            'lights_status': {}
        }
        
        # 加载配置
        self._load_configurations()
    
    def _load_configurations(self):
        """加载所有配置文件"""
        # 加载灯光配置
        light_config_path = Path('corridor_light/light_config.json')
        if light_config_path.exists():
            try:
                self.light_config = LightConfig.load_from_file(str(light_config_path))
                print(f"已加载灯光配置: {len(self.light_config.zones)} 个区域")
            except Exception as e:
                print(f"加载灯光配置失败: {e}")
        
        # 加载摄像头校准
        camera_calib_path = Path('camera_calibration.json')
        if camera_calib_path.exists():
            try:
                self.camera_calibrator = MultiCameraCalibrator()
                self.camera_calibrator.load_calibration(str(camera_calib_path))
                print(f"已加载摄像头校准: {len(self.camera_calibrator.cameras)} 个摄像头")
            except Exception as e:
                print(f"加载摄像头校准失败: {e}")
    
    def update_detection_data(self, detections, camera_id='main'):
        """更新检测数据"""
        # 提取人员位置
        positions = []
        for det in detections:
            if det.get('class') == 'person':
                if 'foot_point' in det:
                    positions.append({
                        'x': det['foot_point'][0],
                        'y': det['foot_point'][1],
                        'camera': camera_id,
                        'timestamp': datetime.now().isoformat()
                    })
                elif 'bbox' in det:
                    # 从bbox计算脚底位置
                    x1, y1, x2, y2 = det['bbox']
                    positions.append({
                        'x': (x1 + x2) // 2,
                        'y': y2,
                        'camera': camera_id,
                        'timestamp': datetime.now().isoformat()
                    })
        
        self.person_positions.extend(positions)
        self.current_status['people_count'] = len(positions)
    
    def update_thermal_data(self, person_count, indoor_temp, outdoor_temp, 
                           laptop_count=0, activity='thinking'):
        """更新热负荷数据"""
        # 计算热负荷
        load_data = self.heat_calculator.calculate_total_load(
            person_count=person_count,
            outdoor_temp=outdoor_temp,
            indoor_temp=indoor_temp,
            laptop_count=laptop_count,
            activity_level=activity
        )
        
        # 更新控制器
        self.ac_controller.update_environment(
            indoor_temp=indoor_temp,
            outdoor_temp=outdoor_temp,
            person_count=person_count
        )
        
        decision = self.ac_controller.make_decision()
        
        # 计算功耗
        ac_power = 0
        fan_power = 0
        
        if decision['ac_on']:
            # 空调功耗根据负荷调节
            load_ratio = min(1.0, load_data['cooling_required'] / 3500)
            ac_power = 1000 + 2000 * load_ratio  # 1000W待机 + 变频功率
        
        if decision['fan_on']:
            fan_power = 50  # 风扇功耗
        
        # 更新能耗统计
        self.energy_estimator.update_light_state('ac', decision['ac_on'])
        
        # 保存历史
        timestamp = datetime.now()
        self.thermal_load_history.append({
            'timestamp': timestamp.isoformat(),
            'load': load_data['total_load'],
            'cooling_required': load_data['cooling_required'],
            'components': {
                'person': load_data['person_heat'],
                'equipment': load_data['equipment_heat'],
                'envelope': load_data['envelope_heat'],
                'solar': load_data['solar_heat']
            }
        })
        
        self.power_consumption_history.append({
            'timestamp': timestamp.isoformat(),
            'ac': ac_power,
            'fan': fan_power,
            'total': ac_power + fan_power
        })
        
        # 更新当前状态
        self.current_status.update({
            'timestamp': timestamp.isoformat(),
            'indoor_temp': indoor_temp,
            'outdoor_temp': outdoor_temp,
            'people_count': person_count,
            'thermal_load': load_data['total_load'],
            'cooling_required': load_data['cooling_required'],
            'ac_power': ac_power,
            'fan_power': fan_power,
            'total_energy_wh': self.energy_estimator.get_statistics()['total_energy_wh'],
            'ac_status': 'on' if decision['ac_on'] else 'off',
            'fan_status': 'on' if decision['fan_on'] else 'off',
            'decision_reason': decision['reason']
        })
    
    def get_light_positions(self):
        """获取灯光位置数据"""
        if not self.light_config:
            return []
        
        lights = []
        for zone in self.light_config.get_all_zones():
            lights.append({
                'id': zone.id,
                'name': zone.name,
                'x': zone.x,
                'y': zone.y,
                'radius': zone.radius,
                'forward_zones': zone.forward_zones,
                'backward_zones': zone.backward_zones
            })
        return lights
    
    def get_camera_positions(self):
        """获取摄像头相对位置"""
        if not self.camera_calibrator:
            return []
        
        cameras = []
        ref_cam = self.camera_calibrator.reference_camera
        
        for cam_id, camera in self.camera_calibrator.cameras.items():
            cam_data = {
                'id': cam_id,
                'is_reference': cam_id == ref_cam,
                'frame_shape': camera.frame_shape,
                'detected_lights': camera.detected_lights,
                'relative_offset': camera.relative_offset,
                'scale_factor': camera.scale_factor
            }
            cameras.append(cam_data)
        
        return cameras
    
    def get_person_positions(self, limit=50):
        """获取最近的人员位置"""
        return list(self.person_positions)[-limit:]
    
    def get_thermal_data(self, minutes=60):
        """获取热负荷历史"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            h for h in self.thermal_load_history
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]
    
    def get_power_data(self, minutes=60):
        """获取功耗历史"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            h for h in self.power_consumption_history
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]
    
    def get_global_status(self):
        """获取全局状态"""
        return {
            'lights': self.get_light_positions(),
            'cameras': self.get_camera_positions(),
            'current': self.current_status,
            'person_positions': self.get_person_positions(30),
            'thermal_history': self.get_thermal_data(30),
            'power_history': self.get_power_data(30)
        }


# 全局数据管理器
data_manager = DashboardDataManager()


@app.route('/')
def index():
    """主页面"""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """API: 获取当前状态"""
    return jsonify(data_manager.current_status)


@app.route('/api/lights')
def api_lights():
    """API: 获取灯光位置"""
    return jsonify(data_manager.get_light_positions())


@app.route('/api/cameras')
def api_cameras():
    """API: 获取摄像头位置"""
    return jsonify(data_manager.get_camera_positions())


@app.route('/api/persons')
def api_persons():
    """API: 获取人员位置"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify(data_manager.get_person_positions(limit))


@app.route('/api/thermal')
def api_thermal():
    """API: 获取热负荷数据"""
    minutes = request.args.get('minutes', 60, type=int)
    return jsonify({
        'history': data_manager.get_thermal_data(minutes),
        'current': {
            'load': data_manager.current_status['thermal_load'],
            'cooling_required': data_manager.current_status['cooling_required']
        }
    })


@app.route('/api/power')
def api_power():
    """API: 获取功耗数据"""
    minutes = request.args.get('minutes', 60, type=int)
    
    power_data = data_manager.get_power_data(minutes)
    
    # 计算统计
    if power_data:
        avg_ac = sum(p['ac'] for p in power_data) / len(power_data)
        avg_fan = sum(p['fan'] for p in power_data) / len(power_data)
        total = sum(p['total'] for p in power_data)
    else:
        avg_ac = avg_fan = total = 0
    
    return jsonify({
        'history': power_data,
        'statistics': {
            'avg_ac_power': round(avg_ac, 2),
            'avg_fan_power': round(avg_fan, 2),
            'total_energy_wh': data_manager.current_status['total_energy_wh']
        }
    })


@app.route('/api/global')
def api_global():
    """API: 获取所有全局数据"""
    return jsonify(data_manager.get_global_status())


@socketio.on('connect')
def handle_connect():
    """WebSocket连接"""
    print('客户端已连接')
    emit('status_update', data_manager.current_status)


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket断开"""
    print('客户端已断开')


def background_update_task():
    """后台更新任务"""
    while True:
        try:
            # 模拟数据更新 (实际应连接真实数据源)
            socketio.emit('status_update', data_manager.current_status)
            time.sleep(2)
        except Exception as e:
            print(f"后台任务错误: {e}")
            time.sleep(5)


def start_dashboard(host='0.0.0.0', port=5000, debug=False):
    """启动管理界面"""
    # 启动后台任务
    update_thread = threading.Thread(target=background_update_task, daemon=True)
    update_thread.start()
    
    print(f"=" * 60)
    print(f"智能节能系统管理界面 v6.0")
    print(f"=" * 60)
    print(f"访问地址: http://{host}:{port}")
    print(f"=" * 60)
    
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='智能节能系统管理界面')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    start_dashboard(args.host, args.port, args.debug)
