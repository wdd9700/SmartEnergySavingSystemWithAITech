"""
建筑智能节能系统 - 命令行接口

提供完整的CLI交互界面，支持：
- 系统启动/停止/状态查看
- 配置管理
- 实时监控
- 告警查看
- 知识库查询
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from building_energy.main import BuildingController, create_controller, SystemState
from building_energy.config.manager import ConfigManager, get_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLI:
    """
    命令行接口类
    
    提供交互式命令行界面和命令处理功能。
    """
    
    def __init__(self):
        self.controller: Optional[BuildingController] = None
        self.config: Optional[ConfigManager] = None
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """创建参数解析器"""
        parser = argparse.ArgumentParser(
            prog='beems',
            description='建筑智能节能系统 (Building Energy Management System)',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  %(prog)s init                          # 初始化系统配置
  %(prog)s start                         # 启动系统
  %(prog)s start -c config.yaml          # 使用指定配置启动
  %(prog)s status                        # 查看系统状态
  %(prog)s stop                          # 停止系统
  %(prog)s query "如何优化空调能耗？"      # 查询知识库
  %(prog)s alerts                        # 查看告警
  %(prog)s config get system.log_level   # 获取配置项
  %(prog)s config set hvac.target_temperature 24  # 设置配置项
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # init 命令
        init_parser = subparsers.add_parser('init', help='初始化系统配置')
        init_parser.add_argument(
            '-p', '--path',
            default='config.yaml',
            help='配置文件保存路径 (默认: config.yaml)'
        )
        init_parser.add_argument(
            '-f', '--force',
            action='store_true',
            help='强制覆盖已有配置'
        )
        
        # start 命令
        start_parser = subparsers.add_parser('start', help='启动系统')
        start_parser.add_argument(
            '-c', '--config',
            default='config.yaml',
            help='配置文件路径 (默认: config.yaml)'
        )
        start_parser.add_argument(
            '-d', '--duration',
            type=int,
            default=None,
            help='运行持续时间（秒），不指定则一直运行'
        )
        start_parser.add_argument(
            '--daemon',
            action='store_true',
            help='后台模式运行'
        )
        
        # stop 命令
        stop_parser = subparsers.add_parser('stop', help='停止系统')
        stop_parser.add_argument(
            '-t', '--timeout',
            type=int,
            default=30,
            help='关闭超时时间（秒）(默认: 30)'
        )
        
        # status 命令
        status_parser = subparsers.add_parser('status', help='查看系统状态')
        status_parser.add_argument(
            '-w', '--watch',
            action='store_true',
            help='持续监控模式'
        )
        status_parser.add_argument(
            '-i', '--interval',
            type=int,
            default=5,
            help='监控刷新间隔（秒）(默认: 5)'
        )
        
        # restart 命令
        restart_parser = subparsers.add_parser('restart', help='重启系统')
        restart_parser.add_argument(
            '-c', '--config',
            default='config.yaml',
            help='配置文件路径 (默认: config.yaml)'
        )
        
        # query 命令
        query_parser = subparsers.add_parser('query', help='查询知识库')
        query_parser.add_argument(
            'question',
            help='查询问题'
        )
        query_parser.add_argument(
            '-c', '--config',
            default='config.yaml',
            help='配置文件路径'
        )
        query_parser.add_argument(
            '-k', '--top-k',
            type=int,
            default=5,
            help='返回结果数量 (默认: 5)'
        )
        
        # alerts 命令
        alerts_parser = subparsers.add_parser('alerts', help='查看告警')
        alerts_parser.add_argument(
            '-s', '--severity',
            choices=['low', 'medium', 'high', 'critical'],
            help='按严重程度过滤'
        )
        alerts_parser.add_argument(
            '-l', '--limit',
            type=int,
            default=20,
            help='显示数量限制 (默认: 20)'
        )
        alerts_parser.add_argument(
            '-c', '--config',
            default='config.yaml',
            help='配置文件路径'
        )
        
        # config 命令
        config_parser = subparsers.add_parser('config', help='配置管理')
        config_subparsers = config_parser.add_subparsers(dest='config_action')
        
        # config get
        config_get = config_subparsers.add_parser('get', help='获取配置项')
        config_get.add_argument('key', help='配置键（如 system.log_level）')
        config_get.add_argument('-c', '--config', default='config.yaml', help='配置文件路径')
        
        # config set
        config_set = config_subparsers.add_parser('set', help='设置配置项')
        config_set.add_argument('key', help='配置键')
        config_set.add_argument('value', help='配置值')
        config_set.add_argument('-c', '--config', default='config.yaml', help='配置文件路径')
        
        # config list
        config_list = config_subparsers.add_parser('list', help='列出所有配置')
        config_list.add_argument('-c', '--config', default='config.yaml', help='配置文件路径')
        
        # config validate
        config_validate = config_subparsers.add_parser('validate', help='验证配置')
        config_validate.add_argument('-c', '--config', default='config.yaml', help='配置文件路径')
        
        # simulate 命令
        simulate_parser = subparsers.add_parser('simulate', help='运行仿真')
        simulate_parser.add_argument(
            '-c', '--config',
            default='config.yaml',
            help='配置文件路径'
        )
        simulate_parser.add_argument(
            '-d', '--days',
            type=int,
            default=1,
            help='仿真天数 (默认: 1)'
        )
        simulate_parser.add_argument(
            '--output',
            help='仿真结果输出路径'
        )
        
        # dashboard 命令
        dashboard_parser = subparsers.add_parser('dashboard', help='启动Web仪表盘')
        dashboard_parser.add_argument(
            '-c', '--config',
            default='config.yaml',
            help='配置文件路径'
        )
        dashboard_parser.add_argument(
            '-p', '--port',
            type=int,
            default=8080,
            help='服务端口 (默认: 8080)'
        )
        dashboard_parser.add_argument(
            '--host',
            default='0.0.0.0',
            help='服务地址 (默认: 0.0.0.0)'
        )
        
        # version 命令
        version_parser = subparsers.add_parser('version', help='显示版本信息')
        
        return parser
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        运行CLI
        
        Args:
            args: 命令行参数，None则使用sys.argv
        
        Returns:
            退出码
        """
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return 1
        
        try:
            return self._execute_command(parsed_args)
        except KeyboardInterrupt:
            print("\n操作已取消")
            return 130
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            return 1
    
    def _execute_command(self, args: argparse.Namespace) -> int:
        """执行命令"""
        command_map = {
            'init': self._cmd_init,
            'start': self._cmd_start,
            'stop': self._cmd_stop,
            'status': self._cmd_status,
            'restart': self._cmd_restart,
            'query': self._cmd_query,
            'alerts': self._cmd_alerts,
            'config': self._cmd_config,
            'simulate': self._cmd_simulate,
            'dashboard': self._cmd_dashboard,
            'version': self._cmd_version,
        }
        
        handler = command_map.get(args.command)
        if handler:
            return handler(args)
        else:
            print(f"未知命令: {args.command}")
            return 1
    
    def _cmd_init(self, args: argparse.Namespace) -> int:
        """初始化命令"""
        config_path = Path(args.path)
        
        if config_path.exists() and not args.force:
            print(f"配置文件已存在: {config_path}")
            print("使用 -f 或 --force 强制覆盖")
            return 1
        
        config = ConfigManager()
        config.create_default_config(str(config_path))
        print(f"配置文件已创建: {config_path}")
        return 0
    
    def _cmd_start(self, args: argparse.Namespace) -> int:
        """启动命令"""
        config_path = args.config
        
        if not Path(config_path).exists():
            print(f"配置文件不存在: {config_path}")
            print("请先运行: beems init")
            return 1
        
        print(f"启动建筑智能节能系统...")
        print(f"配置文件: {config_path}")
        
        try:
            self.controller = create_controller(config_path)
            self.controller.initialize()
            
            print("系统初始化完成")
            print(f"运行模式: {self.controller.config.controller.run_mode}")
            
            if args.daemon:
                print("以守护进程模式运行")
                # 可以在这里实现守护进程逻辑
            
            print("按 Ctrl+C 停止系统\n")
            
            self.controller.run(duration_seconds=args.duration)
            
            return 0
        
        except Exception as e:
            print(f"启动失败: {e}")
            return 1
    
    def _cmd_stop(self, args: argparse.Namespace) -> int:
        """停止命令"""
        print("停止系统...")
        
        # 这里可以实现通过PID文件或其他机制停止运行中的系统
        # 简化版本：提示用户按Ctrl+C
        print("请按 Ctrl+C 停止正在运行的系统")
        return 0
    
    def _cmd_status(self, args: argparse.Namespace) -> int:
        """状态命令"""
        config_path = getattr(args, 'config', 'config.yaml')
        
        try:
            self.config = get_config(config_path)
            self.config.load()
            
            if args.watch:
                self._watch_status(args.interval)
            else:
                self._print_status()
            
            return 0
        
        except Exception as e:
            print(f"获取状态失败: {e}")
            return 1
    
    def _print_status(self) -> None:
        """打印状态信息"""
        print("\n" + "=" * 60)
        print("建筑智能节能系统状态")
        print("=" * 60)
        
        # 系统信息
        print(f"\n系统信息:")
        print(f"  名称: {self.config.system.name}")
        print(f"  版本: {self.config.system.version}")
        print(f"  日志级别: {self.config.system.log_level}")
        
        # 运行配置
        print(f"\n运行配置:")
        print(f"  运行模式: {self.config.controller.run_mode}")
        print(f"  控制间隔: {self.config.controller.control_interval}秒")
        print(f"  数据收集间隔: {self.config.controller.data_collection_interval}秒")
        print(f"  预测_horizon: {self.config.controller.prediction_horizon}小时")
        
        # 模块状态
        print(f"\n模块状态:")
        print(f"  异常检测: {'启用' if self.config.controller.enable_anomaly_detection else '禁用'}")
        print(f"  知识库: {'启用' if self.config.controller.enable_knowledge_base else '禁用'}")
        print(f"  预测模型: {'启用' if self.config.controller.enable_prediction else '禁用'}")
        
        # HVAC配置
        print(f"\nHVAC配置:")
        print(f"  控制模式: {self.config.hvac.control_mode}")
        print(f"  目标温度: {self.config.hvac.target_temperature}°C")
        print(f"  温度范围: {self.config.hvac.min_setpoint}°C - {self.config.hvac.max_setpoint}°C")
        print(f"  舒适容差: ±{self.config.hvac.comfort_tolerance}°C")
        
        print("\n" + "=" * 60)
    
    def _watch_status(self, interval: int) -> None:
        """持续监控状态"""
        print("\n进入监控模式，按 Ctrl+C 退出\n")
        
        try:
            while True:
                # 清屏
                os.system('cls' if os.name == 'nt' else 'clear')
                
                self._print_status()
                
                print(f"\n最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"刷新间隔: {interval}秒")
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n监控已停止")
    
    def _cmd_restart(self, args: argparse.Namespace) -> int:
        """重启命令"""
        print("重启系统...")
        
        # 停止
        stop_args = argparse.Namespace(timeout=30)
        self._cmd_stop(stop_args)
        
        # 启动
        start_args = argparse.Namespace(
            config=args.config,
            duration=None,
            daemon=False
        )
        return self._cmd_start(start_args)
    
    def _cmd_query(self, args: argparse.Namespace) -> int:
        """查询命令"""
        config_path = args.config
        
        try:
            self.controller = create_controller(config_path)
            self.controller.initialize()
            
            print(f"查询: {args.question}\n")
            
            result = self.controller.query_knowledge_base(args.question)
            
            if result:
                print("答案:")
                print("-" * 60)
                print(result.answer)
                print("-" * 60)
                print(f"\n置信度: {result.confidence:.2f}")
                
                if result.sources:
                    print(f"\n参考来源:")
                    for i, source in enumerate(result.sources[:args.top_k], 1):
                        print(f"  {i}. {source.get('title', 'Unknown')} "
                              f"(相似度: {source.get('similarity', 0):.2f})")
            else:
                print("未找到相关答案")
            
            return 0
        
        except Exception as e:
            print(f"查询失败: {e}")
            return 1
        
        finally:
            if self.controller:
                self.controller.shutdown()
    
    def _cmd_alerts(self, args: argparse.Namespace) -> int:
        """告警命令"""
        config_path = args.config
        
        try:
            self.controller = create_controller(config_path)
            self.controller.initialize()
            
            alerts = self.controller.get_alerts(severity=args.severity)
            
            print(f"\n告警列表 ({len(alerts)} 条)\n")
            print("=" * 80)
            
            if not alerts:
                print("暂无告警")
            else:
                for i, alert in enumerate(alerts[-args.limit:], 1):
                    print(f"\n[{i}] 严重程度: {alert.severity.upper()}")
                    print(f"    时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"    类型: {alert.anomaly_type}")
                    print(f"    分数: {alert.anomaly_score:.2f}")
                    print(f"    描述: {alert.description}")
                    if alert.affected_metrics:
                        print(f"    受影响指标: {', '.join(alert.affected_metrics)}")
            
            print("\n" + "=" * 80)
            return 0
        
        except Exception as e:
            print(f"获取告警失败: {e}")
            return 1
        
        finally:
            if self.controller:
                self.controller.shutdown()
    
    def _cmd_config(self, args: argparse.Namespace) -> int:
        """配置命令"""
        config_path = args.config
        
        try:
            self.config = ConfigManager(config_path)
            
            if args.config_action == 'get':
                self.config.load()
                value = self.config.get(args.key)
                print(f"{args.key} = {value}")
            
            elif args.config_action == 'set':
                self.config.load()
                self.config.set(args.key, args.value)
                self.config.save()
                print(f"已设置: {args.key} = {args.value}")
            
            elif args.config_action == 'list':
                self.config.load()
                config_dict = self.config.to_dict()
                print(json.dumps(config_dict, indent=2, ensure_ascii=False))
            
            elif args.config_action == 'validate':
                self.config.load()
                errors = self.config.validate()
                if errors:
                    print("配置验证失败:")
                    for error in errors:
                        print(f"  - {error}")
                    return 1
                else:
                    print("配置验证通过")
            
            return 0
        
        except Exception as e:
            print(f"配置操作失败: {e}")
            return 1
    
    def _cmd_simulate(self, args: argparse.Namespace) -> int:
        """仿真命令"""
        config_path = args.config
        
        print(f"运行建筑能耗仿真...")
        print(f"仿真天数: {args.days}")
        
        try:
            self.config = get_config(config_path)
            self.config.load()
            
            # 设置仿真天数
            self.config.simulation.simulation_days = args.days
            
            # 这里可以实现具体的仿真逻辑
            print("仿真功能需要BuildingSimulator支持")
            
            return 0
        
        except Exception as e:
            print(f"仿真失败: {e}")
            return 1
    
    def _cmd_dashboard(self, args: argparse.Namespace) -> int:
        """仪表盘命令"""
        print(f"启动Web仪表盘...")
        print(f"地址: http://{args.host}:{args.port}")
        
        try:
            # 尝试导入并启动Web服务器
            web_dir = project_root / 'web'
            if (web_dir / 'dashboard_server.py').exists():
                import subprocess
                subprocess.run([
                    sys.executable,
                    str(web_dir / 'dashboard_server.py'),
                    '--host', args.host,
                    '--port', str(args.port),
                    '--config', args.config
                ])
            else:
                print("Web仪表盘模块未找到")
                return 1
            
            return 0
        
        except Exception as e:
            print(f"启动仪表盘失败: {e}")
            return 1
    
    def _cmd_version(self, args: argparse.Namespace) -> int:
        """版本命令"""
        print("建筑智能节能系统 (Building Energy Management System)")
        print("版本: 0.1.0")
        print("\n模块状态:")
        
        # 检查各模块可用性
        try:
            from building_energy.models.anomaly_detector import AnomalyDetector
            print("  ✓ 异常检测模块")
        except ImportError:
            print("  ✗ 异常检测模块 (未安装)")
        
        try:
            from building_energy.knowledge.graph_rag import KnowledgeBase
            print("  ✓ 知识库模块")
        except ImportError:
            print("  ✗ 知识库模块 (未安装)")
        
        try:
            from building_energy.models.predictor import EnergyPredictor
            print("  ✓ 预测模型模块")
        except ImportError:
            print("  ✗ 预测模型模块 (未安装)")
        
        return 0


def main():
    """主入口函数"""
    cli = CLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
