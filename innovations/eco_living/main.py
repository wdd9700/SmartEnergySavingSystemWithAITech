"""
生活节能减排助手 - 主入口

运行方式:
    python -m innovations.eco_living.main
    
或:
    from innovations.eco_living import quick_ask
    print(quick_ask("如何节约用电？"))
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from . import init_knowledge_base, quick_ask
from .web_server import run_server


def interactive_mode():
    """交互式模式"""
    print("🌱 绿色助手 - 节能减排智能顾问")
    print("=" * 50)
    print("输入您的问题，或输入 'quit' 退出\n")
    
    # 初始化知识库
    init_knowledge_base()
    
    while True:
        try:
            question = input("\n💬 您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n感谢您的使用，再见！👋")
                break
            
            if not question:
                continue
            
            print("\n🤖 思考中...")
            answer = quick_ask(question)
            print(f"\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\n再见！👋")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")


def web_mode(host: str = "0.0.0.0", port: int = 5001, debug: bool = False):
    """Web服务模式"""
    print(f"🌐 启动Web服务器: http://{host}:{port}")
    
    # 初始化知识库
    init_knowledge_base()
    
    # 启动服务器
    run_server(host=host, port=port, debug=debug)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生活节能减排智能助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                    # 交互式模式
  %(prog)s --web              # Web服务模式
  %(prog)s --web --port 8080  # 指定端口
  %(prog)s --ask "如何节约用水？"  # 单次提问
        """
    )
    
    parser.add_argument(
        "--web",
        action="store_true",
        help="启动Web服务器模式"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web服务器主机地址 (默认: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Web服务器端口 (默认: 5001)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="单次提问模式"
    )
    
    args = parser.parse_args()
    
    if args.ask:
        # 单次提问模式
        init_knowledge_base()
        answer = quick_ask(args.ask)
        print(answer)
    
    elif args.web:
        # Web服务模式
        web_mode(host=args.host, port=args.port, debug=args.debug)
    
    else:
        # 交互式模式
        interactive_mode()


if __name__ == "__main__":
    main()
