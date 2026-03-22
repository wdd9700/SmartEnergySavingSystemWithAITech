"""
生活节能减排Web服务器

提供LLM对话前端界面和API服务。
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from .llm_service import get_llm_service, get_advisor
from .knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """创建Flask应用"""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static")
    )
    CORS(app)
    
    # 确保模板和静态目录存在
    (Path(__file__).parent / "templates").mkdir(exist_ok=True)
    (Path(__file__).parent / "static").mkdir(exist_ok=True)
    (Path(__file__).parent / "static" / "css").mkdir(exist_ok=True)
    (Path(__file__).parent / "static" / "js").mkdir(exist_ok=True)
    
    return app


app = create_app()


# ==================== 页面路由 ====================

@app.route("/")
def index():
    """主页"""
    return render_template("chat.html")


@app.route("/chat")
def chat_page():
    """聊天页面"""
    return render_template("chat.html")


@app.route("/tips")
def tips_page():
    """节能建议页面"""
    return render_template("tips.html")


@app.route("/stats")
def stats_page():
    """统计页面"""
    return render_template("stats.html")


# ==================== API路由 ====================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    聊天API
    
    Request:
        {
            "message": "用户消息",
            "session_id": "会话ID（可选）"
        }
    
    Response:
        {
            "session_id": "会话ID",
            "response": "回复内容",
            "sources": [来源列表],
            "confidence": 置信度
        }
    """
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id")
        
        if not message:
            return jsonify({"error": "消息不能为空"}), 400
        
        llm = get_llm_service()
        result = llm.chat(message, session_id=session_id)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"聊天API错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """
    流式聊天API
    
    使用SSE（Server-Sent Events）流式返回回复
    """
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id")
        
        if not message:
            return jsonify({"error": "消息不能为空"}), 400
        
        def generate():
            llm = get_llm_service()
            for chunk in llm.chat(message, session_id=session_id, stream=True):
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        logger.error(f"流式聊天API错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tips/daily")
def api_daily_tip():
    """获取每日建议"""
    try:
        advisor = get_advisor()
        tip = advisor.get_daily_tip()
        return jsonify(tip)
    except Exception as e:
        logger.error(f"获取每日建议错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tips/category/<category>")
def api_tips_by_category(category: str):
    """按类别获取建议"""
    try:
        advisor = get_advisor()
        tips = advisor.get_tips_by_category(category)
        return jsonify({"category": category, "tips": tips})
    except Exception as e:
        logger.error(f"获取类别建议错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tips/search")
def api_search_tips():
    """搜索建议"""
    try:
        query = request.args.get("q", "").strip()
        top_k = int(request.args.get("k", "5"))
        
        if not query:
            return jsonify({"error": "查询关键词不能为空"}), 400
        
        kb = get_knowledge_base()
        tips = kb.search(query, top_k=top_k)
        
        return jsonify({
            "query": query,
            "results": [
                {
                    "title": tip.title,
                    "content": tip.content,
                    "category": tip.category,
                    "difficulty": tip.difficulty,
                    "impact": tip.impact,
                    "tags": tip.tags
                }
                for tip in tips
            ]
        })
    
    except Exception as e:
        logger.error(f"搜索建议错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/suggestions")
def api_suggested_questions():
    """获取推荐问题"""
    try:
        llm = get_llm_service()
        questions = llm.get_suggested_questions()
        return jsonify({"questions": questions})
    except Exception as e:
        logger.error(f"获取推荐问题错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """获取知识库统计"""
    try:
        kb = get_knowledge_base()
        stats = kb.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"获取统计错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/history")
def api_session_history(session_id: str):
    """获取会话历史"""
    try:
        llm = get_llm_service()
        history = llm.get_session_history(session_id)
        return jsonify({"session_id": session_id, "history": history})
    except Exception as e:
        logger.error(f"获取会话历史错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/clear", methods=["POST"])
def api_clear_session(session_id: str):
    """清空会话"""
    try:
        llm = get_llm_service()
        llm.clear_session(session_id)
        return jsonify({"success": True, "message": "会话已清空"})
    except Exception as e:
        logger.error(f"清空会话错误: {e}")
        return jsonify({"error": str(e)}), 500


# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "未找到"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误"}), 500


def run_server(host: str = "0.0.0.0", port: int = 5001, debug: bool = False):
    """运行服务器"""
    logging.basicConfig(level=logging.INFO)
    logger.info(f"启动生活节能减排Web服务器: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
