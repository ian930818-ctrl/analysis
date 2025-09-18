"""
人物關係圖API路由
"""

import subprocess
import sys
import uuid
from flask import Blueprint, request, Response
from utils.session_manager import session_manager

graph_bp = Blueprint('graph', __name__)

@graph_bp.route('/api/PersonGraph', methods=['POST'])
def run_person_graph():
    """人物關係圖生成API"""
    data = request.get_json()
    text = data.get('text', '')
    session_id = data.get('sessionId', str(uuid.uuid4()))
    
    def generate():
        print(f"收到人物關係圖請求，會話ID: {session_id}")
        
        # 只支援人物關係圖
        script_name = 'person_graph.py'
        config_file = 'person_graph.json'
        file_prefix = 'person_graph'
        
        # 使用步驟專用文件路徑
        input_file = session_manager.get_step_specific_file_path(session_id, file_prefix, 'input')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        process = subprocess.Popen([
            sys.executable, script_name,
            '--session-id', session_id,
            '--input-file', input_file,
            '--config-file', config_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in process.stdout:
            yield line
        process.stdout.close()
        process.wait()
    
    return Response(generate(), mimetype='application/x-ndjson')

@graph_bp.route('/api/PersonGraphChat', methods=['POST'])
def person_graph_chat():
    """人物關係圖對話API"""
    data = request.get_json()
    message = data.get('message', '')
    current_graph = data.get('currentGraph', '')
    transcript = data.get('transcript', '')
    session_id = data.get('sessionId', str(uuid.uuid4()))
    
    def generate():
        print(f"收到人物關係圖對話請求，會話ID: {session_id}")
        print(f"用戶消息: {message}")
        
        # 只支援人物關係圖
        file_prefix = "person_graph_chat"
        input_file = session_manager.get_step_specific_file_path(session_id, file_prefix, 'input')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(f"原始逐字稿:\n{transcript}\n\n當前人物關係圖JSON:\n{current_graph}\n\n用戶指令:\n{message}")
        
        # 只使用人物關係圖腳本和配置
        script_name = 'person_graph_chat.py'
        config_file = 'person_graph_chat.json'
        
        process = subprocess.Popen([
            sys.executable, script_name,
            '--session-id', session_id,
            '--input-file', input_file,
            '--message', message,
            '--current-graph', current_graph or '{}',
            '--transcript', transcript,
            '--graph-type', 'person',
            '--config-file', config_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in process.stdout:
            yield line
        process.stdout.close()
        process.wait()
    
    return Response(generate(), mimetype='application/x-ndjson')

@graph_bp.route('/cleanup/<session_id>', methods=['DELETE'])
def cleanup_session(session_id: str):
    """清理指定會話的文件"""
    try:
        session_manager.cleanup_session_files(session_id)
        return {'status': 'success', 'message': f'會話 {session_id} 的文件已清理'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500