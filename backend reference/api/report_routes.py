"""
報告生成API路由
"""

import subprocess
import sys
import uuid
from flask import Blueprint, request, Response
from utils.session_manager import session_manager
from utils.file_manager import file_manager

report_bp = Blueprint('report', __name__)

@report_bp.route('/api/run', methods=['POST'])
def run_script():
    """報告生成API"""
    data = request.get_json()
    text = data.get('text', '')
    session_id = data.get('sessionId', str(uuid.uuid4()))
    
    # 處理報告配置參數
    template = data.get('template', '通用社工評估報告')
    selected_sections = data.get('selectedSections', [])
    custom_settings = data.get('customSettings', {})
    
    def generate():
        # 使用步驟專用文件路徑，避免衝突
        input_file = session_manager.get_step_specific_file_path(session_id, 'report', 'input')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # 根據新參數選擇或生成配置文件
        config_file = file_manager.get_config_file_for_request(template, selected_sections, custom_settings)
        
        # 修改 run.py 的調用，傳入會話ID和輸入文件路徑
        process = subprocess.Popen([
            sys.executable, 'run.py', 
            '--session-id', session_id,
            '--input-file', input_file,
            '--config-file', config_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in process.stdout:
            yield line
        process.stdout.close()
        process.wait()
        
        # 清理臨時配置文件（如果是動態生成的）
        file_manager.cleanup_temp_config_if_needed(config_file)
    
    return Response(generate(), mimetype='application/x-ndjson')