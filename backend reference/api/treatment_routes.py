"""
處遇計畫API路由
"""

import subprocess
import sys
import uuid
from flask import Blueprint, request, Response
from utils.session_manager import session_manager
from utils.treatment_plan import treatment_plan_manager
from utils.file_manager import file_manager

treatment_bp = Blueprint('treatment', __name__)

@treatment_bp.route('/api/treatment-plan', methods=['POST'])
def generate_treatment_plan():
    """處遇計畫生成API"""
    data = request.get_json()
    report_content = data.get('reportContent', '')
    main_issue = data.get('mainIssue', '')
    case_type = data.get('caseType', '')
    service_fields = data.get('serviceFields', [])
    custom_settings = data.get('customSettings', {})
    session_id = data.get('sessionId', str(uuid.uuid4()))
    
    def generate():
        print(f"🔄 生成處遇計畫，會話ID: {session_id}")
        print(f"📝 案件類型: {case_type}, 服務領域: {service_fields}")
        print(f"⚙️ 自定義設定: {custom_settings}")
        
        # 使用步驟專用文件路徑，避免與報告生成衝突
        input_file = session_manager.get_step_specific_file_path(session_id, 'treatment', 'input')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(f"報告內容:\n{report_content}\n\n主述議題:\n{main_issue}\n\n案件類型:\n{case_type}")
            if service_fields:
                f.write(f"\n\n社工服務領域:\n{', '.join(service_fields)}")
        
        # 根據參數生成動態配置
        treatment_config = treatment_plan_manager.generate_treatment_plan_config(case_type, service_fields, custom_settings)
        
        # 調用處遇計畫生成腳本
        process = subprocess.Popen([
            sys.executable, 'run.py',
            '--session-id', session_id,
            '--input-file', input_file,
            '--config-file', treatment_config
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in process.stdout:
            yield line
            
        process.stdout.close()
        process.wait()
        
        # 清理臨時配置文件
        file_manager.cleanup_temp_config_if_needed(treatment_config)
    
    return Response(generate(), mimetype='application/x-ndjson')