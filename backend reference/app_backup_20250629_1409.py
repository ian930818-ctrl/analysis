from flask import Flask, Response, request
import subprocess
import sys
import os
import tempfile
import uuid
import json
import time
import glob
import threading
import atexit
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 創建臨時文件目錄來存放用戶會話文件
TEMP_DIR = os.path.join(os.getcwd(), 'temp_sessions')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# 🆕 並發控制
session_locks = {}  # 會話級別的鎖
locks_lock = threading.Lock()  # 保護 session_locks 字典的鎖

def get_session_lock(session_id: str) -> threading.Lock:
    """獲取會話專用的鎖"""
    with locks_lock:
        if session_id not in session_locks:
            session_locks[session_id] = threading.Lock()
        return session_locks[session_id]

# 🆕 清理函數定義
def cleanup_old_temp_configs():
    """定期清理舊的臨時配置文件"""
    for file in glob.glob("temp_config_*.json") + glob.glob("temp_treatment_config_*.json"):
        try:
            # 如果文件超過1小時，則刪除
            if time.time() - os.path.getctime(file) > 3600:
                os.remove(file)
                print(f"🗑️ 清理過期配置: {file}")
        except Exception as e:
            print(f"⚠️ 清理過期配置失敗: {e}")

def cleanup_old_session_files():
    """🆕 清理老舊會話文件"""
    try:
        current_time = time.time()
        
        for session_id in os.listdir(TEMP_DIR):
            session_dir = os.path.join(TEMP_DIR, session_id)
            if not os.path.isdir(session_dir):
                continue
                
            # 檢查會話目錄的最後修改時間
            if current_time - os.path.getmtime(session_dir) > 86400:  # 24小時
                try:
                    import shutil
                    shutil.rmtree(session_dir)
                    print(f"🗑️ 清理過期會話: {session_id}", file=sys.stderr)
                    
                    # 清理對應的會話鎖
                    with locks_lock:
                        if session_id in session_locks:
                            del session_locks[session_id]
                            
                except Exception as e:
                    print(f"⚠️ 清理會話失敗 {session_id}: {e}", file=sys.stderr)
                    
            else:
                # 清理會話內的老舊步驟文件
                cleanup_old_step_files(session_dir)
                
    except Exception as e:
        print(f"⚠️ 會話清理工作失敗: {e}", file=sys.stderr)

def cleanup_old_step_files(session_dir: str):
    """🆕 清理會話內老舊的步驟文件"""
    try:
        current_time = time.time()
        
        for filename in os.listdir(session_dir):
            file_path = os.path.join(session_dir, filename)
            
            # 只清理步驟文件（包含時間戳的文件）
            if '_' in filename and filename.endswith('.txt'):
                try:
                    # 如果文件超過2小時且不是最近的，則刪除
                    if current_time - os.path.getmtime(file_path) > 7200:
                        os.remove(file_path)
                        print(f"🗑️ 清理步驟文件: {file_path}", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️ 清理步驟文件失敗 {file_path}: {e}", file=sys.stderr)
                    
    except Exception as e:
        print(f"⚠️ 步驟文件清理失敗: {e}", file=sys.stderr)

# 🆕 定期清理任務
def cleanup_worker():
    """背景清理工作"""
    while True:
        try:
            cleanup_old_temp_configs()
            cleanup_old_session_files()
            time.sleep(300)  # 每5分鐘清理一次
        except Exception as e:
            print(f"⚠️ 清理工作出錯: {e}", file=sys.stderr)
            time.sleep(60)

# 啟動背景清理線程
cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
cleanup_thread.start()

# 程序退出時清理
atexit.register(lambda: cleanup_old_temp_configs())

def get_session_file_path(session_id: str, filename: str) -> str:
    """根據會話ID獲取文件路徑（保持向下相容）"""
    session_dir = os.path.join(TEMP_DIR, session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    return os.path.join(session_dir, filename)

def get_step_specific_file_path(session_id: str, step: str, file_type: str = 'input') -> str:
    """🆕 根據步驟和時間戳獲取唯一文件路徑，避免衝突"""
    timestamp = str(int(time.time() * 1000))  # 使用毫秒時間戳
    filename = f"{step}_{file_type}_{timestamp}.txt"
    
    session_dir = os.path.join(TEMP_DIR, session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    file_path = os.path.join(session_dir, filename)
    print(f"📁 創建步驟文件: {file_path}", file=sys.stderr)
    return file_path

def get_concurrent_safe_file_path(session_id: str, operation: str) -> str:
    """🆕 獲取並發安全的文件路徑"""
    import threading
    thread_id = threading.get_ident()
    timestamp = str(int(time.time() * 1000))
    filename = f"{operation}_{thread_id}_{timestamp}.txt"
    
    session_dir = os.path.join(TEMP_DIR, session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    return os.path.join(session_dir, filename)

def cleanup_session_files(session_id: str):
    """清理會話文件"""
    session_dir = os.path.join(TEMP_DIR, session_id)
    if os.path.exists(session_dir):
        import shutil
        shutil.rmtree(session_dir)

def get_config_file_for_request(template: str, selected_sections: list = None, custom_settings: dict = None) -> str:
    """根據請求參數選擇或生成配置文件（向下相容）"""
    
    # 🛡️ 向下相容：如果沒有新參數，使用預設配置
    if selected_sections is None and not custom_settings:
        return get_default_config_for_template(template)
    
    # 🆕 新功能：根據選擇的段落動態生成配置
    base_config = get_default_config_for_template(template)
    
    if selected_sections:
        return generate_custom_config(base_config, selected_sections, custom_settings)
    
    return base_config

def get_default_config_for_template(template: str) -> str:
    """根據模板名稱獲取預設配置文件"""
    
    # 配置文件映射（向下相容現有模板）
    config_map = {
        '司法社工家庭訪視模板': 'run.json',
        '士林地院家事服務中心格式(ChatGPT)': 'run.json', 
        '士林地院家事服務中心格式(Claude)': 'run.json',
        '珍珠社會福利協會格式(ChatGPT)': 'run.json',
        '珍珠社會福利協會格式(Claude)': 'run.json',
        # 🆕 新增通用模板
        'universal_social_work_claude': 'run.json',
        '通用社工評估報告': 'run.json'
    }
    
    config_file = config_map.get(template, 'run.json')
    
    # 確保配置文件存在
    if not os.path.exists(config_file):
        print(f"⚠️ 配置文件不存在: {config_file}，使用預設 run.json")
        return 'run.json'
    
    return config_file

def generate_custom_config(base_config_file: str, selected_sections: list, custom_settings: dict = None) -> str:
    """根據選擇的段落動態生成配置文件"""
    
    try:
        # 讀取基礎配置
        with open(base_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 🆕 實作段落選擇邏輯
        if selected_sections:
            print(f"📝 選擇的段落: {selected_sections}")
            custom_prompt = build_custom_prompt(selected_sections, custom_settings)
            
            # 修改配置中的 template
            if 'steps' in config and len(config['steps']) > 0:
                config['steps'][0]['template'] = custom_prompt
                print(f"🔧 已生成客製化Prompt，長度: {len(custom_prompt)} 字元")
        
        if custom_settings:
            print(f"⚙️ 自定義設定: {custom_settings}")
            # 根據風格偏好調整溫度參數
            if 'style' in custom_settings:
                config['temperature'] = get_temperature_for_style(custom_settings['style'])
        
        # 生成臨時配置文件
        temp_config_path = f"temp_config_{uuid.uuid4().hex[:8]}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return temp_config_path
        
    except Exception as e:
        print(f"⚠️ 生成自定義配置失敗: {e}")
        return base_config_file

def get_section_definitions():
    """獲取段落定義 - 按照社工訪視紀錄標準架構"""
    return {
        # === 必須顯示項目 ===
        'main_issue': {
            'name': '一、主述議題',
            'description': '求助者的身分、求助方式、求助問題',
            'order': 1,
            'required': True
        },
        
        # === 二、個案概況 (必須顯示) ===
        'case_family_situation': {
            'name': '二、個案概況 - (一)家庭狀況',
            'description': '家庭人員的組成（結構、年齡）、家人的相處模式（關係）、教育程度、婚姻關係、家庭的權力結構、經濟狀況、就業情形、財產分配或收入、重大事件、居住環境',
            'order': 2,
            'required': True
        },
        'case_children_situation': {
            'name': '二、個案概況 - (二)子女狀況',
            'description': '子女生活或教育上的問題、教養的問題、親子關係、過往照顧的狀況、是否有特殊疾病或狀況等',
            'order': 3,
            'required': True
        },
        'case_relationship_chart': {
            'name': '二、個案概況 - (三)人物關係圖',
            'description': '家庭成員及重要他人的關係網絡圖，將另開分頁顯示AI生成結果',
            'order': 4,
            'required': True
        },

        # === 三、個案狀況 (可複選) ===
        'legal_status': {
            'name': '三、個案狀況 - (一)法律相關狀況',
            'description': '是否有訴訟(如民事離婚、保護令、暫時處份、強制執行、刑事案件-家暴、妨害性自主、法律爭議、法院未成年子女相關訴訟(如酌定親權-監護權、會面交往、給付扶養)、是否有犯罪服刑、涉及家庭暴力...等等',
            'order': 5
        },
        'economic_financial_status': {
            'name': '三、個案狀況 - (二)經濟或財務狀況',
            'description': '主要收入來源、主要經濟提供者、是否有人身保險、是否負債、個案謀生能力、主要花費負擔',
            'order': 6
        },
        'safety_security_status': {
            'name': '三、個案狀況 - (三)人身或安全狀況',
            'description': '是否具有攻擊風險、訪視時應注意事項、是否有家暴或受虐可能、是否有家人間的性騷擾或性侵害、是否擔心受害、是否有人身安全問題、是否需要搬離住所或聯繫當地警局協助等',
            'order': 7,
            'required': True  # 安全狀況應為必選
        },
        'psychological_emotional_status': {
            'name': '三、個案狀況 - (四)心理或情緒狀況',
            'description': '個案或其家人的人格特質、情緒穩定度、訪視的態度、身心狀況、是否有諮商或看精神科（或疾病史）、是否有自我傷害傾向、重大壓力事件',
            'order': 8
        },
        'parenting_education_status': {
            'name': '三、個案狀況 - (五)教養或教育狀況',
            'description': '個案或其家庭的親職能力、親職教養上的困難、孩子接受課後照顧或補習情形、孩子學業成績表現、學校中的師生關係、孩子與同儕的關係或互動、學業壓力',
            'order': 9
        },
        'early_intervention_childcare_status': {
            'name': '三、個案狀況 - (六)早療或幼兒狀況',
            'description': '個案與配偶之間的互動頻率、彼此情感支持狀況、家務責任分工、與孩子互動的頻率與深度、是否有隔代教養的問題、孩子與祖父母的情感關係、教養因應問題的策略或技巧',
            'order': 10
        },
        'medical_physical_status': {
            'name': '三、個案狀況 - (七)醫療或生理狀況',
            'description': '個案或其家人的罹病與診治史、對疾病的認識與態度、是否有長期用藥、是否具有身心障礙資格或有重大傷病卡、是否有慢性疾病或有重大疾病，服藥穩定度、對醫療的期待、醫療團隊的評估',
            'order': 11
        },
        'support_system_status': {
            'name': '三、個案狀況 - (八)支持系統或狀況',
            'description': '支持系統(正式系統、非正式系統)、主要照顧者、是否有委任律師、資源使用的能力、經常請教討論的對象、這些支持系統或支持者所提供的訊息或協助',
            'order': 12
        },
        'cultural_traditional_status': {
            'name': '三、個案狀況 - (九)文化與傳統狀況',
            'description': '國籍(若非台灣國籍)、民族(若非漢族)、宗教信仰背景、與台灣主流文化不同的生活習慣、生活價值觀、生活適應問題、語言溝通問題、與遠地或國外家人的關係',
            'order': 13
        },

        # === 四、需求與評估 (AI生成) ===
        'case_needs_expectations': {
            'name': '四、需求與評估 - (一)個案需求與期待',
            'description': '個案對目前狀況的想法或規劃、所表達的需求、根據過往經驗而希望改進的地方、陪同或喘息的需求、希望能解決的問題、期待從政府或相關單位得到的資源或協助',
            'order': 14,
            'required': True
        },
        'family_function_assessment': {
            'name': '四、需求與評估 - (二)家庭功能評估',
            'description': '從專業且資深的社工角度，評估個案家庭功能的優勢處、劣勢處、目前的危機、未來可改變的機會',
            'order': 15,
            'required': True
        },
        'overall_assessment_recommendations': {
            'name': '四、需求與評估 - (三)整體評估建議',
            'description': '從專業且資深的社工角度，評估個案目前的能動性、主要需要解決的問題、可能需要立即協助的需求、需要長期陪伴的需求、需要搭配其他單位資源的需求等等。儘量從不同面向完整提供，避免遺漏。',
            'order': 16,
            'required': True
        }
    }

def sort_sections_by_order(selected_sections: list, section_definitions: dict) -> list:
    """按order字段排序段落"""
    section_with_order = []
    for section_id in selected_sections:
        if section_id in section_definitions:
            order = section_definitions[section_id].get('order', 999)
            section_with_order.append((order, section_id))
    
    # 按order排序
    section_with_order.sort(key=lambda x: x[0])
    return [section_id for order, section_id in section_with_order]

def get_section_writing_guide(section_id: str, section_def: dict) -> str:
    """為每個段落生成具體的撰寫指引 - 按照社工訪視紀錄標準架構"""
    guides = {
        # === 必須顯示項目 ===
        'main_issue': '以自然段落方式描述求助者的身分背景、採用的求助方式（如電話、親自到訪等）、以及具體的求助問題，整合呈現為完整的主述議題描述。',
        
        # === 二、個案概況 (必須顯示) ===
        'case_family_situation': '用連貫的文字段落詳述家庭人員的組成結構（包含年齡分布）、家人間的相處模式與關係品質、各成員的教育程度、婚姻關係狀態、家庭內的權力結構與決策模式、經濟狀況與就業情形、財產分配或收入狀況、曾發生的重大事件，以及目前的居住環境狀況。',
        'case_children_situation': '以敘述性文字描述子女在生活或教育方面遭遇的問題、親職教養上的困難與挑戰、親子關係的互動品質、過往的照顧狀況與模式、以及是否存在特殊疾病或發展狀況等，綜合呈現子女的整體狀況。',
        'case_relationship_chart': '用文字描述家庭成員及重要他人之間的關係網絡，說明各成員間的親密程度、互動頻率、支持關係等。此段落內容將作為人物關係圖生成的基礎，另會在人物關係圖分頁中以視覺化方式呈現。',

        # === 三、個案狀況 (可複選) ===
        'legal_status': '以完整段落詳述個案是否涉及各類訴訟程序（包含民事離婚、保護令申請、暫時處分、強制執行等）、刑事案件（如家庭暴力、妨害性自主等）、法律爭議狀況、法院關於未成年子女的相關訴訟（如親權酌定、監護權、會面交往、扶養費給付等）、是否有犯罪紀錄或服刑經歷、以及任何涉及家庭暴力的法律問題。',
        'economic_financial_status': '用段落形式分析個案的主要收入來源與穩定性、家庭主要經濟提供者、是否具備人身保險保障、負債狀況、個案的謀生能力與就業狀況、主要花費負擔項目，綜合評估經濟財務狀況對家庭生活的影響。',
        'safety_security_status': '以連續文字評估個案是否具有攻擊性風險、進行訪視時需要注意的安全事項、是否存在家庭暴力或受虐的可能性、家庭成員間是否有性騷擾或性侵害情況、個案是否擔心自身安全、是否存在人身安全威脅、以及是否需要搬離現居住所或聯繫當地警局協助等安全相關議題。',
        'psychological_emotional_status': '用完整段落描述個案或其家人的人格特質、情緒穩定程度、在訪視過程中展現的態度、整體身心狀況、是否接受心理諮商或精神科治療（包含疾病史）、是否有自我傷害的傾向、以及面臨的重大壓力事件，呈現心理情緒狀況的全貌。',
        'parenting_education_status': '以敘述方式評估個案或其家庭的親職能力表現、在親職教養方面遭遇的困難、孩子接受課後照顧或補習的情形、孩子的學業成績表現、在學校中與師長的關係、孩子與同儕間的關係與互動狀況、以及面臨的學業壓力等教養教育相關議題。',
        'early_intervention_childcare_status': '用段落形式描述個案與配偶之間的互動頻率與品質、彼此提供的情感支持狀況、家務責任的分工安排、與孩子互動的頻率與深度、是否存在隔代教養的問題、孩子與祖父母的情感關係、以及在面對教養問題時所採用的因應策略或技巧。',
        'medical_physical_status': '以完整文字記錄個案或其家人的疾病罹患與診治史、對疾病的認識程度與態度、是否需要長期用藥、是否具有身心障礙資格或持有重大傷病卡、是否患有慢性疾病或重大疾病、服藥的穩定程度、對醫療照護的期待、以及醫療團隊的專業評估意見。',
        'support_system_status': '以敘述性文字盤點個案的支持系統（包含正式系統如社會服務機構、非正式系統如親友網絡）、主要照顧者的身分與角色、是否有委任律師提供法律協助、運用資源的能力、經常請教討論的對象、以及這些支持系統或支持者所能提供的具體訊息或協助內容。',
        'cultural_traditional_status': '用連貫段落考慮個案的國籍背景（若非台灣國籍）、民族身分（若非漢族）、宗教信仰背景、與台灣主流文化存在差異的生活習慣、生活價值觀念、在台灣的生活適應問題、語言溝通上的困難、以及與遠地或國外家人的關係維繫狀況。',

        # === 四、需求與評估 (AI生成) ===
        'case_needs_expectations': '以段落方式整理個案對目前狀況的想法或未來規劃、明確表達的需求、根據過往經驗希望改進的具體事項、對陪同服務或喘息服務的需求、希望能夠解決的核心問題、以及期待從政府機關或相關單位獲得的資源或協助內容。',
        'family_function_assessment': '從專業且資深的社工角度，用完整段落評估個案家庭功能的優勢之處、存在的劣勢或困難、目前面臨的危機狀況、以及未來可能改變或改善的機會，提供專業的家庭功能分析。',
        'overall_assessment_recommendations': '從專業且資深的社工角度，以敘述性文字評估個案目前的能動性與主動性、主要需要解決的核心問題、可能需要立即協助的緊急需求、需要長期陪伴支持的項目、需要搭配其他單位資源的合作需求等。請儘量從不同面向提供完整的評估建議，避免遺漏重要層面。'
    }
    return guides.get(section_id, '請根據段落性質提供相關內容，以自然流暢的文字段落呈現，符合社工專業評估標準。')

def build_custom_prompt(selected_sections: list, custom_settings: dict = None) -> str:
    """🆕 根據選擇的段落動態構建Prompt - 與前端議題完全對應"""
    
    try:
        section_definitions = get_section_definitions()
        
        base_prompt = """你是一位專業的社會工作師，請根據以下逐字稿內容，撰寫一份結構化的社工評估報告。

請遵循以下專業標準：
- 使用客觀、專業的第三人稱描述
- 基於逐字稿內容進行分析，避免過度推測
- 保持社工倫理和保密原則
- 確保內容具體、可操作、符合台灣社工實務標準
- 每個段落使用自然流暢的文字敘述，避免條列式或小標題格式
- 將相關資訊整合成連貫的段落，呈現完整的情況描述"""
        
        # 按順序處理選擇的段落
        sorted_sections = sort_sections_by_order(selected_sections, section_definitions)
        
        # 根據選擇的段落生成指示
        section_instructions = []
        for section_id in sorted_sections:
            if section_id in section_definitions:
                section_def = section_definitions[section_id]
                required_mark = " (必要項目)" if section_def.get('required', False) else ""
                
                instruction = f"""**{section_def['name']}{required_mark}**
內容要求：{section_def['description']}
撰寫指引：{get_section_writing_guide(section_id, section_def)}"""
                section_instructions.append(instruction)
        
        # 組合完整的Prompt
        full_prompt = f"""{base_prompt}

報告結構與內容要求：

{chr(10).join(section_instructions)}

整體撰寫要求：
- 請嚴格按照上述段落順序組織報告內容
- 每個段落都要有明確的大標題（如「一、基本資料」、「二、個案概況」等）
- 段落內容以自然流暢的文字段落呈現，不要使用小標題或條列式格式
- 將各項相關資訊整合成連貫的敘述性文字，讓社工能看到完整的情況描述
- 根據逐字稿內容進行專業分析，不要逐字複製
- 以客觀、清晰的社工專業文風撰寫
- 若某段落在逐字稿中資訊不足，請在段落中自然地提及「相關資訊有限，建議後續評估中進一步了解」
- 確保每個段落內容充實且符合專業標準
- 必要項目請務必包含，即使資訊有限也要基於專業判斷提供基本評估"""

        # 添加自定義設定
        if custom_settings:
            if 'notes' in custom_settings and custom_settings['notes'].strip():
                full_prompt += f"\n\n特殊要求：\n{custom_settings['notes']}"
            
            if 'style' in custom_settings:
                style_instruction = get_style_instruction(custom_settings['style'])
                full_prompt += f"\n\n報告風格：{style_instruction}"
        
        full_prompt += "\n\n以下是逐字稿內容：\n{input}"
        
        print(f"📝 已生成動態Prompt，包含 {len(sorted_sections)} 個段落")
        return full_prompt
        
    except Exception as e:
        print(f"⚠️ 構建自定義Prompt失敗: {e}")
        import traceback
        traceback.print_exc()
        # 如果失敗，返回預設的Prompt
        return "請你根據下面的逐字稿內容，幫我整理成結構化的社工評估報告：\n\n{input}"

def get_temperature_for_style(style: str) -> float:
    """🆕 根據風格偏好獲取模型溫度參數"""
    style_temperatures = {
        'formal': 0.2,    # 正式風格，較低創造性
        'detailed': 0.4,  # 詳細分析，中等創造性  
        'concise': 0.3    # 簡潔風格，適中創造性
    }
    return style_temperatures.get(style, 0.3)

def get_style_instruction(style: str) -> str:
    """🆕 根據風格偏好獲取寫作指示"""
    style_instructions = {
        'formal': '請使用正式、客觀、專業的語調，條理清晰、邏輯嚴謹，採用第三人稱撰寫',
        'detailed': '請提供詳細、深入的分析，全面覆蓋各個層面，使用豐富的描述和專業術語',
        'concise': '請保持簡潔、重點突出，精簡有力地表達核心要點，言簡意賅'
    }
    return style_instructions.get(style, '請保持專業、客觀的寫作風格')

def cleanup_temp_config_if_needed(config_file: str):
    """清理臨時配置文件"""
    if (config_file.startswith('temp_config_') or config_file.startswith('temp_treatment_config_')) and os.path.exists(config_file):
        try:
            os.remove(config_file)
            print(f"🗑️ 清理臨時配置: {config_file}")
        except Exception as e:
            print(f"⚠️ 清理配置文件失敗: {e}")

@app.route('/api/run', methods=['POST'])
def run_script():
    data = request.get_json()
    text = data.get('text', '')
    session_id = data.get('sessionId', str(uuid.uuid4()))  # 如果沒有提供 sessionId，生成一個新的
        
        for session_id in os.listdir(TEMP_DIR):
            session_dir = os.path.join(TEMP_DIR, session_id)
            if not os.path.isdir(session_dir):
                continue
                
            # 檢查會話目錄的最後修改時間
            if current_time - os.path.getmtime(session_dir) > 86400:  # 24小時
                try:
                    import shutil
                    shutil.rmtree(session_dir)
                    print(f"🗑️ 清理過期會話: {session_id}", file=sys.stderr)
                    
                    # 清理對應的會話鎖
                    with locks_lock:
                        if session_id in session_locks:
                            del session_locks[session_id]
                            
                except Exception as e:
                    print(f"⚠️ 清理會話失敗 {session_id}: {e}", file=sys.stderr)
                    
            else:
                # 清理會話內的老舊步驟文件
                cleanup_old_step_files(session_dir)
                
    except Exception as e:
        print(f"⚠️ 會話清理工作失敗: {e}", file=sys.stderr)

def cleanup_old_step_files(session_dir: str):
    """🆕 清理會話內老舊的步驟文件"""
    try:
        current_time = time.time()
        
        for filename in os.listdir(session_dir):
            file_path = os.path.join(session_dir, filename)
            
            # 只清理步驟文件（包含時間戳的文件）
            if '_' in filename and filename.endswith('.txt'):
                try:
                    # 如果文件超過2小時且不是最近的，則刪除
                    if current_time - os.path.getmtime(file_path) > 7200:
                        os.remove(file_path)
                        print(f"🗑️ 清理步驟文件: {file_path}", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️ 清理步驟文件失敗 {file_path}: {e}", file=sys.stderr)
                    
    except Exception as e:
        print(f"⚠️ 步驟文件清理失敗: {e}", file=sys.stderr)

@app.route('/api/run', methods=['POST'])
def run_script():
    data = request.get_json()
    text = data.get('text', '')
    session_id = data.get('sessionId', str(uuid.uuid4()))  # 如果沒有提供 sessionId，生成一個新的
    
    # 🆕 新增：處理報告配置參數
    template = data.get('template', '通用社工評估報告')
    selected_sections = data.get('selectedSections', [])
    custom_settings = data.get('customSettings', {})
    
    def generate():
        # 🆕 使用步驟專用文件路徑，避免衝突
        input_file = get_step_specific_file_path(session_id, 'report', 'input')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # 🆕 根據新參數選擇或生成配置文件
        config_file = get_config_file_for_request(template, selected_sections, custom_settings)
        
        # 修改 run.py 的調用，傳入會話ID和輸入文件路徑
        process = subprocess.Popen([
            sys.executable, 'run.py', 
            '--session-id', session_id,
            '--input-file', input_file,
            '--config', config_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in process.stdout:
            yield line
        process.stdout.close()
        process.wait()
        
        # 清理臨時配置文件（如果是動態生成的）
        cleanup_temp_config_if_needed(config_file)
    
    return Response(generate(), mimetype='application/x-ndjson')

@app.route('/api/PersonGraph', methods=['POST'])
def run_person_graph():
    data = request.get_json()
    text = data.get('text', '')
    session_id = data.get('sessionId', str(uuid.uuid4()))
    
    def generate():
        print(f"收到人物關係圖請求，會話ID: {session_id}", file=sys.stderr)
        
        # 🆕 使用步驟專用文件路徑，避免衝突
        input_file = get_step_specific_file_path(session_id, 'person_graph', 'input')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        process = subprocess.Popen([
            sys.executable, 'person_graph.py',
            '--session-id', session_id,
            '--input-file', input_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in process.stdout:
            yield line
        process.stdout.close()
        process.wait()
    
    return Response(generate(), mimetype='application/x-ndjson')

@app.route('/api/PersonGraphChat', methods=['POST'])
def person_graph_chat():
    data = request.get_json()
    message = data.get('message', '')
    current_graph = data.get('currentGraph', '')
    transcript = data.get('transcript', '')  # 新增逐字稿
    graph_type = data.get('graphType', 'person')  # 新增圖表類型，預設為人物關係圖
    session_id = data.get('sessionId', str(uuid.uuid4()))
    
    def generate():
        graph_type_name = '人物關係圖' if graph_type == 'person' else '家庭關係圖'
        print(f"收到{graph_type_name}對話請求，會話ID: {session_id}", file=sys.stderr)
        print(f"用戶消息: {message}", file=sys.stderr)
        
        # 🆕 使用步驟專用文件路徑，根據圖表類型區分
        file_prefix = f"{graph_type}_graph_chat"
        input_file = get_step_specific_file_path(session_id, file_prefix, 'input')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(f"原始逐字稿:\n{transcript}\n\n當前{graph_type_name}JSON:\n{current_graph}\n\n用戶指令:\n{message}")
        
        process = subprocess.Popen([
            sys.executable, 'person_graph_chat.py',
            '--session-id', session_id,
            '--input-file', input_file,
            '--message', message,
            '--current-graph', current_graph or '{}',
            '--transcript', transcript,
            '--graph-type', graph_type  # 新增圖表類型參數
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in process.stdout:
            yield line
        process.stdout.close()
        process.wait()
    
    return Response(generate(), mimetype='application/x-ndjson')

@app.route('/api/treatment-plan', methods=['POST'])
def generate_treatment_plan():
    """新增：處遇計畫生成API"""
    data = request.get_json()
    report_content = data.get('reportContent', '')
    main_issue = data.get('mainIssue', '')
    case_type = data.get('caseType', '')
    service_fields = data.get('serviceFields', [])
    custom_settings = data.get('customSettings', {})
    session_id = data.get('sessionId', str(uuid.uuid4()))
    
    def generate():
        print(f"🔄 生成處遇計畫，會話ID: {session_id}", file=sys.stderr)
        print(f"📝 案件類型: {case_type}, 服務領域: {service_fields}", file=sys.stderr)
        print(f"⚙️ 自定義設定: {custom_settings}", file=sys.stderr)
        
        # 🆕 使用步驟專用文件路徑，避免與報告生成衝突
        input_file = get_step_specific_file_path(session_id, 'treatment', 'input')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(f"報告內容:\n{report_content}\n\n主述議題:\n{main_issue}\n\n案件類型:\n{case_type}")
            if service_fields:
                f.write(f"\n\n社工服務領域:\n{', '.join(service_fields)}")
        
        # 🆕 根據參數生成動態配置
        treatment_config = generate_treatment_plan_config(case_type, service_fields, custom_settings)
        
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
        cleanup_temp_config_if_needed(treatment_config)
    
    return Response(generate(), mimetype='application/x-ndjson')

def generate_treatment_plan_config(case_type: str, service_fields: list, custom_settings: dict) -> str:
    """🆕 根據參數動態生成處遇計畫配置文件"""
    
    try:
        # 讀取基礎處遇計畫配置
        base_config_file = 'treatment_plan.json' if os.path.exists('treatment_plan.json') else 'run.json'
        with open(base_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 🆕 構建動態處遇計畫prompt
        dynamic_prompt = build_treatment_plan_prompt(case_type, service_fields, custom_settings)
        
        # 修改配置中的template
        if 'steps' in config and len(config['steps']) > 0:
            config['steps'][0]['template'] = dynamic_prompt
            print(f"🔧 已生成客製化處遇計畫Prompt，長度: {len(dynamic_prompt)} 字元", file=sys.stderr)
        
        # 根據自定義設定調整參數
        if custom_settings:
            if 'style' in custom_settings:
                config['temperature'] = get_temperature_for_style(custom_settings['style'])
                print(f"🌡️ 根據風格 '{custom_settings['style']}' 調整溫度: {config['temperature']}", file=sys.stderr)
        
        # 生成臨時配置文件
        temp_config_path = f"temp_treatment_config_{uuid.uuid4().hex[:8]}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return temp_config_path
        
    except Exception as e:
        print(f"⚠️ 生成處遇計畫配置失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 'treatment_plan.json' if os.path.exists('treatment_plan.json') else 'run.json'

def build_treatment_plan_prompt(case_type: str, service_fields: list, custom_settings: dict) -> str:
    """🆕 構建動態處遇計畫prompt"""
    
    # 案件類型對應的專業重點
    case_type_focus = {
        'family_mediation': '家事調解和衝突處理',
        'parent_child': '親子關係修復和溝通改善',
        'marriage_counseling': '婚姻關係諮商和伴侶治療',
        'child_protection': '兒童保護和安全評估',
        'domestic_violence': '家庭暴力防治和創傷復原',
        'other_family': '一般家庭問題處理'
    }
    
    # 服務領域對應的專業策略
    service_field_strategies = {
        'judicial_correction': '司法程序配合、法庭評估、矯治資源連結',
        'economic_assistance': '經濟扶助申請、就業輔導、理財規劃',
        'new_residents': '文化適應、語言學習、社區融入',
        'protection_services': '保護服務、風險評估、安全計畫',
        'children_youth': '兒少發展、教育支持、才能培養',
        'school_education': '學校合作、學習支援、特殊教育',
        'women_family': '婦女權益、家庭功能、性別議題',
        'medical_related': '醫療資源、復健服務、照護計畫',
        'psychological_mental': '心理治療、精神醫療、情緒支持',
        'disability': '身心障礙服務、輔具申請、無障礙環境',
        'elderly_longterm_care': '長照服務、老人照護、安養規劃'
    }
    
    base_prompt = """你是一位專業的社會工作師，請根據以下社工報告，生成專業的處遇計畫。

處遇計畫應該包含以下結構：

一、處遇目標
(一)短期目標（1-3個月）
(二)中期目標（3-6個月）
(三)長期目標（6個月以上）

二、處遇策略
(一)個案工作策略
(二)家族治療策略
(三)資源連結策略
(四)環境調整策略

三、實施步驟
(一)評估階段
(二)介入階段
(三)維持階段
(四)結案評估

四、預期成效
(一)個人層面成效
(二)家庭層面成效
(三)社會功能改善
(四)風險降低程度

五、評估指標
(一)量化指標
(二)質化指標
(三)時程安排
(四)檢核方式

六、資源需求
(一)人力資源
(二)經費需求
(三)外部資源
(四)專業協助

撰寫要求：
- 請根據報告中的具體問題和需求制定切實可行的處遇計畫
- 目標設定應該具體、可測量、可達成
- 策略應該具有專業性和可操作性
- 時程安排要合理且具有彈性
- 充分考慮案主的能力、資源和限制
- 體現社會工作的專業價值和倫理"""
    
    # 根據案件類型調整重點
    if case_type and case_type in case_type_focus:
        focus = case_type_focus[case_type]
        base_prompt += f"\n\n特別重點：\n本案件為{focus}案件，請在處遇計畫中特別重視相關的專業介入策略和技巧。"
    
    # 根據服務領域調整策略建議
    if service_fields:
        selected_strategies = []
        for field in service_fields:
            if field in service_field_strategies:
                selected_strategies.append(service_field_strategies[field])
        
        if selected_strategies:
            base_prompt += f"\n\n專業領域策略：\n請在處遇計畫中整合以下專業領域的策略和資源：\n- " + "\n- ".join(selected_strategies)
    
    # 添加自定義設定
    if custom_settings:
        if 'notes' in custom_settings and custom_settings['notes'].strip():
            base_prompt += f"\n\n特殊要求：\n{custom_settings['notes']}"
        
        if 'style' in custom_settings:
            style_instruction = get_treatment_plan_style_instruction(custom_settings['style'])
            base_prompt += f"\n\n撰寫風格：{style_instruction}"
    
    base_prompt += "\n\n請基於以下內容生成處遇計畫：\n\n{input}"
    
    return base_prompt

def get_treatment_plan_style_instruction(style: str) -> str:
    """🆕 處遇計畫風格指令"""
    style_instructions = {
        'formal': '請使用正式、專業的語調，嚴格遵循社工實務標準，條理清晰地呈現每個處遇環節',
        'detailed': '請提供詳細、具體的處遇步驟，包含完整的實施細節、時程規劃和評估標準',
        'concise': '請保持簡潔有力，重點突出核心處遇策略和關鍵步驟，避免冗長描述'
    }
    return style_instructions.get(style, '請保持專業、實用的撰寫風格，確保處遇計畫具有可操作性')

@app.route('/cleanup/<session_id>', methods=['DELETE'])
def cleanup_session(session_id: str):
    """清理指定會話的文件"""
    try:
        cleanup_session_files(session_id)
        return {'status': 'success', 'message': f'會話 {session_id} 的文件已清理'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5353, debug=True) 
