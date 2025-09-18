"""
文件管理模組
負責處理配置文件管理、臨時文件清理和文件操作
"""

import os
import time
import glob
import json
import uuid
import threading
import atexit


class FileManager:
    def __init__(self):
        # 啟動背景清理線程
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        # 程序退出時清理
        atexit.register(self.cleanup_old_temp_configs)
    
    def cleanup_old_temp_configs(self):
        """定期清理舊的臨時配置文件"""
        for file in glob.glob("temp_config_*.json") + glob.glob("temp_treatment_config_*.json"):
            try:
                # 如果文件超過1小時，則刪除
                if time.time() - os.path.getctime(file) > 3600:
                    os.remove(file)
                    print(f"🗑️ 清理過期配置: {file}")
            except Exception as e:
                print(f"⚠️ 清理過期配置失敗: {e}")
    
    def cleanup_temp_config_if_needed(self, config_file: str):
        """清理臨時配置文件（如果是動態生成的）"""
        if (config_file.startswith('temp_config_') or config_file.startswith('temp_treatment_config_')) and os.path.exists(config_file):
            try:
                os.remove(config_file)
                print(f"🗑️ 清理臨時配置: {config_file}")
            except Exception as e:
                print(f"⚠️ 清理配置文件失敗: {e}")
    
    def get_default_config_for_template(self, template: str) -> str:
        """根據模板選擇對應的配置文件（向下相容）"""
        # 處遇計畫的配置文件映射
        if template == 'treatment_plan' or template == '處遇計畫':
            return 'treatment_plan.json' if os.path.exists('treatment_plan.json') else 'run.json'
        
        # 報告配置文件映射
        config_mapping = {
            '家事調解中心': 'run_court_format_claude.json',
            '社會福利機構': 'run_association_format_claude.json',
            '通用社工評估報告': 'run.json'
        }
        
        config_file = config_mapping.get(template, 'run.json')
        
        # 如果指定配置文件不存在，回退到預設
        if not os.path.exists(config_file):
            return 'run.json'
        
        return config_file
    
    def get_config_file_for_request(self, template: str, selected_sections: list = None, custom_settings: dict = None) -> str:
        """根據請求參數選擇或生成配置文件（向下相容）"""
        
        # 向下相容：如果沒有新參數，使用預設配置
        if selected_sections is None and not custom_settings:
            return self.get_default_config_for_template(template)
        
        # 新功能：根據選擇的段落動態生成配置
        base_config = self.get_default_config_for_template(template)
        
        try:
            with open(base_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 如果有自定義段落選擇，修改配置
            if selected_sections and len(selected_sections) > 0:
                # 載入段落映射
                if os.path.exists('section_mappings.json'):
                    with open('section_mappings.json', 'r', encoding='utf-8') as f:
                        section_mappings = json.load(f)
                    
                    # 根據選擇的段落重建template
                    if 'steps' in config and len(config['steps']) > 0:
                        original_template = config['steps'][0]['template']
                        
                        # 構建新的段落要求
                        section_requirements = []
                        for section_key in selected_sections:
                            if section_key in section_mappings:
                                section_info = section_mappings[section_key]
                                section_requirements.append(f"• {section_info['title']}: {section_info['description']}")
                        
                        if section_requirements:
                            sections_text = "\n".join(section_requirements)
                            config['steps'][0]['template'] = f"{original_template}\n\n特別要求包含以下段落：\n{sections_text}"
            
            # 如果有自定義設定，調整參數
            if custom_settings:
                if custom_settings.get('detailed_analysis'):
                    config['temperature'] = max(0.1, config.get('temperature', 0.3) - 0.1)
                
                if custom_settings.get('focus_keywords'):
                    keywords = custom_settings['focus_keywords']
                    if 'steps' in config and len(config['steps']) > 0:
                        config['steps'][0]['template'] += f"\n\n請特別關注以下關鍵詞：{keywords}"
            
            # 生成臨時配置文件
            temp_config_path = f"temp_config_{uuid.uuid4().hex[:8]}.json"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            return temp_config_path
            
        except Exception as e:
            print(f"⚠️ 生成配置文件失敗: {e}")
            import traceback
            traceback.print_exc()
            return base_config
    
    def get_temperature_for_style(self, style: str) -> float:
        """根據風格設定返回對應的溫度值"""
        temperature_mapping = {
            'formal': 0.2,      # 正式風格 - 較低溫度，更穩定輸出
            'detailed': 0.4,    # 詳細風格 - 中等溫度，平衡創造性和穩定性
            'concise': 0.1,     # 簡潔風格 - 最低溫度，最穩定輸出
        }
        return temperature_mapping.get(style, 0.3)  # 預設溫度
    
    def _cleanup_worker(self):
        """背景清理工作"""
        while True:
            try:
                self.cleanup_old_temp_configs()
                time.sleep(300)  # 每5分鐘清理一次
            except Exception as e:
                print(f"⚠️ 清理工作出錯: {e}")
                time.sleep(60)


# 全局文件管理器實例
file_manager = FileManager()