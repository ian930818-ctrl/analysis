"""
重構後的主要Flask應用
採用模組化架構，將功能分離到不同模組中
"""

from flask import Flask
from flask_cors import CORS

# 導入各模組的管理器
from utils.session_manager import session_manager
from utils.file_manager import file_manager

# 導入API路由藍圖
from api.report_routes import report_bp
from api.treatment_routes import treatment_bp
from api.graph_routes import graph_bp

# 創建Flask應用
app = Flask(__name__)
CORS(app)

# 註冊藍圖
app.register_blueprint(report_bp)
app.register_blueprint(treatment_bp)
app.register_blueprint(graph_bp)

# 健康檢查端點
@app.route('/api/health', methods=['GET'])
def health_check():
    """API健康檢查"""
    return {
        'status': 'healthy',
        'message': '社工報告系統運行正常',
        'modules': {
            'session_manager': 'active',
            'file_manager': 'active',
            'treatment_plan': 'active'
        }
    }

# 根路由
@app.route('/', methods=['GET'])
def index():
    """根路由"""
    return {
        'message': '歡迎使用社工報告自動化系統',
        'version': '2.0',
        'architecture': 'modular'
    }

if __name__ == '__main__':
    print("🚀 啟動重構後的社工報告系統...")
    print("📦 模組化架構已啟用")
    print("🔧 會話管理器: 已初始化")
    print("📁 文件管理器: 已初始化")
    print("📋 處遇計畫管理器: 已初始化")
    
    app.run(host='0.0.0.0', port=5353, debug=True)