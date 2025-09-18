"""
會話管理模組
負責處理用戶會話、會話鎖管理和會話文件操作
"""

import os
import time
import threading
import uuid


class SessionManager:
    def __init__(self, temp_dir='temp_sessions'):
        self.temp_dir = temp_dir
        self.session_locks = {}
        self.locks_lock = threading.Lock()
        
        # 創建臨時文件目錄
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def get_session_lock(self, session_id: str) -> threading.Lock:
        """獲取會話專用的鎖"""
        with self.locks_lock:
            if session_id not in self.session_locks:
                self.session_locks[session_id] = threading.Lock()
            return self.session_locks[session_id]
    
    def get_session_file_path(self, session_id: str, filename: str) -> str:
        """根據會話ID獲取文件路徑（保持向下相容）"""
        session_dir = os.path.join(self.temp_dir, session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        return os.path.join(session_dir, filename)
    
    def get_step_specific_file_path(self, session_id: str, step: str, file_type: str = 'input') -> str:
        """根據步驟和時間戳獲取唯一文件路徑，避免衝突"""
        timestamp = str(int(time.time() * 1000))  # 使用毫秒時間戳
        filename = f"{step}_{file_type}_{timestamp}.txt"
        
        session_dir = os.path.join(self.temp_dir, session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        
        file_path = os.path.join(session_dir, filename)
        print(f"📁 創建步驟文件: {file_path}")
        return file_path
    
    def get_concurrent_safe_file_path(self, session_id: str, operation: str) -> str:
        """獲取並發安全的文件路徑"""
        thread_id = threading.get_ident()
        timestamp = str(int(time.time() * 1000))
        filename = f"{operation}_{thread_id}_{timestamp}.txt"
        
        session_dir = os.path.join(self.temp_dir, session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        
        return os.path.join(session_dir, filename)
    
    def cleanup_session_files(self, session_id: str):
        """清理會話文件"""
        session_dir = os.path.join(self.temp_dir, session_id)
        if os.path.exists(session_dir):
            import shutil
            shutil.rmtree(session_dir)
    
    def cleanup_old_session_files(self):
        """清理老舊會話文件"""
        try:
            current_time = time.time()
            
            for session_id in os.listdir(self.temp_dir):
                session_dir = os.path.join(self.temp_dir, session_id)
                if not os.path.isdir(session_dir):
                    continue
                    
                # 檢查會話目錄的最後修改時間
                if current_time - os.path.getmtime(session_dir) > 86400:  # 24小時
                    try:
                        import shutil
                        shutil.rmtree(session_dir)
                        print(f"🗑️ 清理過期會話: {session_id}")
                        
                        # 清理對應的會話鎖
                        with self.locks_lock:
                            if session_id in self.session_locks:
                                del self.session_locks[session_id]
                                
                    except Exception as e:
                        print(f"⚠️ 清理會話失敗 {session_id}: {e}")
                        
                else:
                    # 清理會話內的老舊步驟文件
                    self._cleanup_old_step_files(session_dir)
                    
        except Exception as e:
            print(f"⚠️ 會話清理工作失敗: {e}")
    
    def _cleanup_old_step_files(self, session_dir: str):
        """清理會話內老舊的步驟文件"""
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
                            print(f"🗑️ 清理步驟文件: {file_path}")
                    except Exception as e:
                        print(f"⚠️ 清理步驟文件失敗 {file_path}: {e}")
                        
        except Exception as e:
            print(f"⚠️ 步驟文件清理失敗: {e}")


# 全局會話管理器實例
session_manager = SessionManager()