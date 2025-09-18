#!/usr/bin/env python3
"""
關係圖功能完整性測試腳本
"""

import requests
import json
import time
import re

def clean_json_content(content):
    """清理AI返回的內容，提取純JSON"""
    # 移除markdown代碼塊標記
    content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
    
    # 嘗試找到JSON內容的開始和結束
    # 對於數組格式 [...]
    array_match = re.search(r'(\[.*\])', content, re.DOTALL)
    if array_match:
        return array_match.group(1).strip()
    
    # 對於對象格式 {...}
    object_match = re.search(r'(\{.*\})', content, re.DOTALL)
    if object_match:
        return object_match.group(1).strip()
    
    # 如果都找不到，返回原內容
    return content.strip()

def test_person_graph_generation():
    """測試人物關係圖生成"""
    print("🧪 測試人物關係圖生成...")
    
    test_data = {
        "text": """
        訪視紀錄
        
        案主王小美，35歲，與丈夫陳大明（38歲）結婚10年，育有兩子：
        - 大兒子陳小華，8歲，就讀小學二年級
        - 小兒子陳小明，5歲，就讀幼稚園
        
        案主的母親林阿嬤，65歲，偶爾會來幫忙照顧孫子。
        案主的姊姊王大美，40歲，住在附近，關係良好。
        
        丈夫的父親陳老爺爺，70歲，身體不太好，需要照護。
        """,
        "graphType": "person",
        "sessionId": "test_person_graph"
    }
    
    try:
        response = requests.post(
            'http://localhost:5353/api/PersonGraph',
            headers={'Content-Type': 'application/json'},
            json=test_data,
            stream=True,
            timeout=30
        )
        
        print(f"📡 人物關係圖API響應狀態: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ 人物關係圖測試失敗，狀態碼: {response.status_code}")
            return False
        
        # 收集流式響應
        content_parts = []
        for line in response.iter_lines():
            if line:
                try:
                    line_str = line.decode('utf-8')
                    if line_str.strip():
                        data = json.loads(line_str)
                        if 'content' in data:
                            content_parts.append(data['content'])
                            print(".", end="", flush=True)
                except json.JSONDecodeError:
                    continue
        
        full_content = ''.join(content_parts)
        print(f"\n📄 人物關係圖JSON長度: {len(full_content)} 字元")
        
        # 清理JSON內容
        cleaned_content = clean_json_content(full_content)
        
        # 驗證JSON格式
        try:
            graph_data = json.loads(cleaned_content)
            
            # 檢查基本結構
            if 'nodes' not in graph_data or 'edges' not in graph_data:
                print("❌ JSON結構不完整，缺少nodes或edges")
                return False
            
            nodes_count = len(graph_data['nodes'])
            edges_count = len(graph_data['edges'])
            print(f"✅ 人物關係圖包含 {nodes_count} 個節點，{edges_count} 個關係")
            
            # 檢查是否包含主要人物
            node_labels = [node.get('label', '') for node in graph_data['nodes']]
            if any('王小美' in label or '案主' in label for label in node_labels):
                print("✅ 包含案主資訊")
            else:
                print("⚠️ 可能缺少案主資訊")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ 生成的JSON格式錯誤: {e}")
            print(f"內容預覽: {full_content[:200]}...")
            return False
        
    except Exception as e:
        print(f"❌ 人物關係圖測試異常: {e}")
        return False

# 移除家庭關係圖測試功能

def test_graph_chat_functionality():
    """測試關係圖對話功能"""
    print("\n🧪 測試關係圖對話功能...")
    
    test_data = {
        "message": "請加入案主的朋友小李，他們是大學同學",
        "currentGraph": '{"nodes": [{"id": "案主", "label": "案主"}], "edges": []}',
        "transcript": "案主提到她的大學同學小李經常會來探望",
        "graphType": "person",
        "sessionId": "test_graph_chat"
    }
    
    try:
        response = requests.post(
            'http://localhost:5353/api/PersonGraphChat',
            headers={'Content-Type': 'application/json'},
            json=test_data,
            stream=True,
            timeout=20
        )
        
        print(f"📡 對話功能API響應狀態: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ 對話功能測試失敗，狀態碼: {response.status_code}")
            return False
        
        # 收集響應
        content_parts = []
        for line in response.iter_lines():
            if line:
                try:
                    line_str = line.decode('utf-8')
                    if line_str.strip():
                        data = json.loads(line_str)
                        if 'content' in data:
                            content_parts.append(data['content'])
                            print(".", end="", flush=True)
                except json.JSONDecodeError:
                    continue
        
        full_content = ''.join(content_parts)
        print(f"\n📄 對話回應長度: {len(full_content)} 字元")
        
        if len(full_content) > 50:
            print("✅ 對話功能響應正常")
            return True
        else:
            print("⚠️ 對話功能響應可能過短")
            return False
        
    except Exception as e:
        print(f"❌ 對話功能測試異常: {e}")
        return False

if __name__ == '__main__':
    print("🚀 開始關係圖功能完整性測試...\n")
    
    # 等待服務啟動
    print("⏳ 等待服務啟動...")
    time.sleep(2)
    
    test_results = []
    
    # 執行測試
    test_results.append(test_person_graph_generation())
    # 移除家庭關係圖測試
    test_results.append(test_graph_chat_functionality())
    
    # 總結
    success_count = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 關係圖功能測試總結: {success_count}/{total_tests} 通過")
    
    if success_count == total_tests:
        print("🎉 所有關係圖功能測試通過！")
    else:
        print("⚠️ 部分功能需要進一步檢查")