from prompt_core.prompt import PromptManager, PromptLibrary
import json
import sys
import argparse
import os

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='人物關係圖對話處理腳本')
    parser.add_argument('--session-id', required=True, help='會話ID')
    parser.add_argument('--input-file', required=True, help='輸入文件路徑')
    parser.add_argument('--message', required=True, help='用戶消息')
    parser.add_argument('--current-graph', required=True, help='當前人物關係圖JSON')
    parser.add_argument('--transcript', default='', help='原始逐字稿')
    parser.add_argument('--config-file', default='person_graph_chat.json', help='配置文件路徑')
    parser.add_argument('--graph-type', default='person', help='關係圖類型 (person/family)')
    
    args = parser.parse_args()
    
    # 檢查配置文件是否存在
    if not os.path.exists(args.config_file):
        print(f"錯誤：配置文件 {args.config_file} 不存在", file=sys.stderr)
        return

    with open(args.config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    default_model_id = config_data.get("default_model_id")
    
    pm = PromptManager(default_model_id=default_model_id)
    conversation_id = f"person_graph_chat_{args.session_id}"
    pm.create_conversation(conversation_id)

    print(f"開始處理人物關係圖對話，會話 {args.session_id}", file=sys.stderr)
    print(f"用戶消息: {args.message}", file=sys.stderr)
    print(f"逐字稿長度: {len(args.transcript)} 字符", file=sys.stderr)

    # 構建增強的系統提示詞
    system_prompt = """你是一個專業的人物關係圖編輯助手。你的任務是根據原始逐字稿和用戶指令來修改人物關係圖。

## 你的能力：
1. 理解用戶的自然語言指令
2. 分析原始逐字稿中的人物關係
3. 參考當前的人物關係圖JSON結構
4. 根據用戶指令智能修改人物關係圖
5. 返回更新後的JSON和清晰的說明

## 上下文信息：

### 原始逐字稿：
{transcript}

### 當前人物關係圖：
{current_graph}

### 用戶指令：
{user_message}

## 輸出要求：
1. **友好回應**：用自然語言解釋你理解了什麼，以及你將如何修改
2. **純JSON格式**：提供完整的、格式正確的人物關係圖JSON，包含nodes（節點）和edges（邊）

## 🚨 重要格式要求（必須嚴格遵守）：

### JSON 結構要求：
- JSON 必須是純JSON格式，不要包含任何 markdown 標記（如 ```json 或 ```）
- JSON 必須是完整且有效的格式
- 必須包含 "nodes" 和 "edges" 兩個主要數組

### Nodes 格式要求：
```
"nodes": [
  {{ "id": "唯一標識符", "label": "顯示名稱" }}
]
```

### Edges 格式要求：
```
"edges": [
  {{ "from": "起始節點ID", "to": "目標節點ID", "label": "關係描述" }}
]
```

### 🔥 關鍵一致性要求：
1. **ID 一致性**：edges 中的 "from" 和 "to" 值必須與 nodes 中的 "id" 完全一致
2. **大小寫敏感**：ID 必須完全匹配，包括大小寫、空格、標點符號
3. **不能有孤立邊**：每個 edge 的 from 和 to 都必須在 nodes 中存在對應的 id
4. **ID 唯一性**：每個 node 的 id 必須是唯一的

### 檢查清單（生成 JSON 前必須檢查）：
- [ ] 所有 nodes 都有唯一的 id 和 label
- [ ] 所有 edges 的 from 值都在 nodes 的 id 中存在
- [ ] 所有 edges 的 to 值都在 nodes 的 id 中存在
- [ ] 沒有重複的 node id
- [ ] JSON 格式完全正確

## 注意事項：
- 如果當前沒有人物關係圖，請基於逐字稿創建一個新的
- 修改時要保持與原始逐字稿的一致性
- 人物名稱要與逐字稿中的一致
- 關係類型要反映逐字稿中的實際情況
- 在生成 JSON 前，務必檢查所有 edges 的 from/to 是否在 nodes 中存在

請先用自然語言回應用戶，然後直接提供純JSON格式的人物關係圖（不要使用代碼塊標記）。

⚠️ 特別提醒：生成 JSON 後，請在心中再次檢查每個 edge 的 from 和 to 是否都對應到實際存在的 node id。"""

    # 格式化提示詞
    formatted_prompt = system_prompt.format(
        transcript=args.transcript or "（無逐字稿）",
        current_graph=args.current_graph or "（無現有人物關係圖）",
        user_message=args.message
    )

    print("-----------------------------------------", file=sys.stderr)
    print(f"[人物關係圖對話 {args.session_id}] 處理用戶指令", file=sys.stderr)
    
    response_content = ""
    json_started = False
    json_content = ""
    brace_count = 0
    
    for chunk in pm.chat(conversation_id, formatted_prompt, model_id=default_model_id, temperature=0.3, stream=True, as_generator=True):
        response_content += chunk
        
        # 檢查是否開始 JSON 部分
        if not json_started:
            if "{" in chunk:
                json_started = True
                # 找到第一個 { 的位置
                start_pos = chunk.find("{")
                json_content = chunk[start_pos:]
                brace_count = json_content.count("{") - json_content.count("}")
            else:
                # 輸出回應內容
                print(json.dumps({"type": "response", "content": chunk}), flush=True)
        else:
            # 已經在 JSON 部分
            json_content += chunk
            brace_count += chunk.count("{") - chunk.count("}")
            
            # 如果大括號平衡，JSON 結束
            if brace_count == 0:
                # 清理 JSON 內容，移除可能的 markdown 標記
                clean_json = json_content.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()
                
                # 驗證 JSON 格式
                try:
                    parsed_json = json.loads(clean_json)
                    
                    # 驗證人物關係圖的一致性
                    if validate_person_graph(parsed_json):
                        # 輸出清理後的 JSON
                        print(json.dumps({"type": "graph", "content": clean_json}), flush=True)
                    else:
                        print(f"人物關係圖一致性檢查失敗", file=sys.stderr)
                        print(json.dumps({"type": "response", "content": "生成的人物關係圖存在一致性問題，請重新生成。"}), flush=True)
                except json.JSONDecodeError:
                    print(f"JSON 格式錯誤: {clean_json}", file=sys.stderr)
                
                json_started = False
                json_content = ""
                brace_count = 0

    print(f"\n人物關係圖對話會話 {args.session_id} 處理完成", file=sys.stderr)
    print("-----------------------------------------", file=sys.stderr)

def validate_person_graph(graph_data):
    """驗證人物關係圖的一致性"""
    try:
        if not isinstance(graph_data, dict):
            return False
        
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return False
        
        # 收集所有 node id
        node_ids = set()
        for node in nodes:
            if not isinstance(node, dict) or 'id' not in node:
                return False
            node_ids.add(node['id'])
        
        # 檢查 edges 的一致性
        for edge in edges:
            if not isinstance(edge, dict):
                return False
            
            from_id = edge.get('from')
            to_id = edge.get('to')
            
            if from_id not in node_ids:
                print(f"Edge from '{from_id}' 不存在於 nodes 中", file=sys.stderr)
                return False
            
            if to_id not in node_ids:
                print(f"Edge to '{to_id}' 不存在於 nodes 中", file=sys.stderr)
                return False
        
        return True
    except Exception as e:
        print(f"驗證過程中發生錯誤: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    main() 