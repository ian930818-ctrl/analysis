from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import re
import anthropic
import os

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

# 讀取配置文件
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("配置文件 config.json 不存在，使用環境變量")
        return None
    except json.JSONDecodeError:
        print("配置文件格式錯誤，使用環境變量")
        return None

config = load_config()

# 初始化Claude API
if config and config.get('claude_api_key'):
    CLAUDE_API_KEY = config['claude_api_key']
    print("[OK] 從 config.json 讀取 API 密鑰")
else:
    CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
    if not CLAUDE_API_KEY:
        print("警告: 未設置CLAUDE_API_KEY環境變量")
        CLAUDE_API_KEY = ""

claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/hello', methods=['GET'])
def hello_api():
    return jsonify({"message": "Hello from LLM-integrated backend!"})

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    return jsonify({"received": data, "status": "success"})

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        manual_corrections = data.get('manual_corrections', {})
        
        print(f"[DEBUG] 收到分析請求，文本長度: {len(text)}")
        print(f"[DEBUG] 文本前100字符: {text[:100]}...")
        
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400
        
        # 使用Claude API進行人物提取
        print("[DEBUG] 開始Claude人物提取...")
        characters = extract_characters_with_claude(text)
        print(f"[DEBUG] Claude提取到 {len(characters)} 個人物")
        
        # 顯示提取到的人物名稱
        if characters:
            names = [char['name'] for char in characters]
            print(f"[DEBUG] 識別的人物: {', '.join(names)}")
        
        # Apply manual corrections if provided
        if manual_corrections:
            characters = apply_manual_corrections(characters, manual_corrections)
        
        # 使用Claude API進行關係分析
        print("[DEBUG] 開始Claude關係分析...")
        relationships = generate_relationships_with_claude(text, characters)
        print(f"[DEBUG] 生成了 {len(relationships)} 個關係")
        
        result = {
            "characters": characters,
            "relationships": relationships,
            "status": "success",
            "corrections_applied": len(manual_corrections) > 0,
            "source": "claude_api"
        }
        
        print(f"[DEBUG] 返回結果: {len(characters)} 個人物, {len(relationships)} 個關係")
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] 分析出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"分析失敗: {str(e)}"}), 500

def extract_characters_with_claude(text):
    """使用Claude API提取人物和詳細行為"""
    if not claude_client:
        print("[WARNING] Claude API未初始化，使用降級模式")
        return simple_extract_characters(text)

    try:
        prompt = f"""請仔細分析以下文本，識別其中的人物角色，並分析每個人物的行為、動作和與其他人物的關係。

請用JSON格式回應，包含詳細信息：
{{
  "characters": [
    {{
      "name": "人物名稱",
      "description": "人物的簡短描述或身份",
      "behaviors": [
        "具體描述這個人物做了什麼事情",
        "描述與其他人物的互動",
        "其他重要行為或特徵"
      ],
      "relationships": [
        "與某某人的關係：朋友/同事/家人等",
        "與某某人一起做了什麼事情"
      ]
    }}
  ]
}}

請確保behaviors欄位詳細描述人物的具體行為和動作，relationships欄位描述與其他人物的關係和互動。

文本：{text}"""

        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text
        print(f"[DEBUG] Claude人物回應: {response_text}")

        # 解析JSON
        if "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_text = response_text[json_start:json_end]

            parsed = json.loads(json_text)
            raw_characters = parsed.get("characters", [])

            # 轉換為標準格式
            characters = []
            for i, char_data in enumerate(raw_characters):
                name = char_data.get("name", f"人物{i+1}")
                description = char_data.get("description", f"{name} - Claude識別的人物")
                behaviors = char_data.get("behaviors", [])
                relationships = char_data.get("relationships", [])

                # 合併行為和關係信息
                all_behaviors = behaviors + relationships

                characters.append({
                    "id": f"char_{i}",
                    "name": name,
                    "description": description,
                    "importance": 3,
                    "frequency": text.count(name),
                    "confidence": 0.9,
                    "source": "claude_api",
                    "events": [],
                    "attributes": [],
                    "behaviors": all_behaviors
                })

            return characters
        else:
            print("[WARNING] Claude回應不包含JSON")
            return []

    except Exception as e:
        print(f"[ERROR] Claude人物提取失敗: {str(e)}")
        # 降級到簡單正則提取
        return simple_extract_characters(text)

def generate_relationships_with_claude(text, characters):
    """使用Claude API生成關係"""
    if len(characters) < 2:
        return []
        
    if not claude_client:
        print("[WARNING] Claude API未初始化，無法生成關係")
        return []
        
    try:
        character_names = [char['name'] for char in characters]
        prompt = f"""分析以下人物之間的關係：{', '.join(character_names)}

請用JSON格式回應：
{{"relationships": [{{"source": "人物1", "target": "人物2", "type": "朋友", "strength": 3}}]}}

文本：{text}"""

        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        print(f"[DEBUG] Claude關係回應: {response_text}")
        
        # 解析JSON
        if "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_text = response_text[json_start:json_end]
            
            parsed = json.loads(json_text)
            raw_relationships = parsed.get("relationships", [])
            
            # 轉換為標準格式
            relationships = []
            char_name_to_id = {char['name']: char['id'] for char in characters}
            
            for i, rel in enumerate(raw_relationships):
                source_name = rel.get("source", "")
                target_name = rel.get("target", "")
                
                source_id = char_name_to_id.get(source_name)
                target_id = char_name_to_id.get(target_name)
                
                if source_id and target_id and source_id != target_id:
                    relationships.append({
                        "id": f"rel_{i}",
                        "source": source_id,
                        "target": target_id,
                        "type": rel.get("type", "一般關係"),
                        "strength": rel.get("strength", 3),
                        "details": {
                            "cooccurrence": 1,
                            "interactions": ["Claude分析"],
                            "emotional_tone": "中性"
                        }
                    })
            
            return relationships
        else:
            return []
            
    except Exception as e:
        print(f"[ERROR] Claude關係分析失敗: {str(e)}")
        return []

def simple_extract_characters(text):
    """簡單的正則表達式人物提取（降級模式）"""
    characters = []
    patterns = [
        r'([一-龥]{2,4})說',
        r'([一-龥]{2,4})問',
        r'([一-龥]+老師)',
        r'(小[一-龥]{1,2})',
    ]
    
    found_names = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            name = match.strip()
            if len(name) >= 2 and len(name) <= 4:
                if name not in ['他們', '我們', '大家', '這個', '那個']:
                    found_names.add(name)
    
    for i, name in enumerate(sorted(found_names)):
        frequency = text.count(name)
        characters.append({
            "id": f"char_{i}",
            "name": name,
            "description": f"{name} - 簡單提取",
            "importance": min(5, max(1, frequency)),
            "frequency": frequency,
            "confidence": 0.6,
            "source": "fallback_regex",
            "events": [],
            "attributes": [],
            "behaviors": []
        })
    
    return characters

@app.route('/api/correct-characters', methods=['POST'])
def correct_characters():
    """Endpoint for manual character corrections"""
    data = request.get_json()
    original_characters = data.get('characters', [])
    corrections = data.get('corrections', {})
    
    corrected_characters = apply_manual_corrections(original_characters, corrections)
    
    return jsonify({
        "characters": corrected_characters,
        "status": "success"
    })

def apply_manual_corrections(characters, corrections):
    """Apply manual corrections to character list"""
    corrected = []
    
    for char in characters:
        char_id = char['id']
        char_name = char['name']
        
        # Check for corrections
        if char_id in corrections:
            correction = corrections[char_id]
            
            if correction['action'] == 'remove':
                continue  # Skip this character
            elif correction['action'] == 'rename':
                char['name'] = correction['new_name']
                char['description'] = correction.get('new_description', char['description'])
            elif correction['action'] == 'modify':
                char.update(correction['updates'])
        
        corrected.append(char)
    
    # Add manually added characters
    for correction in corrections.values():
        if correction.get('action') == 'add':
            new_char = {
                'id': f"manual_{len(corrected)}",
                'name': correction['name'],
                'description': correction.get('description', f"{correction['name']} - 手動添加角色"),
                'importance': correction.get('importance', 3),
                'frequency': correction.get('frequency', 1),
                'events': [],
                'attributes': []
            }
            corrected.append(new_char)
    
    return corrected

@app.route('/api/llm-status', methods=['GET'])
def llm_status():
    """檢查Claude API狀態"""
    if not claude_client:
        return jsonify({
            "status": "error",
            "message": "Claude API未初始化",
            "fallback": "simple_extraction"
        })
        
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            temperature=0.0,
            messages=[{"role": "user", "content": "測試"}]
        )
        
        response_text = response.content[0].text if response.content else ""
        
        return jsonify({
            "status": "loaded",
            "model_type": "Claude API Direct Integration",
            "model": "claude-3-5-sonnet-20241022",
            "test_response_length": len(response_text)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "fallback": "simple_extraction"
        })

if __name__ == '__main__':
    print("[START] Starting Claude-integrated Text Analysis System...")
    print("[INFO] Direct Claude API integration")
    print("[INFO] Fallback mode: simple regex extraction")
    
    app.run(debug=True, host='0.0.0.0', port=5001)