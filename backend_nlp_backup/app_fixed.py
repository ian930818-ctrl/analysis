"""
Fixed Flask Application - Simple and Effective Character Extraction
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import re
import json
from datetime import datetime

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')

CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Fixed text analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        print(f"[DEBUG] Analyzing text length: {len(text)}")
        print(f"[DEBUG] Text preview: {text[:50]}...")
        
        # Extract characters using improved method
        characters = extract_characters_fixed(text)
        
        print(f"[DEBUG] Found {len(characters)} characters: {[c['name'] for c in characters]}")
        
        # Generate simple relationships
        relationships = []
        if len(characters) >= 2:
            for i, char1 in enumerate(characters):
                for char2 in characters[i+1:]:
                    name1, name2 = char1['name'], char2['name']
                    if f"{name1}和{name2}" in text or f"{name2}和{name1}" in text:
                        relationships.append({
                            "id": f"rel_{len(relationships)}",
                            "source": name1,
                            "target": name2,
                            "type": "interaction",
                            "description": f"{name1}和{name2}有互動"
                        })
        
        response = {
            "success": True,
            "text": text,
            "characters": characters,
            "relationships": relationships,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({"error": error_msg}), 500

def extract_characters_fixed(text):
    """Fixed character extraction with comprehensive patterns"""
    print("[DEBUG] Starting character extraction")
    
    characters = []
    found_names = set()
    
    # Comprehensive Chinese character patterns
    patterns = [
        # 1. Direct speech patterns (most reliable)
        (r'([一-龥]{2,4})說', 'speaker'),
        (r'([一-龥]{2,4})問', 'speaker'),
        (r'([一-龥]{2,4})回答', 'speaker'),
        (r'([一-龥]{2,4})講', 'speaker'),
        
        # 2. Teacher patterns (high priority)
        (r'([一-龥]{1,3})老師', 'teacher'),
        
        # 3. Student patterns (common prefix)
        (r'(小[一-龥]{1,2})', 'student'),
        
        # 4. Action subjects
        (r'([一-龥]{2,4})去', 'actor'),
        (r'([一-龥]{2,4})來', 'actor'),
        (r'([一-龥]{2,4})做', 'actor'),
        (r'([一-龥]{2,4})看', 'observer'),
        (r'([一-龥]{2,4})想', 'thinker'),
        (r'([一-龥]{2,4})學習', 'learner'),
        (r'([一-龥]{2,4})讀書', 'reader'),
        
        # 5. Interaction patterns
        (r'([一-龥]{2,4})和', 'partner'),
        (r'和([一-龥]{2,4})一起', 'companion'),
        (r'跟([一-龥]{2,4})一起', 'companion'),
        
        # 6. Possession/relationship patterns
        (r'([一-龥]{2,4})的', 'possessor'),
        
        # 7. Observation patterns
        (r'看到([一-龥]{2,4})', 'observed'),
        (r'遇到([一-龥]{2,4})', 'encountered'),
        
        # 8. Specific role patterns
        (r'([一-龥]{2,4})同學', 'classmate'),
        (r'([一-龥]{2,4})朋友', 'friend'),
    ]
    
    # Process each pattern
    for i, (pattern, role) in enumerate(patterns):
        try:
            matches = re.findall(pattern, text)
            if matches:
                print(f"[DEBUG] Pattern {i+1} ({role}): found {len(matches)} matches - {matches}")
                
                for match in matches:
                    name = match.strip()
                    
                    # Validate name
                    if is_valid_character_name(name):
                        found_names.add(name)
                        print(f"[DEBUG] Valid character: {name} (role: {role})")
                    else:
                        print(f"[DEBUG] Rejected: {name} (invalid)")
        except Exception as e:
            print(f"[DEBUG] Pattern {i+1} error: {str(e)}")
    
    # Special processing for teachers
    if '老師' in text:
        # Find specific teacher names
        teacher_patterns = [
            r'([一-龥]{1,3})老師',
            r'老師([一-龥]{1,3})',
        ]
        
        teacher_found = False
        for pattern in teacher_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match and len(match) >= 1:
                    if pattern.endswith('老師'):
                        teacher_name = match + '老師'
                    else:
                        teacher_name = '老師' + match
                    
                    if is_valid_character_name(teacher_name):
                        found_names.add(teacher_name)
                        teacher_found = True
                        print(f"[DEBUG] Teacher found: {teacher_name}")
        
        # Fallback to generic teacher
        if not teacher_found and not any('老師' in name for name in found_names):
            found_names.add('老師')
            print("[DEBUG] Added generic teacher")
    
    print(f"[DEBUG] All found names: {list(found_names)}")
    
    # Create character objects with enhanced information
    for i, name in enumerate(sorted(found_names)):
        char_type = classify_character_type(name)
        frequency = text.count(name)
        behaviors = extract_character_behaviors(name, text)
        
        character = {
            "id": f"char_{i}",
            "name": name,
            "description": f"{name} - {char_type}",
            "importance": min(5, max(1, frequency + len(behaviors))),
            "frequency": frequency,
            "source": "pattern_based",
            "events": [],
            "attributes": [],
            "behaviors": behaviors
        }
        
        characters.append(character)
        print(f"[DEBUG] Created character: {name} (type: {char_type}, freq: {frequency})")
    
    print(f"[DEBUG] Final result: {len(characters)} characters")
    return characters

def is_valid_character_name(name):
    """Validate if a name is a valid character"""
    if not name or len(name) < 2 or len(name) > 4:
        return False
    
    # Blacklist of invalid names
    invalid_names = {
        '他們', '我們', '大家', '這個', '那個', '一個', '所有', '每個',
        '什麼', '怎麼', '為什麼', '哪個', '今天', '明天', '昨天',
        '上課', '下課', '學習', '讀書', '很好', '非常', '特別',
        '應該', '可以', '不能', '沒有', '還有', '已經', '正在'
    }
    
    if name in invalid_names:
        return False
    
    # Must contain Chinese characters
    if not re.match(r'^[一-龥]+$', name):
        return False
    
    return True

def classify_character_type(name):
    """Classify character type based on name"""
    if '老師' in name or '教師' in name:
        return '教師'
    elif name.startswith('小') and len(name) <= 3:
        return '學生'
    elif '同學' in name:
        return '同學'
    elif '朋友' in name:
        return '朋友'
    elif any(title in name for title in ['先生', '女士', '小姐']):
        return '成人'
    else:
        return '人物'

def extract_character_behaviors(character, text):
    """Extract behaviors for a specific character"""
    behaviors = []
    
    # Behavior patterns
    behavior_patterns = {
        '說話': [f'{character}說', f'{character}問', f'{character}回答', f'{character}講'],
        '行動': [f'{character}去', f'{character}來', f'{character}走', f'{character}跑'],
        '學習': [f'{character}學習', f'{character}讀書', f'{character}寫', f'{character}上課'],
        '觀察': [f'{character}看', f'{character}見', f'{character}注意'],
        '思考': [f'{character}想', f'{character}思考', f'{character}考慮'],
        '互動': [f'{character}和', f'和{character}', f'{character}跟', f'跟{character}']
    }
    
    for category, patterns in behavior_patterns.items():
        count = 0
        actions = []
        
        for pattern in patterns:
            if pattern in text:
                count += text.count(pattern)
                actions.append(pattern)
        
        if count > 0:
            behaviors.append({
                "category": category,
                "count": count,
                "actions": actions[:3]  # Limit to top 3 actions
            })
    
    return behaviors

if __name__ == '__main__':
    print("Starting Fixed Character Analysis System...")
    print("System optimized for Chinese character recognition")
    print("Access at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)