from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import jieba
import jieba.posseg as pseg
import re
from collections import Counter

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/hello', methods=['GET'])
def hello_api():
    return jsonify({"message": "Hello from backend!"})

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    return jsonify({"received": data, "status": "success"})

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    # Simple character extraction
    characters = extract_characters(text)
    relationships = generate_relationships(text, characters)
    
    return jsonify({
        "characters": characters,
        "relationships": relationships,
        "status": "success"
    })

def extract_characters(text):
    """Advanced character extraction using hybrid approach"""
    character_names = set()
    
    # Method 1: Enhanced regex patterns for various character types
    patterns = [
        # Traditional names
        r'[一-龥]{2,4}(?=說|問|答|喊|笑|哭|想|看|聽|跑|走|來|去)',
        # Names with titles
        r'[一-龥]{1,3}(?:先生|小姐|博士|老師|師傅|大人|老爺|夫人)',
        # Names with prefixes
        r'(?:小|老|大)[一-龥]{1,3}',
        # Animal characters
        r'[一-龥]*(?:兔子|狐狸|松鼠|貓頭鷹|青蛙|熊|貓|狗|鳥|魚)[一-龥]*',
        # Names in quotes or after specific words
        r'(?:叫做|名叫|是)[一-龥]{2,4}',
        # Family relations
        r'[一-龥]{1,3}(?:爸爸|媽媽|哥哥|姐姐|弟弟|妹妹|爺爺|奶奶)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            name = match.strip()
            if len(name) > 1 and len(name) <= 6:  # Reasonable name length
                character_names.add(name)
    
    # Method 2: Jieba word segmentation with POS tagging
    words = pseg.cut(text)
    for word, flag in words:
        # Look for person names (nr) and other name-like tags
        if flag in ['nr', 'nrt'] and len(word) >= 2:
            character_names.add(word)
        # Look for words that could be names before action verbs
        elif flag in ['n', 'ns'] and len(word) >= 2 and len(word) <= 4:
            if any(action in text[text.find(word):text.find(word)+20] 
                   for action in ['說', '問', '答', '喊', '笑', '哭', '想', '看']):
                character_names.add(word)
    
    # Method 3: Context-based extraction
    # Find quoted speech and extract speakers
    dialogue_pattern = r'([一-龥]{1,4})(?:說|問|答|喊|笑著說|哭著說)[:：]?[「『"]([^」』"]+)[」』"]'
    dialogue_matches = re.findall(dialogue_pattern, text)
    for speaker, _ in dialogue_matches:
        if len(speaker) >= 2:
            character_names.add(speaker)
    
    # Method 4: Frequency-based filtering and validation
    validated_characters = []
    for name in character_names:
        # Count frequency
        frequency = len(re.findall(re.escape(name), text))
        
        # Skip if too infrequent or contains common words
        if frequency < 1:
            continue
            
        # Skip common words that aren't likely to be names
        skip_words = ['東西', '地方', '時候', '事情', '問題', '辦法', '樣子', '大家', '他們', '我們']
        if name in skip_words:
            continue
            
        # Calculate importance based on frequency and length
        importance = min(5, max(1, frequency + len(name) // 2))
        
        validated_characters.append({
            "name": name,
            "frequency": frequency,
            "importance": importance
        })
    
    # Sort by frequency and importance
    validated_characters.sort(key=lambda x: (x['frequency'], x['importance']), reverse=True)
    
    # Create final character objects
    characters = []
    for i, char_data in enumerate(validated_characters[:15]):  # Limit to top 15
        characters.append({
            "id": f"char_{i}",
            "name": char_data["name"],
            "description": generate_character_description(char_data["name"], text),
            "importance": char_data["importance"],
            "frequency": char_data["frequency"]
        })
    
    return characters

def generate_character_description(name, text=""):
    """Generate intelligent character descriptions based on context"""
    # Extract context around the character name
    context_actions = []
    sentences = re.split(r'[。！？\n]+', text)
    
    for sentence in sentences:
        if name in sentence:
            # Extract actions associated with this character
            action_patterns = [
                r'{}.*?([說問答喊笑哭想看聽跑走來去做拿給])'.format(re.escape(name)),
                r'{}.*?([是有會能]).*?([一-龥]{{1,3}})'.format(re.escape(name)),
            ]
            for pattern in action_patterns:
                matches = re.findall(pattern, sentence)
                context_actions.extend([match if isinstance(match, str) else ''.join(match) for match in matches])
    
    # Generate description based on context
    if context_actions:
        common_actions = Counter(context_actions).most_common(2)
        action_desc = '、'.join([action[0] for action in common_actions])
        return f'經常{action_desc}的角色'
    
    # Fallback to pattern-based descriptions
    if any(animal in name for animal in ['兔子', '狐狸', '松鼠', '貓頭鷹', '青蛙', '熊', '貓', '狗', '鳥', '魚']):
        return f'{name} - 動物角色'
    elif any(title in name for title in ['博士', '先生', '小姐', '老師', '師傅']):
        return f'{name} - 專業人士'
    elif any(family in name for family in ['爸爸', '媽媽', '哥哥', '姐姐', '弟弟', '妹妹', '爺爺', '奶奶']):
        return f'{name} - 家庭成員'
    elif name.startswith('小'):
        return f'{name} - 年輕角色'
    elif name.startswith('老'):
        return f'{name} - 年長角色'
    else:
        return f'{name} - 故事角色'

def generate_relationships(text, characters):
    """Advanced relationship analysis based on multiple factors"""
    relationships = []
    sentences = re.split(r'[。！？\n]+', text)
    
    for i, char1 in enumerate(characters):
        for j, char2 in enumerate(characters[i+1:], i+1):
            relationship_data = analyze_character_relationship(char1['name'], char2['name'], text, sentences)
            
            if relationship_data['strength'] > 0:
                relationships.append({
                    "id": f"rel_{i}_{j}",
                    "source": char1['id'],
                    "target": char2['id'],
                    "type": relationship_data['type'],
                    "strength": relationship_data['strength'],
                    "details": relationship_data['details']
                })
    
    return relationships

def analyze_character_relationship(name1, name2, text, sentences):
    """Analyze relationship between two characters"""
    cooccurrence = 0
    interaction_types = []
    emotional_context = []
    
    for sentence in sentences:
        if name1 in sentence and name2 in sentence:
            cooccurrence += 1
            
            # Analyze interaction types
            if any(word in sentence for word in ['說', '問', '答', '對話', '交談']):
                interaction_types.append('對話')
            if any(word in sentence for word in ['一起', '共同', '合作', '幫助']):
                interaction_types.append('合作')
            if any(word in sentence for word in ['爭吵', '吵架', '生氣', '反對']):
                interaction_types.append('衝突')
            if any(word in sentence for word in ['朋友', '喜歡', '愛', '關心']):
                interaction_types.append('友好')
            if any(word in sentence for word in ['家人', '父子', '母女', '兄弟', '姐妹']):
                interaction_types.append('家庭')
                
            # Analyze emotional context
            if any(word in sentence for word in ['開心', '快樂', '高興', '笑']):
                emotional_context.append('正面')
            elif any(word in sentence for word in ['難過', '傷心', '生氣', '哭']):
                emotional_context.append('負面')
    
    # Determine relationship type
    if interaction_types:
        most_common_interaction = Counter(interaction_types).most_common(1)[0][0]
        relationship_type = most_common_interaction
    else:
        relationship_type = determine_relationship_type(name1, name2)
    
    # Calculate relationship strength
    strength = min(5, max(1, cooccurrence))
    if '合作' in interaction_types:
        strength += 1
    if '友好' in interaction_types:
        strength += 1
    if '衝突' in interaction_types:
        strength = max(1, strength - 1)
    
    strength = min(5, strength)
    
    return {
        'strength': strength,
        'type': relationship_type,
        'details': {
            'cooccurrence': cooccurrence,
            'interactions': list(set(interaction_types)),
            'emotional_tone': Counter(emotional_context).most_common(1)[0][0] if emotional_context else '中性'
        }
    }

def determine_relationship_type(name1, name2):
    """Determine relationship type based on character names"""
    if (('兔子' in name1 or '兔子' in name2) and 
        ('狐狸' in name1 or '狐狸' in name2)):
        return '合作夥伴'
    elif '博士' in name1 or '博士' in name2:
        return '師生'
    else:
        return '朋友'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)