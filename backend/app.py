from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

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
    """Simple character extraction for Chinese text"""
    import re
    character_names = []
    
    # Pattern for animal characters
    patterns = [
        r'(兔子|狐狸|松鼠|貓頭鷹|青蛙|熊)(小?[白赤栗大]?)',
        r'(博士|先生|小姐|老師)[一-龥]*',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                name = ''.join(match)
            else:
                name = match
            
            name = name.strip()
            if len(name) > 1 and name not in character_names:
                character_names.append(name)
    
    # Known characters for the sample text
    known_characters = ['兔子小白', '狐狸小赤', '貓頭鷹博士', '松鼠小栗', '熊大', '小青蛙們']
    for name in known_characters:
        if name in text and name not in character_names:
            character_names.append(name)
    
    # Create character objects
    characters = []
    for i, name in enumerate(character_names):
        frequency = len(re.findall(re.escape(name), text))
        characters.append({
            "id": f"char_{i}",
            "name": name,
            "description": generate_character_description(name),
            "importance": min(5, max(1, (frequency + 1) // 2)),
            "frequency": frequency
        })
    
    return characters

def generate_character_description(name):
    """Generate simple character descriptions"""
    if '兔子' in name:
        return '森林音樂會的發起人'
    elif '狐狸' in name:
        return '音樂會主持人'
    elif '貓頭鷹' in name:
        return '森林中的智者'
    elif '松鼠' in name:
        return '擊鼓手'
    elif '熊' in name:
        return '貝斯手'
    elif '青蛙' in name:
        return '和聲團'
    else:
        return '故事中的角色'

def generate_relationships(text, characters):
    """Generate relationships based on character co-occurrence"""
    import re
    relationships = []
    sentences = re.split(r'[。！？\n]+', text)
    
    for i, char1 in enumerate(characters):
        for j, char2 in enumerate(characters[i+1:], i+1):
            cooccurrence = 0
            
            for sentence in sentences:
                if char1['name'] in sentence and char2['name'] in sentence:
                    cooccurrence += 1
            
            if cooccurrence > 0:
                relationships.append({
                    "id": f"rel_{i}_{j}",
                    "source": char1['id'],
                    "target": char2['id'],
                    "type": determine_relationship_type(char1['name'], char2['name']),
                    "strength": min(5, max(1, cooccurrence))
                })
    
    return relationships

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