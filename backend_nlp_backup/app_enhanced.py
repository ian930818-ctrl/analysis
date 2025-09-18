"""
Enhanced Flask Application with Advanced Chinese NLP Pipeline
Integrates BERT+CRF, CKIP, Fine-tuned models, and Dictionary support
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import sys
import os
import re
import jieba
import json
from datetime import datetime

# Add the nlp_models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'nlp_models'))

# Import advanced NLP components
try:
    from nlp_models.unified_nlp_pipeline import UnifiedNLPPipeline, create_advanced_nlp_pipeline
    from nlp_models.chinese_dictionary import ChineseDictionary, DictionaryEnhancedNLP
    ADVANCED_NLP_AVAILABLE = True
    print("Advanced NLP pipeline loaded successfully")
except ImportError as e:
    print(f"Advanced NLP not available: {e}")
    ADVANCED_NLP_AVAILABLE = False

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')

CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLP components
nlp_pipeline = None
chinese_dictionary = None

def initialize_nlp_components():
    """Initialize advanced NLP components"""
    global nlp_pipeline, chinese_dictionary
    
    if ADVANCED_NLP_AVAILABLE:
        try:
            print("Initializing advanced NLP pipeline...")
            nlp_pipeline = create_advanced_nlp_pipeline()
            chinese_dictionary = ChineseDictionary()
            print("Advanced NLP pipeline initialized successfully")
        except Exception as e:
            print(f"Failed to initialize advanced NLP: {e}")
            nlp_pipeline = None
            chinese_dictionary = None
    else:
        print("Using fallback NLP methods")

# Initialize on startup
initialize_nlp_components()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Enhanced text analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        print(f"[DEBUG] Analyzing text: {text[:100]}...")
        
        # Use advanced NLP pipeline if available
        if nlp_pipeline:
            characters = analyze_with_advanced_nlp(text)
        else:
            characters = analyze_with_fallback_nlp(text)
        
        print(f"[DEBUG] Extracted {len(characters)} characters")
        
        # Generate relationships (simplified for now)
        relationships = generate_simple_relationships(text, characters)
        
        response = {
            "success": True,
            "text": text,
            "characters": characters,
            "relationships": relationships,
            "analysis_method": "advanced_nlp" if nlp_pipeline else "fallback",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({"error": error_msg}), 500

def analyze_with_advanced_nlp(text: str) -> list:
    """Analyze text using advanced NLP pipeline"""
    try:
        # Use unified NLP pipeline
        result = nlp_pipeline.analyze_text_comprehensive(text)
        
        # Convert to expected format
        characters = []
        for char in result.get('characters', []):
            character_data = {
                "id": f"char_{len(characters)}",
                "name": char.get('name', ''),
                "description": char.get('description', f"{char.get('name', '')} - {char.get('type', 'person')}"),
                "importance": min(5, max(1, len(char.get('behaviors', [])) + 1)),
                "frequency": char.get('vote_count', 1) if 'vote_count' in char else 1,
                "source": char.get('source', 'advanced_nlp'),
                "confidence": char.get('confidence', 0.8),
                "model_sources": char.get('model_sources', []),
                "events": [
                    {"type": behavior.get('category', 'action'), "description": behavior.get('action', '')}
                    for behavior in char.get('behaviors', [])[:3]
                ],
                "attributes": [
                    {"type": "characteristic", "value": attr}
                    for attr in char.get('attributes', [])[:3]
                ],
                "behaviors": [
                    {
                        "category": behavior.get('category', 'general'),
                        "count": 1,
                        "actions": [behavior.get('action', '')]
                    }
                    for behavior in char.get('behaviors', [])[:5]
                ]
            }
            characters.append(character_data)
        
        return characters
        
    except Exception as e:
        print(f"[ERROR] Advanced NLP analysis failed: {e}")
        return analyze_with_fallback_nlp(text)

def analyze_with_fallback_nlp(text: str) -> list:
    """Fallback NLP analysis using patterns and dictionary"""
    characters = []
    
    try:
        # Use dictionary enhancement if available
        if chinese_dictionary:
            characters = analyze_with_dictionary_enhancement(text)
        else:
            characters = analyze_with_basic_patterns(text)
            
    except Exception as e:
        print(f"[ERROR] Dictionary analysis failed: {e}")
        characters = analyze_with_basic_patterns(text)
    
    return characters

def analyze_with_dictionary_enhancement(text: str) -> list:
    """Analyze using dictionary enhancement"""
    try:
        # Initial pattern-based extraction
        initial_chars = analyze_with_basic_patterns(text)
        
        # Enhance with dictionary
        enhanced_nlp = DictionaryEnhancedNLP(chinese_dictionary)
        enhanced_chars = enhanced_nlp.extract_characters_with_dictionary(
            text, 
            lambda t: initial_chars
        )
        
        # Convert to expected format
        characters = []
        for i, char in enumerate(enhanced_chars):
            character_data = {
                "id": f"char_{i}",
                "name": char.get('name', ''),
                "description": char.get('description', f"{char.get('name', '')} - {char.get('type', 'person')}"),
                "importance": min(5, max(1, int(char.get('confidence', 0.5) * 5))),
                "frequency": text.count(char.get('name', '')),
                "source": "dictionary_enhanced",
                "events": [],
                "attributes": [],
                "behaviors": []
            }
            characters.append(character_data)
        
        return characters
        
    except Exception as e:
        print(f"[ERROR] Dictionary enhancement failed: {e}")
        return analyze_with_basic_patterns(text)

def analyze_with_basic_patterns(text: str) -> list:
    """Basic pattern-based character extraction"""
    characters = []
    found_names = set()
    
    # Enhanced patterns
    patterns = [
        # Direct speech patterns
        r'([一-龥]{2,4})說',
        r'([一-龥]{2,4})問',
        r'([一-龥]{2,4})回答',
        
        # Teacher patterns  
        r'([一-龥]+老師)',
        
        # Names starting with 小
        r'(小[一-龥]{1,2})',
        
        # Action subjects
        r'([一-龥]{2,3})和',
        r'([一-龥]{2,3})與',
        
        # Together patterns
        r'和([一-龥]{2,4})一起',
        r'跟([一-龥]{2,4})一起',
        
        # Observation patterns
        r'看到([一-龥]{2,4})',
        r'遇到([一-龥]{2,4})',
    ]
    
    # Process each pattern
    for i, pattern in enumerate(patterns):
        try:
            matches = re.findall(pattern, text)
            if matches:
                print(f"[DEBUG] Pattern {i+1} found: {len(matches)} matches")
                
                for match in matches:
                    name = match.strip() if isinstance(match, str) else match
                    
                    # Filter invalid names
                    if (len(name) >= 2 and len(name) <= 4 and
                        name not in ['他們', '我們', '大家', '這個', '那個', '一個', '所有', '每個']):
                        found_names.add(name)
                        print(f"[DEBUG] Valid character found: {name}")
        except Exception as e:
            print(f"[DEBUG] Pattern {i+1} error: {str(e)}")
    
    # Special handling for teacher characters
    if '老師' in text:
        # Find "X teacher" patterns
        teacher_matches = re.findall(r'([一-龥]+)老師', text)
        for match in teacher_matches:
            if len(match) >= 1:
                teacher_name = match + '老師'
                found_names.add(teacher_name)
                print(f"[DEBUG] Teacher found: {teacher_name}")
        
        # If no specific teacher name found, use generic
        if not any('老師' in name for name in found_names):
            found_names.add('老師')
            print("[DEBUG] Generic teacher added")
    
    print(f"[DEBUG] Final character list: {list(found_names)}")
    
    # Create character objects
    for i, name in enumerate(sorted(found_names)):
        frequency = text.count(name)
        
        # Analyze character type
        if "老師" in name:
            character_type = "教育工作者"
        elif name.startswith("小"):
            character_type = "學生"
        else:
            character_type = "人物"
        
        characters.append({
            "id": f"char_{i}",
            "name": name,
            "description": f"{name} - {character_type}",
            "importance": min(5, max(1, frequency)),
            "frequency": frequency,
            "source": "basic_pattern",
            "events": [],
            "attributes": [],
            "behaviors": []
        })
    
    print(f"[DEBUG] Created {len(characters)} character objects")
    return characters

def generate_simple_relationships(text: str, characters: list) -> list:
    """Generate simple relationships between characters"""
    relationships = []
    
    if len(characters) < 2:
        return relationships
    
    # Find interaction patterns
    for i, char1 in enumerate(characters):
        for j, char2 in enumerate(characters[i+1:], i+1):
            name1 = char1['name']
            name2 = char2['name']
            
            # Check for interaction patterns
            interaction_patterns = [
                f"{name1}和{name2}",
                f"{name1}跟{name2}",
                f"{name2}和{name1}",
                f"{name2}跟{name1}",
            ]
            
            for pattern in interaction_patterns:
                if pattern in text:
                    relationships.append({
                        "id": f"rel_{len(relationships)}",
                        "source": name1,
                        "target": name2,
                        "type": "interaction",
                        "description": f"{name1}和{name2}有互動",
                        "strength": 0.8
                    })
                    break
    
    return relationships

@app.route('/api/nlp-status', methods=['GET'])
def nlp_status():
    """Get NLP pipeline status"""
    
    status = {
        "advanced_nlp_available": ADVANCED_NLP_AVAILABLE,
        "nlp_pipeline_loaded": nlp_pipeline is not None,
        "dictionary_loaded": chinese_dictionary is not None,
        "components": []
    }
    
    if nlp_pipeline:
        try:
            pipeline_info = nlp_pipeline.get_pipeline_info()
            status["components"] = pipeline_info.get("models_loaded", [])
            status["device"] = pipeline_info.get("device", "unknown")
        except Exception as e:
            status["error"] = str(e)
    
    if chinese_dictionary:
        try:
            dict_stats = chinese_dictionary.get_dictionary_stats()
            status["dictionary_stats"] = dict_stats
        except Exception as e:
            status["dictionary_error"] = str(e)
    
    return jsonify(status)

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Endpoint to trigger model training"""
    
    if not ADVANCED_NLP_AVAILABLE:
        return jsonify({"error": "Advanced NLP not available"}), 400
    
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'bert_finetuned')
        
        if model_type == 'bert_finetuned':
            # Trigger fine-tuning
            from nlp_models.bert_finetuning import BertFineTuner
            
            tuner = BertFineTuner(
                model_name="bert-base-chinese",
                output_dir="./chinese_ner_finetuned_new",
                use_wandb=False
            )
            
            # Quick training
            trainer = tuner.train(num_epochs=2, batch_size=4)
            
            return jsonify({
                "success": True,
                "message": "Model training completed",
                "model_path": "./chinese_ner_finetuned_new"
            })
        
        else:
            return jsonify({"error": "Unsupported model type"}), 400
    
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

@app.route('/api/add-dictionary-entry', methods=['POST'])
def add_dictionary_entry():
    """Add entry to dictionary"""
    
    if not chinese_dictionary:
        return jsonify({"error": "Dictionary not available"}), 400
    
    try:
        data = request.get_json()
        entry_type = data.get('type', 'word')  # 'word' or 'name'
        
        if entry_type == 'word':
            word = data.get('word')
            definition = data.get('definition')
            pos_tag = data.get('pos_tag', 'n')
            
            chinese_dictionary.add_word_to_dictionary(word, definition, pos_tag)
            
        elif entry_type == 'name':
            name = data.get('name')
            gender = data.get('gender', 'unisex')
            meaning = data.get('meaning', '')
            
            chinese_dictionary.add_name_to_dictionary(name, gender, meaning)
        
        else:
            return jsonify({"error": "Invalid entry type"}), 400
        
        return jsonify({
            "success": True,
            "message": f"Added {entry_type} to dictionary"
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to add entry: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting Advanced Chinese Character Analysis System...")
    print("Enhanced NLP System Enabled" if ADVANCED_NLP_AVAILABLE else "Using Fallback NLP")
    
    # Print system information
    if nlp_pipeline:
        try:
            info = nlp_pipeline.get_pipeline_info()
            print(f"Loaded models: {', '.join(info.get('models_loaded', []))}")
            print(f"Device: {info.get('device', 'unknown')}")
        except:
            pass
    
    if chinese_dictionary:
        try:
            stats = chinese_dictionary.get_dictionary_stats()
            print(f"Dictionary: {stats['words']} words, {stats['names']} names")
        except:
            pass
    
    print("System started, accessible at http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)