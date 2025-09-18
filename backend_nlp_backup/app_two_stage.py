"""
Two-Stage Character Analysis: Sentence Segmentation + Character Extraction
Stage 1: Sentence segmentation with POS tagging
Stage 2: Entity recognition with attribute extraction and cross-validation
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import re
import json
import jieba
import jieba.posseg as pseg
from datetime import datetime
from collections import defaultdict, Counter

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')

CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwoStageNLPProcessor:
    """Two-stage NLP processor with sentence segmentation and entity extraction"""
    
    def __init__(self):
        # Initialize jieba with custom dictionary
        self.setup_custom_dictionary()
        
        # POS tag mappings
        self.pos_categories = {
            'noun': ['n', 'nr', 'nz', 'nt', 'ns', 'ni', 'nw'],
            'verb': ['v', 'vd', 'vn', 'vf', 'vx', 'vi', 'vl', 'vg'],
            'adjective': ['a', 'ad', 'an', 'ag', 'al'],
            'pronoun': ['r', 'rr', 'rz', 'rzt', 'rzs', 'rzv', 'ry', 'ryt', 'rys', 'ryv'],
            'adverb': ['d', 'dg', 'dl', 'dt'],
            'conjunction': ['c', 'cc'],
            'preposition': ['p', 'pba', 'pbei'],
            'particle': ['u', 'uz', 'ug', 'ul', 'uv', 'uj'],
            'number': ['m', 'mq'],
            'time': ['t', 'tg']
        }
        
        # Entity patterns with validation rules
        self.entity_patterns = self.build_entity_patterns()
        
        print("[INIT] Two-stage NLP processor initialized")
    
    def setup_custom_dictionary(self):
        """Setup custom dictionary for better segmentation"""
        custom_words = [
            # Names
            "小明 10 nr", "小華 10 nr", "小美 10 nr", "小強 10 nr", "小紅 10 nr",
            "小志 10 nr", "小文 10 nr", "小玲 10 nr", "小芳 10 nr", "小傑 10 nr",
            
            # Teachers
            "王老師 20 nr", "李老師 20 nr", "張老師 20 nr", "陳老師 20 nr", 
            "林老師 20 nr", "黃老師 20 nr", "劉老師 20 nr",
            
            # Roles
            "同學 15 n", "學生 15 n", "老師 20 n", "教師 15 n",
            "先生 10 n", "女士 10 n", "朋友 10 n",
            
            # Actions
            "學習 20 v", "讀書 15 v", "上課 15 v", "下課 10 v",
            "討論 10 v", "思考 10 v", "回答 15 v", "提問 10 v"
        ]
        
        # Create temporary dictionary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            for word in custom_words:
                f.write(word + '\n')
            temp_dict_path = f.name
        
        # Load custom dictionary
        jieba.load_userdict(temp_dict_path)
        
        # Clean up
        os.unlink(temp_dict_path)
        
        print("[INIT] Custom dictionary loaded")
    
    def build_entity_patterns(self):
        """Build comprehensive entity recognition patterns"""
        patterns = {
            'person_name': [
                # Direct name patterns
                r'([一-龥]{2,4})(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                r'小([一-龥]{1,2})(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                r'([一-龥]{1,3})老師(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                
                # Possessive patterns
                r'([一-龥]{2,4})的(?=朋友|同學|老師|學生|家人|父母)',
                
                # Interaction patterns  
                r'和([一-龥]{2,4})(?=一起|共同|合作)',
                r'跟([一-龥]{2,4})(?=一起|共同|合作)',
                
                # Observation patterns
                r'看到([一-龥]{2,4})',
                r'遇到([一-龥]{2,4})',
                r'找到([一-龥]{2,4})',
            ],
            
            'role_title': [
                r'([一-龥]{1,3})(老師|教師|先生|女士|同學|朋友)',
                r'(老師|教師|先生|女士|同學|朋友)([一-龥]{1,3})',
            ],
            
            'action_verb': [
                r'(說|問|回答|講|談|討論|思考|想|做|去|來|看|聽|學習|讀書|寫|畫)',
            ],
            
            'attribute_adj': [
                r'(聰明|努力|認真|友善|開朗|安靜|活潑|善良|勇敢|溫和)',
            ]
        }
        
        return patterns
    
    def stage1_sentence_segmentation(self, text):
        """Stage 1: Advanced sentence segmentation with POS tagging"""
        print("[STAGE1] Starting sentence segmentation and POS tagging")
        
        # Step 1: Split into sentences
        sentences = self.split_sentences(text)
        print(f"[STAGE1] Split into {len(sentences)} sentences")
        
        # Step 2: Segment each sentence and add POS tags
        segmented_sentences = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Jieba segmentation with POS tagging
                words_pos = list(pseg.cut(sentence))
                
                # Categorize words by POS
                categorized_words = self.categorize_words_by_pos(words_pos)
                
                sentence_data = {
                    'id': i,
                    'text': sentence,
                    'words_pos': [(word, pos) for word, pos in words_pos],
                    'categorized': categorized_words,
                    'length': len(sentence),
                    'word_count': len(words_pos)
                }
                
                segmented_sentences.append(sentence_data)
                print(f"[STAGE1] Sentence {i}: {len(words_pos)} words, categories: {list(categorized_words.keys())}")
        
        return segmented_sentences
    
    def split_sentences(self, text):
        """Advanced sentence splitting"""
        # Primary sentence endings
        sentences = re.split(r'[。！？；]', text)
        
        # Handle quotation marks and dialogue
        refined_sentences = []
        for sentence in sentences:
            if sentence.strip():
                # Split on colon followed by quotation marks (dialogue)
                sub_sentences = re.split(r'[:：](?=["「『])', sentence)
                refined_sentences.extend([s.strip() for s in sub_sentences if s.strip()])
        
        return refined_sentences
    
    def categorize_words_by_pos(self, words_pos):
        """Categorize words by their POS tags"""
        categorized = defaultdict(list)
        
        for word, pos in words_pos:
            # Find category for this POS tag
            category_found = False
            for category, pos_tags in self.pos_categories.items():
                if pos in pos_tags:
                    categorized[category].append((word, pos))
                    category_found = True
                    break
            
            if not category_found:
                categorized['other'].append((word, pos))
        
        return dict(categorized)
    
    def stage2_entity_extraction(self, segmented_sentences):
        """Stage 2: Entity recognition with cross-validation"""
        print("[STAGE2] Starting entity extraction and validation")
        
        # Collect all potential entities
        potential_entities = defaultdict(list)
        
        # Extract entities from each sentence
        for sentence_data in segmented_sentences:
            sentence_entities = self.extract_entities_from_sentence(sentence_data)
            
            for entity_type, entities in sentence_entities.items():
                potential_entities[entity_type].extend(entities)
        
        # Cross-validate and merge entities
        validated_entities = self.cross_validate_entities(potential_entities, segmented_sentences)
        
        return validated_entities
    
    def extract_entities_from_sentence(self, sentence_data):
        """Extract entities from a single sentence"""
        text = sentence_data['text']
        words_pos = sentence_data['words_pos']
        categorized = sentence_data['categorized']
        
        entities = defaultdict(list)
        
        # Method 1: Pattern-based extraction
        pattern_entities = self.extract_by_patterns(text)
        
        # Method 2: POS-based extraction
        pos_entities = self.extract_by_pos(words_pos, categorized)
        
        # Method 3: Context-based extraction
        context_entities = self.extract_by_context(text, words_pos)
        
        # Merge results
        for entity_type in ['person_name', 'role_title', 'action_verb', 'attribute_adj']:
            entities[entity_type].extend(pattern_entities.get(entity_type, []))
            entities[entity_type].extend(pos_entities.get(entity_type, []))
            entities[entity_type].extend(context_entities.get(entity_type, []))
        
        return entities
    
    def extract_by_patterns(self, text):
        """Extract entities using regex patterns"""
        entities = defaultdict(list)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        # Handle grouped matches
                        for group in match:
                            if group and self.validate_entity(group, entity_type):
                                entities[entity_type].append({
                                    'text': group,
                                    'method': 'pattern',
                                    'confidence': 0.8,
                                    'pattern': pattern
                                })
                    else:
                        if match and self.validate_entity(match, entity_type):
                            entities[entity_type].append({
                                'text': match,
                                'method': 'pattern',
                                'confidence': 0.8,
                                'pattern': pattern
                            })
        
        return entities
    
    def extract_by_pos(self, words_pos, categorized):
        """Extract entities based on POS tags"""
        entities = defaultdict(list)
        
        # Extract person names from nouns with specific POS tags
        person_pos_tags = ['nr', 'nrf']  # Person name tags
        for word, pos in words_pos:
            if pos in person_pos_tags and self.validate_entity(word, 'person_name'):
                entities['person_name'].append({
                    'text': word,
                    'method': 'pos',
                    'confidence': 0.7,
                    'pos_tag': pos
                })
        
        # Extract verbs as actions
        if 'verb' in categorized:
            for word, pos in categorized['verb']:
                if self.validate_entity(word, 'action_verb'):
                    entities['action_verb'].append({
                        'text': word,
                        'method': 'pos',
                        'confidence': 0.6,
                        'pos_tag': pos
                    })
        
        # Extract adjectives as attributes
        if 'adjective' in categorized:
            for word, pos in categorized['adjective']:
                if self.validate_entity(word, 'attribute_adj'):
                    entities['attribute_adj'].append({
                        'text': word,
                        'method': 'pos',
                        'confidence': 0.6,
                        'pos_tag': pos
                    })
        
        return entities
    
    def extract_by_context(self, text, words_pos):
        """Extract entities based on context analysis"""
        entities = defaultdict(list)
        
        # Analyze word sequences for names
        for i, (word, pos) in enumerate(words_pos):
            # Look for names before action verbs
            if i < len(words_pos) - 1:
                next_word, next_pos = words_pos[i + 1]
                if next_pos in ['v', 'vd', 'vn'] and self.validate_entity(word, 'person_name'):
                    entities['person_name'].append({
                        'text': word,
                        'method': 'context',
                        'confidence': 0.75,
                        'context': f'before_verb_{next_word}'
                    })
            
            # Look for names in interaction contexts
            if word in ['和', '跟'] and i < len(words_pos) - 1:
                next_word, next_pos = words_pos[i + 1]
                if self.validate_entity(next_word, 'person_name'):
                    entities['person_name'].append({
                        'text': next_word,
                        'method': 'context',
                        'confidence': 0.8,
                        'context': f'after_conjunction_{word}'
                    })
        
        return entities
    
    def validate_entity(self, text, entity_type):
        """Validate if text is a valid entity of given type"""
        if not text or len(text) < 1:
            return False
        
        # Common invalid words
        invalid_words = {
            '這個', '那個', '什麼', '怎麼', '為什麼', '哪個',
            '他們', '我們', '大家', '所有', '每個', '一個',
            '今天', '明天', '昨天', '現在', '以前', '將來',
            '很好', '非常', '特別', '應該', '可以', '不能',
            '沒有', '還有', '已經', '正在', '開始', '結束'
        }
        
        if text in invalid_words:
            return False
        
        # Entity-specific validation
        if entity_type == 'person_name':
            # Must be 2-4 Chinese characters
            if not re.match(r'^[一-龥]{2,4}$', text):
                return False
            
            # Check for common name patterns
            if text.startswith('小') or text.endswith('老師'):
                return True
            
            # Check if contains common name characters
            common_name_chars = set('明華強志偉勇軍磊濤亮建國文武東南西北中正美麗紅玲芳燕娟霞雲鳳萍月英秀珍莉敏靜婷慧')
            if any(char in common_name_chars for char in text):
                return True
        
        elif entity_type == 'action_verb':
            # Must be valid action verb
            action_verbs = {'說', '問', '回答', '講', '談', '討論', '思考', '想', '做', '去', '來', '看', '聽', '學習', '讀書', '寫', '畫', '玩', '跑', '走'}
            return text in action_verbs
        
        elif entity_type == 'attribute_adj':
            # Must be valid attribute adjective
            attributes = {'聰明', '努力', '認真', '友善', '開朗', '安靜', '活潑', '善良', '勇敢', '溫和', '高興', '滿意', '開心', '快樂'}
            return text in attributes
        
        return True
    
    def cross_validate_entities(self, potential_entities, segmented_sentences):
        """Cross-validate entities using multiple methods"""
        print("[STAGE2] Cross-validating entities")
        
        validated_entities = {}
        
        for entity_type, entities in potential_entities.items():
            if not entities:
                continue
            
            # Group entities by text
            entity_groups = defaultdict(list)
            for entity in entities:
                entity_groups[entity['text']].append(entity)
            
            # Validate each entity group
            valid_entities = []
            for text, entity_list in entity_groups.items():
                # Calculate consensus confidence
                methods = [e['method'] for e in entity_list]
                avg_confidence = sum(e['confidence'] for e in entity_list) / len(entity_list)
                
                # Boost confidence for multiple methods
                if len(set(methods)) > 1:
                    avg_confidence += 0.1
                
                # Context validation
                context_score = self.validate_entity_context(text, entity_type, segmented_sentences)
                final_confidence = (avg_confidence + context_score) / 2
                
                if final_confidence >= 0.5:  # Threshold for acceptance
                    valid_entities.append({
                        'text': text,
                        'confidence': final_confidence,
                        'methods': list(set(methods)),
                        'frequency': len(entity_list),
                        'context_score': context_score
                    })
            
            # Sort by confidence
            valid_entities.sort(key=lambda x: x['confidence'], reverse=True)
            validated_entities[entity_type] = valid_entities
            
            print(f"[STAGE2] {entity_type}: {len(valid_entities)} validated entities")
        
        return validated_entities
    
    def validate_entity_context(self, entity_text, entity_type, segmented_sentences):
        """Validate entity based on context across all sentences"""
        context_score = 0.0
        total_text = ' '.join(s['text'] for s in segmented_sentences)
        
        if entity_type == 'person_name':
            # Check for typical person behaviors
            person_actions = ['說', '問', '回答', '去', '來', '做', '看', '想', '學習']
            for action in person_actions:
                if f"{entity_text}{action}" in total_text:
                    context_score += 0.1
            
            # Check for interactions
            interaction_patterns = [f"{entity_text}和", f"和{entity_text}", f"{entity_text}跟", f"跟{entity_text}"]
            for pattern in interaction_patterns:
                if pattern in total_text:
                    context_score += 0.15
            
            # Check for descriptions
            description_patterns = [f"{entity_text}是", f"{entity_text}很", f"{entity_text}的"]
            for pattern in description_patterns:
                if pattern in total_text:
                    context_score += 0.1
        
        return min(context_score, 1.0)
    
    def convert_to_character_format(self, validated_entities):
        """Convert validated entities to character format"""
        characters = []
        
        person_names = validated_entities.get('person_name', [])
        actions = validated_entities.get('action_verb', [])
        attributes = validated_entities.get('attribute_adj', [])
        
        for i, person in enumerate(person_names):
            name = person['text']
            confidence = person['confidence']
            
            # Classify character type
            if '老師' in name:
                char_type = '教師'
            elif name.startswith('小'):
                char_type = '學生'
            else:
                char_type = '人物'
            
            # Find related actions
            related_actions = []
            for action in actions:
                # This is simplified - in reality you'd check context
                if len(related_actions) < 3:  # Limit to 3 actions
                    related_actions.append({
                        'category': '行為',
                        'count': 1,
                        'actions': [action['text']]
                    })
            
            character = {
                'id': f'char_{i}',
                'name': name,
                'description': f'{name} - {char_type}',
                'confidence': confidence,
                'methods': person['methods'],
                'frequency': person['frequency'],
                'context_score': person['context_score'],
                'behaviors': related_actions,
                'events': [],
                'attributes': [{'type': attr['text']} for attr in attributes[:3]]  # Top 3 attributes
            }
            
            characters.append(character)
        
        return characters


# Initialize the processor
nlp_processor = TwoStageNLPProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Two-stage text analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        print(f"[API] Starting two-stage analysis for text: {text[:50]}...")
        
        # Stage 1: Sentence segmentation with POS tagging
        segmented_sentences = nlp_processor.stage1_sentence_segmentation(text)
        
        # Stage 2: Entity extraction with cross-validation
        validated_entities = nlp_processor.stage2_entity_extraction(segmented_sentences)
        
        # Convert to character format
        characters = nlp_processor.convert_to_character_format(validated_entities)
        
        print(f"[API] Final result: {len(characters)} characters identified")
        for char in characters:
            print(f"[API] - {char['name']} (confidence: {char['confidence']:.2f}, methods: {char['methods']})")
        
        # Simple relationships
        relationships = []
        if len(characters) >= 2:
            for i, char1 in enumerate(characters):
                for char2 in characters[i+1:]:
                    if f"{char1['name']}和{char2['name']}" in text or f"{char2['name']}和{char1['name']}" in text:
                        relationships.append({
                            'source': char1['name'],
                            'target': char2['name'],
                            'type': 'interaction'
                        })
        
        response = {
            "success": True,
            "text": text,
            "characters": characters,
            "relationships": relationships,
            "analysis_method": "two_stage_nlp",
            "stage1_sentences": len(segmented_sentences),
            "validated_entities": {k: len(v) for k, v in validated_entities.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Two-stage analysis failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    print("Starting Two-Stage Character Analysis System...")
    print("Stage 1: Sentence Segmentation + POS Tagging")
    print("Stage 2: Entity Recognition + Cross-Validation")
    print("Access at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)