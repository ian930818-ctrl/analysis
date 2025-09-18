"""
Enhanced Two-Stage Character Analysis with Diversified Corpus and Advanced NER
Features:
- Diversified Chinese name and title corpus
- Advanced name recognition patterns
- Rule-based dictionary assistance
- Multi-pattern validation
- Title-name simultaneous recognition
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

class EnhancedNLPProcessor:
    """Enhanced NLP processor with diversified corpus and advanced name recognition"""
    
    def __init__(self):
        # Initialize enhanced dictionary
        self.setup_enhanced_dictionary()
        
        # Advanced POS tag mappings
        self.pos_categories = {
            'person_noun': ['nr', 'nrt'],  # Person names
            'common_noun': ['n', 'nz', 'nt', 'ns', 'ni', 'nw'],
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
        
        # Enhanced entity patterns
        self.entity_patterns = self.build_enhanced_patterns()
        
        # Chinese name dictionary for validation
        self.name_dictionary = self.build_name_dictionary()
        
        # Title dictionary
        self.title_dictionary = self.build_title_dictionary()
        
        print("[INIT] Enhanced NLP processor with diversified corpus initialized")
    
    def setup_enhanced_dictionary(self):
        """Setup enhanced dictionary with diversified Chinese names and titles"""
        # Comprehensive Chinese names corpus
        enhanced_words = [
            # Common Chinese surnames + given names
            "王小明 50 nr", "李小華 50 nr", "張小美 50 nr", "陳小強 50 nr", "林小紅 50 nr",
            "黃小志 50 nr", "劉小文 50 nr", "楊小玲 50 nr", "周小芳 50 nr", "吳小傑 50 nr",
            "徐小偉 40 nr", "孫小勇 40 nr", "朱小軍 40 nr", "高小磊 40 nr", "郭小濤 40 nr",
            "何小亮 40 nr", "羅小建 40 nr", "宋小國 40 nr", "梁小東 40 nr", "韓小南 40 nr",
            
            # Professional titles with names
            "王老師 60 nr", "李老師 60 nr", "張老師 60 nr", "陳老師 60 nr", "林老師 60 nr",
            "黃老師 60 nr", "劉老師 60 nr", "楊老師 60 nr", "周老師 60 nr", "吳老師 60 nr",
            "王教授 50 nr", "李教授 50 nr", "張教授 50 nr", "陳教授 50 nr", "林教授 50 nr",
            "王主任 40 nr", "李主任 40 nr", "張主任 40 nr", "陳主任 40 nr", "林主任 40 nr",
            "王校長 50 nr", "李校長 50 nr", "張校長 50 nr", "陳校長 50 nr", "林校長 50 nr",
            "王醫生 40 nr", "李醫生 40 nr", "張醫生 40 nr", "陳醫生 40 nr", "林醫生 40 nr",
            
            # Student names with nicknames
            "小明同學 40 nr", "小華同學 40 nr", "小美同學 40 nr", "小強同學 40 nr", "小紅同學 40 nr",
            "阿明 30 nr", "阿華 30 nr", "阿美 30 nr", "阿強 30 nr", "阿紅 30 nr",
            "小王 30 nr", "小李 30 nr", "小張 30 nr", "小陳 30 nr", "小林 30 nr",
            
            # Professional roles
            "老師 30 n", "教師 25 n", "教授 25 n", "主任 20 n", "校長 25 n",
            "醫生 20 n", "護士 20 n", "警察 20 n", "司機 15 n", "工程師 20 n",
            "學生 30 n", "同學 25 n", "朋友 20 n", "同事 15 n", "同伴 15 n",
            
            # Family relations
            "爸爸 25 n", "媽媽 25 n", "哥哥 20 n", "姐姐 20 n", "弟弟 20 n", "妹妹 20 n",
            "爺爺 20 n", "奶奶 20 n", "外公 15 n", "外婆 15 n", "叔叔 15 n", "阿姨 15 n",
            
            # Action verbs
            "學習 25 v", "讀書 20 v", "上課 20 v", "下課 15 v", "教學 20 v", "教導 15 v",
            "討論 15 v", "思考 15 v", "回答 20 v", "提問 15 v", "解釋 15 v", "說明 15 v",
            "寫作 15 v", "閱讀 15 v", "計算 15 v", "實驗 15 v", "研究 15 v", "分析 15 v",
            
            # Attributes and characteristics
            "聰明 15 a", "努力 15 a", "認真 15 a", "友善 15 a", "開朗 15 a", "安靜 15 a",
            "活潑 15 a", "善良 15 a", "勇敢 15 a", "溫和 15 a", "勤奮 15 a", "細心 15 a"
        ]
        
        # Create temporary dictionary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            for word in enhanced_words:
                f.write(word + '\n')
            temp_dict_path = f.name
        
        # Load custom dictionary
        jieba.load_userdict(temp_dict_path)
        
        # Clean up
        os.unlink(temp_dict_path)
        
        print("[INIT] Enhanced dictionary with diversified corpus loaded")
    
    def build_enhanced_patterns(self):
        """Build comprehensive entity recognition patterns with enhanced coverage"""
        patterns = {
            'person_name': [
                # Direct name patterns with enhanced context
                r'([一-龥]{2,4})(?=說|問|回答|講|談|想|做|去|來|看|聽|學習|教|寫|讀)',
                r'([一-龥]{1,2}[小阿])(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                r'(小[一-龥]{1,3})(?=說|問|回答|講|談|想|做|去|來|看|聽|很|是|在|有)',
                r'(阿[一-龥]{1,2})(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                
                # Professional title patterns
                r'([一-龥]{1,3})(老師|教授|主任|校長|醫生|護士|警察|工程師)(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                r'([一-龥]{1,3})老師(?=說|問|回答|講|談|想|做|去|來|看|聽|很|是|在|有)',
                
                # Student name patterns
                r'([一-龥]{2,4})同學(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                r'([一-龥]{1,3})學生(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                
                # Possessive patterns
                r'([一-龥]{2,4})的(?=朋友|同學|老師|學生|家人|父母|同事|同伴)',
                
                # Interaction patterns  
                r'和([一-龥]{2,4})(?=一起|共同|合作|討論|學習|工作)',
                r'跟([一-龥]{2,4})(?=一起|共同|合作|討論|學習|工作)',
                r'與([一-龥]{2,4})(?=一起|共同|合作|討論|學習|工作)',
                
                # Observation patterns
                r'看到([一-龥]{2,4})', r'遇到([一-龥]{2,4})', r'找到([一-龥]{2,4})',
                r'認識([一-龥]{2,4})', r'聽到([一-龥]{2,4})', r'見到([一-龥]{2,4})',
                
                # Family relation patterns
                r'([一-龥]{1,3})(爸爸|媽媽|哥哥|姐姐|弟弟|妹妹|爺爺|奶奶|叔叔|阿姨)',
                
                # Address patterns
                r'([一-龥]{2,4})同學', r'([一-龥]{2,4})先生', r'([一-龥]{2,4})女士',
                r'([一-龥]{2,4})小姐', r'([一-龥]{2,4})朋友'
            ],
            
            'title_only': [
                r'(老師|教授|主任|校長|醫生|護士|警察|工程師|學生|同學)(?=說|問|回答|講|談|想|做|去|來|看|聽)',
                r'(爸爸|媽媽|哥哥|姐姐|弟弟|妹妹|爺爺|奶奶|叔叔|阿姨)(?=說|問|回答|講|談|想|做|去|來|看|聽)'
            ],
            
            'name_title_combined': [
                r'([一-龥]{1,3})(老師|教授|主任|校長|醫生|護士|警察|工程師)',
                r'([一-龥]{2,4})(同學|朋友|先生|女士|小姐)',
                r'([一-龥]{1,3})(爸爸|媽媽|哥哥|姐姐|弟弟|妹妹|爺爺|奶奶|叔叔|阿姨)'
            ],
            
            'action_verb': [
                r'(說|問|回答|講|談|討論|思考|想|做|去|來|看|聽|學習|讀書|寫|畫|教|解釋|說明|研究|分析)',
            ],
            
            'attribute_adj': [
                r'(聰明|努力|認真|友善|開朗|安靜|活潑|善良|勇敢|溫和|勤奮|細心|耐心|負責|專業|優秀)',
            ]
        }
        
        return patterns
    
    def build_name_dictionary(self):
        """Build comprehensive Chinese name dictionary for validation"""
        # Common Chinese surnames
        surnames = [
            '王', '李', '張', '劉', '陳', '楊', '黃', '趙', '周', '吳',
            '徐', '孫', '朱', '馬', '胡', '郭', '林', '何', '高', '梁',
            '鄭', '羅', '宋', '謝', '唐', '韓', '曹', '許', '鄧', '蕭'
        ]
        
        # Common given names
        given_names = [
            '小明', '小華', '小美', '小強', '小紅', '小志', '小文', '小玲', '小芳', '小傑',
            '小偉', '小勇', '小軍', '小磊', '小濤', '小亮', '小建', '小國', '小東', '小南',
            '明', '華', '美', '強', '紅', '志', '文', '玲', '芳', '傑', '偉', '勇',
            '軍', '磊', '濤', '亮', '建', '國', '東', '南', '西', '北', '中', '正',
            '秀', '珍', '莉', '敏', '靜', '婷', '慧', '英', '雲', '鳳', '萍', '月'
        ]
        
        # Nicknames and informal names
        nicknames = [
            '阿明', '阿華', '阿美', '阿強', '阿紅', '阿志', '阿文', '阿玲', '阿芳', '阿傑',
            '小王', '小李', '小張', '小劉', '小陳', '小楊', '小黃', '小趙', '小周', '小吳'
        ]
        
        return {
            'surnames': set(surnames),
            'given_names': set(given_names),
            'nicknames': set(nicknames),
            'full_names': set([s + g for s in surnames for g in given_names[:10]])  # Limited combinations
        }
    
    def build_title_dictionary(self):
        """Build comprehensive title dictionary"""
        return {
            'professional': {'老師', '教授', '主任', '校長', '醫生', '護士', '警察', '工程師', '司機'},
            'academic': {'學生', '同學', '研究生', '博士生', '碩士生'},
            'family': {'爸爸', '媽媽', '哥哥', '姐姐', '弟弟', '妹妹', '爺爺', '奶奶', '叔叔', '阿姨'},
            'social': {'朋友', '同事', '同伴', '夥伴', '室友'},
            'formal': {'先生', '女士', '小姐', '太太'}
        }
    
    def stage1_enhanced_segmentation(self, text):
        """Enhanced Stage 1: Advanced sentence segmentation with improved POS tagging"""
        print("[STAGE1] Starting enhanced sentence segmentation")
        
        # Step 1: Advanced sentence splitting
        sentences = self.enhanced_split_sentences(text)
        print(f"[STAGE1] Split into {len(sentences)} sentences")
        
        # Step 2: Enhanced segmentation with validation
        segmented_sentences = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Jieba segmentation with enhanced POS tagging
                words_pos = list(pseg.cut(sentence))
                
                # Filter and enhance POS results
                enhanced_words_pos = self.enhance_pos_results(words_pos)
                
                # Categorize words by enhanced POS
                categorized_words = self.categorize_enhanced_pos(enhanced_words_pos)
                
                # Extract potential names directly from POS
                pos_names = self.extract_names_from_pos(enhanced_words_pos)
                
                sentence_data = {
                    'id': i,
                    'text': sentence,
                    'words_pos': enhanced_words_pos,
                    'categorized': categorized_words,
                    'pos_names': pos_names,
                    'length': len(sentence),
                    'word_count': len(enhanced_words_pos)
                }
                
                segmented_sentences.append(sentence_data)
                print(f"[STAGE1] Sentence {i}: {len(enhanced_words_pos)} words, potential names: {len(pos_names)}")
        
        return segmented_sentences
    
    def enhanced_split_sentences(self, text):
        """Enhanced sentence splitting with better dialogue handling"""
        # Primary sentence endings
        sentences = re.split(r'[。！？；]', text)
        
        # Handle quotation marks and dialogue
        refined_sentences = []
        for sentence in sentences:
            if sentence.strip():
                # Split on colon followed by quotation marks (dialogue)
                sub_sentences = re.split(r'[:：](?=["「『])', sentence)
                # Split on comma in long sentences
                final_sentences = []
                for sub in sub_sentences:
                    if len(sub) > 20:  # Long sentence threshold
                        comma_split = re.split(r'[，,](?![「『"])', sub)
                        final_sentences.extend([s.strip() for s in comma_split if len(s.strip()) > 3])
                    else:
                        final_sentences.append(sub.strip())
                
                refined_sentences.extend([s for s in final_sentences if s])
        
        return refined_sentences
    
    def enhance_pos_results(self, words_pos):
        """Enhance POS tagging results with custom validation"""
        enhanced_results = []
        
        for word, pos in words_pos:
            # Check if word is in our enhanced dictionary
            if self.is_known_name(word):
                # Override POS tag for known names
                enhanced_results.append((word, 'nr'))
            elif self.is_known_title(word):
                # Override POS tag for known titles
                enhanced_results.append((word, 'n'))
            else:
                enhanced_results.append((word, pos))
        
        return enhanced_results
    
    def categorize_enhanced_pos(self, words_pos):
        """Enhanced categorization with name-specific handling"""
        categorized = defaultdict(list)
        
        for word, pos in words_pos:
            # Handle person names specifically
            if pos == 'nr' or self.is_known_name(word):
                categorized['person_name'].append((word, pos))
            else:
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
    
    def extract_names_from_pos(self, words_pos):
        """Extract potential names from POS tagging results"""
        potential_names = []
        
        for word, pos in words_pos:
            if pos == 'nr' or self.is_known_name(word):
                confidence = 0.8 if self.is_known_name(word) else 0.6
                potential_names.append({
                    'text': word,
                    'confidence': confidence,
                    'method': 'pos_tagging',
                    'pos_tag': pos
                })
        
        return potential_names
    
    def is_known_name(self, text):
        """Check if text is a known name in our dictionary"""
        if not text or len(text) < 2:
            return False
        
        # Check exact matches
        if text in self.name_dictionary['full_names'] or text in self.name_dictionary['nicknames']:
            return True
        
        # Check patterns
        if text.startswith('小') and len(text) <= 4:
            return True
        
        if text.startswith('阿') and len(text) <= 3:
            return True
        
        # Check surname + title combinations
        for title in self.title_dictionary['professional']:
            if text.endswith(title) and len(text) > len(title):
                surname_part = text[:-len(title)]
                if surname_part in self.name_dictionary['surnames']:
                    return True
        
        return False
    
    def is_known_title(self, text):
        """Check if text is a known title"""
        for category, titles in self.title_dictionary.items():
            if text in titles:
                return True
        return False
    
    def stage2_enhanced_extraction(self, segmented_sentences):
        """Enhanced Stage 2: Advanced entity extraction with rule-dictionary fusion"""
        print("[STAGE2] Starting enhanced entity extraction")
        
        # Collect all potential entities from multiple sources
        potential_entities = defaultdict(list)
        
        # Source 1: POS-based names (from stage 1)
        for sentence_data in segmented_sentences:
            for name_entity in sentence_data.get('pos_names', []):
                potential_entities['person_name'].append(name_entity)
        
        # Source 2: Pattern-based extraction with enhanced patterns
        for sentence_data in segmented_sentences:
            pattern_entities = self.extract_by_enhanced_patterns(sentence_data['text'])
            for entity_type, entities in pattern_entities.items():
                potential_entities[entity_type].extend(entities)
        
        # Source 3: Dictionary-assisted extraction
        for sentence_data in segmented_sentences:
            dict_entities = self.extract_by_dictionary_rules(sentence_data['text'], sentence_data['words_pos'])
            for entity_type, entities in dict_entities.items():
                potential_entities[entity_type].extend(entities)
        
        # Source 4: Context-based validation and enhancement
        context_enhanced = self.enhance_with_context(potential_entities, segmented_sentences)
        
        # Final validation and ranking
        validated_entities = self.advanced_validation(context_enhanced, segmented_sentences)
        
        return validated_entities
    
    def extract_by_enhanced_patterns(self, text):
        """Extract entities using enhanced regex patterns"""
        entities = defaultdict(list)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        # Handle grouped matches
                        for group in match:
                            if group and self.enhanced_validate_entity(group, entity_type):
                                entities[entity_type].append({
                                    'text': group,
                                    'confidence': 0.7,
                                    'method': 'enhanced_pattern',
                                    'pattern_type': entity_type
                                })
                    else:
                        if match and self.enhanced_validate_entity(match, entity_type):
                            entities[entity_type].append({
                                'text': match,
                                'confidence': 0.7,
                                'method': 'enhanced_pattern',
                                'pattern_type': entity_type
                            })
        
        return entities
    
    def extract_by_dictionary_rules(self, text, words_pos):
        """Extract entities using dictionary rules and validation"""
        entities = defaultdict(list)
        
        # Rule 1: Check each word against our dictionaries
        for word, pos in words_pos:
            if self.is_known_name(word):
                entities['person_name'].append({
                    'text': word,
                    'confidence': 0.85,
                    'method': 'dictionary_lookup',
                    'dict_source': 'name_dict'
                })
            
            if self.is_known_title(word):
                entities['title_only'].append({
                    'text': word,
                    'confidence': 0.8,
                    'method': 'dictionary_lookup',
                    'dict_source': 'title_dict'
                })
        
        # Rule 2: Look for name+title combinations
        for i, (word, pos) in enumerate(words_pos[:-1]):
            next_word, next_pos = words_pos[i + 1]
            combined = word + next_word
            
            # Check if it's a name+title combination
            if self.is_known_title(next_word) and len(word) <= 3:
                entities['name_title_combined'].append({
                    'text': combined,
                    'confidence': 0.9,
                    'method': 'dictionary_combination',
                    'name_part': word,
                    'title_part': next_word
                })
        
        # Rule 3: Contextual dictionary enhancement
        for entity_type in ['person_name', 'title_only']:
            for entity in entities[entity_type]:
                # Boost confidence if appears in typical contexts
                if self.appears_in_typical_context(entity['text'], text):
                    entity['confidence'] = min(0.95, entity['confidence'] + 0.1)
                    entity['context_boost'] = True
        
        return entities
    
    def appears_in_typical_context(self, entity_text, full_text):
        """Check if entity appears in typical name/title contexts"""
        typical_contexts = [
            f"{entity_text}說", f"{entity_text}問", f"{entity_text}回答",
            f"{entity_text}是", f"{entity_text}很", f"{entity_text}在",
            f"和{entity_text}", f"跟{entity_text}", f"與{entity_text}",
            f"{entity_text}老師", f"{entity_text}同學", f"{entity_text}朋友"
        ]
        
        return any(context in full_text for context in typical_contexts)
    
    def enhanced_validate_entity(self, text, entity_type):
        """Enhanced entity validation with comprehensive rules"""
        if not text or len(text.strip()) == 0:
            return False
        
        text = text.strip()
        
        # Length validation
        if len(text) < 1 or len(text) > 8:
            return False
        
        # Must contain Chinese characters
        if not re.search(r'[一-龥]', text):
            return False
        
        # Common exclusions
        invalid_names = {
            '他們', '我們', '大家', '這個', '那個', '一個', '所有', '每個',
            '什麼', '怎麼', '為什麼', '哪個', '今天', '明天', '昨天',
            '上課', '下課', '學習', '讀書', '很好', '非常', '特別',
            '應該', '可以', '不能', '沒有', '還有', '已經', '正在',
            '因為', '所以', '但是', '然後', '如果', '雖然', '雖',
            '而且', '或者', '還是', '不過', '只是', '就是', '也是'
        }
        
        if text in invalid_names:
            return False
        
        # Entity-specific enhanced validation
        if entity_type in ['person_name', 'name_title_combined']:
            # Check against our name dictionary
            if self.is_known_name(text):
                return True
            
            # Must be 2-4 Chinese characters for person names
            if not re.match(r'^[一-龥]{2,4}$', text):
                return False
            
            # Check for common name patterns
            if (text.startswith('小') or text.startswith('阿') or 
                any(title in text for title in self.title_dictionary['professional']) or
                text.endswith('老師') or text.endswith('同學')):
                return True
            
            # Check for surname patterns
            if text[0] in self.name_dictionary['surnames'] and len(text) >= 2:
                return True
            
        elif entity_type == 'title_only':
            # Must be valid title
            return self.is_known_title(text)
        
        elif entity_type == 'action_verb':
            # Must be valid action verb
            action_verbs = {
                '說', '問', '回答', '講', '談', '討論', '思考', '想', '做', '去', '來', '看', '聽',
                '學習', '讀書', '寫', '畫', '玩', '跑', '走', '教', '解釋', '說明', '研究', '分析'
            }
            return text in action_verbs
        
        elif entity_type == 'attribute_adj':
            # Must be valid attribute adjective
            attributes = {
                '聰明', '努力', '認真', '友善', '開朗', '安靜', '活潑', '善良', '勇敢', '溫和',
                '勤奮', '細心', '耐心', '負責', '專業', '優秀', '高興', '滿意', '開心', '快樂'
            }
            return text in attributes
        
        return True
    
    def enhance_with_context(self, potential_entities, segmented_sentences):
        """Enhance entities with contextual information"""
        print("[STAGE2] Enhancing entities with context")
        
        full_text = ' '.join(s['text'] for s in segmented_sentences)
        enhanced_entities = defaultdict(list)
        
        for entity_type, entities in potential_entities.items():
            for entity in entities:
                entity_text = entity['text']
                
                # Context analysis
                context_score = self.analyze_entity_context(entity_text, entity_type, full_text)
                entity['context_score'] = context_score
                
                # Cross-validation with other extraction methods
                cross_validation_score = self.cross_validate_entity(entity, potential_entities)
                entity['cross_validation'] = cross_validation_score
                
                # Final confidence adjustment
                original_confidence = entity['confidence']
                entity['confidence'] = min(0.98, 
                    original_confidence * 0.6 + 
                    context_score * 0.3 + 
                    cross_validation_score * 0.1
                )
                
                enhanced_entities[entity_type].append(entity)
        
        return enhanced_entities
    
    def analyze_entity_context(self, entity_text, entity_type, full_text):
        """Analyze entity context for validation"""
        context_score = 0.0
        
        if entity_type in ['person_name', 'name_title_combined']:
            # Check for typical person behaviors and contexts
            person_patterns = [
                f"{entity_text}說", f"{entity_text}問", f"{entity_text}回答", f"{entity_text}講",
                f"{entity_text}去", f"{entity_text}來", f"{entity_text}做", f"{entity_text}看",
                f"{entity_text}想", f"{entity_text}學習", f"{entity_text}是", f"{entity_text}很",
                f"和{entity_text}", f"跟{entity_text}", f"與{entity_text}"
            ]
            
            for pattern in person_patterns:
                if pattern in full_text:
                    context_score += 0.1
            
            # Boost for educational contexts
            if any(word in full_text for word in ['老師', '學生', '上課', '學習', '讀書']):
                context_score += 0.1
        
        return min(1.0, context_score)
    
    def cross_validate_entity(self, entity, all_entities):
        """Cross-validate entity against other extraction methods"""
        entity_text = entity['text']
        entity_method = entity['method']
        
        # Count how many different methods found this entity
        methods = set()
        for entity_type, entities in all_entities.items():
            for e in entities:
                if e['text'] == entity_text:
                    methods.add(e['method'])
        
        # Score based on method diversity
        if len(methods) >= 3:
            return 0.9
        elif len(methods) == 2:
            return 0.7
        else:
            return 0.5
    
    def advanced_validation(self, enhanced_entities, segmented_sentences):
        """Advanced validation with multi-criteria filtering"""
        print("[STAGE2] Performing advanced validation")
        
        validated_entities = {}
        
        for entity_type, entities in enhanced_entities.items():
            if not entities:
                continue
            
            # Group entities by text
            entity_groups = defaultdict(list)
            for entity in entities:
                entity_groups[entity['text']].append(entity)
            
            # Validate each entity group
            valid_entities = []
            for text, entity_list in entity_groups.items():
                # Calculate final scores
                methods = [e['method'] for e in entity_list]
                avg_confidence = sum(e['confidence'] for e in entity_list) / len(entity_list)
                max_confidence = max(e['confidence'] for e in entity_list)
                
                # Method diversity bonus
                method_diversity_bonus = len(set(methods)) * 0.05
                
                # Frequency bonus
                frequency_bonus = min(0.2, len(entity_list) * 0.05)
                
                # Final confidence calculation
                final_confidence = min(0.99, 
                    max_confidence * 0.7 + 
                    avg_confidence * 0.2 + 
                    method_diversity_bonus + 
                    frequency_bonus
                )
                
                # Threshold-based acceptance
                confidence_threshold = 0.5
                if entity_type == 'person_name':
                    confidence_threshold = 0.6  # Higher threshold for names
                elif entity_type == 'name_title_combined':
                    confidence_threshold = 0.7  # Highest threshold for combined
                
                if final_confidence >= confidence_threshold:
                    # Create comprehensive entity record
                    entity_record = {
                        'text': text,
                        'confidence': final_confidence,
                        'methods': list(set(methods)),
                        'frequency': len(entity_list),
                        'avg_confidence': avg_confidence,
                        'max_confidence': max_confidence,
                        'method_diversity': len(set(methods)),
                        'extraction_details': entity_list[:3]  # Keep top 3 extraction details
                    }
                    
                    # Add type-specific information
                    if entity_type == 'name_title_combined':
                        # Try to separate name and title parts
                        name_part, title_part = self.separate_name_title(text)
                        if name_part and title_part:
                            entity_record['name_part'] = name_part
                            entity_record['title_part'] = title_part
                    
                    valid_entities.append(entity_record)
            
            # Sort by confidence
            valid_entities.sort(key=lambda x: x['confidence'], reverse=True)
            validated_entities[entity_type] = valid_entities
            
            print(f"[STAGE2] {entity_type}: {len(valid_entities)} validated entities")
        
        return validated_entities
    
    def separate_name_title(self, combined_text):
        """Separate name and title from combined text"""
        for category, titles in self.title_dictionary.items():
            for title in titles:
                if combined_text.endswith(title):
                    name_part = combined_text[:-len(title)]
                    if len(name_part) >= 1:
                        return name_part, title
        return None, None
    
    def process_text(self, text):
        """Main processing pipeline"""
        print(f"[MAIN] Processing text: {text[:50]}...")
        
        # Stage 1: Enhanced segmentation
        segmented_sentences = self.stage1_enhanced_segmentation(text)
        
        # Stage 2: Enhanced extraction
        extracted_entities = self.stage2_enhanced_extraction(segmented_sentences)
        
        # Convert to character format for frontend
        characters = self.convert_to_character_format(extracted_entities)
        
        # Generate relationships
        relationships = self.generate_enhanced_relationships(characters, text)
        
        return characters, relationships
    
    def convert_to_character_format(self, extracted_entities):
        """Convert extracted entities to character format for frontend"""
        characters = []
        char_id = 0
        
        # Process different entity types
        all_persons = []
        
        # Add person names
        if 'person_name' in extracted_entities:
            all_persons.extend(extracted_entities['person_name'])
        
        # Add name+title combinations (prioritize these)
        if 'name_title_combined' in extracted_entities:
            all_persons.extend(extracted_entities['name_title_combined'])
        
        # Add title-only if no names found
        if not all_persons and 'title_only' in extracted_entities:
            all_persons.extend(extracted_entities['title_only'])
        
        # Remove duplicates while preserving order and highest confidence
        seen = {}
        unique_persons = []
        for person in all_persons:
            text = person['text']
            if text not in seen or person['confidence'] > seen[text]['confidence']:
                seen[text] = person
        
        unique_persons = list(seen.values())
        unique_persons.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Convert to character format
        for person in unique_persons:
            # Determine character type
            char_type = self.classify_enhanced_character_type(person['text'])
            
            # Extract behaviors
            behaviors = self.extract_enhanced_behaviors(person['text'], extracted_entities)
            
            character = {
                "id": f"char_{char_id}",
                "name": person['text'],
                "description": f"{person['text']} - {char_type} (信心度: {person['confidence']:.2f})",
                "importance": min(5, max(1, int(person['confidence'] * 5))),
                "frequency": person.get('frequency', 1),
                "confidence": person['confidence'],
                "methods": person.get('methods', []),
                "source": "enhanced_nlp",
                "behaviors": behaviors,
                "extraction_details": person.get('extraction_details', [])
            }
            
            characters.append(character)
            char_id += 1
        
        print(f"[MAIN] Created {len(characters)} character records")
        return characters
    
    def classify_enhanced_character_type(self, name):
        """Enhanced character type classification"""
        if any(title in name for title in self.title_dictionary['professional']):
            return '專業人士'
        elif any(title in name for title in self.title_dictionary['academic']):
            return '學術人員'
        elif any(title in name for title in self.title_dictionary['family']):
            return '家庭成員'
        elif name.startswith('小') and len(name) <= 4:
            return '學生'
        elif name.startswith('阿') and len(name) <= 3:
            return '朋友/同伴'
        elif name.endswith('老師') or name.endswith('教授'):
            return '教育工作者'
        elif '老師' in name or '教師' in name:
            return '教師'
        else:
            return '人物'
    
    def extract_enhanced_behaviors(self, character_name, extracted_entities):
        """Extract enhanced behaviors for character"""
        behaviors = []
        
        # Get actions if available
        if 'action_verb' in extracted_entities:
            actions = [e['text'] for e in extracted_entities['action_verb']]
            if actions:
                behaviors.append({
                    "category": "行動",
                    "count": len(actions),
                    "actions": actions[:5]  # Top 5 actions
                })
        
        # Get attributes if available
        if 'attribute_adj' in extracted_entities:
            attributes = [e['text'] for e in extracted_entities['attribute_adj']]
            if attributes:
                behaviors.append({
                    "category": "特質",
                    "count": len(attributes),
                    "actions": attributes[:3]  # Top 3 attributes
                })
        
        return behaviors
    
    def generate_enhanced_relationships(self, characters, text):
        """Generate enhanced relationships between characters"""
        relationships = []
        
        if len(characters) < 2:
            return relationships
        
        # Enhanced relationship patterns
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                name1, name2 = char1['name'], char2['name']
                
                # Check for various interaction patterns
                interaction_patterns = [
                    f"{name1}和{name2}", f"{name2}和{name1}",
                    f"{name1}跟{name2}", f"{name2}跟{name1}",
                    f"{name1}與{name2}", f"{name2}與{name1}",
                    f"{name1}、{name2}", f"{name2}、{name1}"
                ]
                
                relationship_found = False
                relationship_type = "互動"
                relationship_desc = f"{name1}和{name2}有互動"
                
                for pattern in interaction_patterns:
                    if pattern in text:
                        relationship_found = True
                        break
                
                # Check for specific relationship types
                if not relationship_found:
                    # Teacher-student relationships
                    if ('老師' in name1 and ('學生' in char2['description'] or name2.startswith('小'))) or \
                       ('老師' in name2 and ('學生' in char1['description'] or name1.startswith('小'))):
                        relationship_found = True
                        relationship_type = "師生關係"
                        relationship_desc = f"{name1}和{name2}是師生關係"
                    
                    # Peer relationships
                    elif (name1.startswith('小') and name2.startswith('小')) or \
                         ('同學' in name1 and '同學' in name2):
                        relationship_found = True
                        relationship_type = "同學關係"
                        relationship_desc = f"{name1}和{name2}是同學"
                
                if relationship_found:
                    relationships.append({
                        "id": f"rel_{len(relationships)}",
                        "source": name1,
                        "target": name2,
                        "type": relationship_type,
                        "description": relationship_desc
                    })
        
        return relationships

# Initialize the enhanced processor
enhanced_processor = EnhancedNLPProcessor()

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
        
        print(f"[DEBUG] Analyzing text length: {len(text)}")
        print(f"[DEBUG] Text preview: {text[:50]}...")
        
        # Process with enhanced NLP
        characters, relationships = enhanced_processor.process_text(text)
        
        print(f"[DEBUG] Found {len(characters)} characters with enhanced NLP")
        
        response = {
            "success": True,
            "text": text,
            "characters": characters,
            "relationships": relationships,
            "timestamp": datetime.now().isoformat(),
            "processing_method": "enhanced_nlp_with_diversified_corpus"
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Enhanced analysis failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    print("Starting Enhanced Character Analysis System with Diversified Corpus...")
    print("Features: Advanced NER, Dictionary-Rule Fusion, Name-Title Recognition")
    print("Access at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)