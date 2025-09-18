"""
Chinese Dictionary Support for Enhanced NLP Accuracy
Provides dictionary-based fallback and semantic reference
"""

import json
import pickle
import sqlite3
import requests
import logging
from typing import List, Dict, Optional, Tuple, Set
import os
import re
from concurrent.futures import ThreadPoolExecutor
import jieba
import jieba.posseg as pseg

logger = logging.getLogger(__name__)

class ChineseDictionary:
    """Chinese dictionary with multiple data sources"""
    
    def __init__(self, cache_dir: str = "./dict_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize database
        self.db_path = os.path.join(cache_dir, "chinese_dict.db")
        self.init_database()
        
        # Load dictionaries
        self.dictionaries = {}
        self.load_dictionaries()
        
        # Initialize jieba with custom dictionary
        self.setup_jieba()
        
        logger.info("Chinese Dictionary initialized")
    
    def init_database(self):
        """Initialize SQLite database for dictionary data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Words table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY,
                word TEXT UNIQUE,
                pinyin TEXT,
                definition TEXT,
                pos_tags TEXT,
                frequency INTEGER DEFAULT 0,
                source TEXT,
                is_name BOOLEAN DEFAULT 0,
                is_place BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Phrases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS phrases (
                id INTEGER PRIMARY KEY,
                phrase TEXT UNIQUE,
                meaning TEXT,
                usage_example TEXT,
                frequency INTEGER DEFAULT 0,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Character names table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS character_names (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                gender TEXT,
                meaning TEXT,
                popularity INTEGER DEFAULT 0,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_words_word ON words(word)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_phrases_phrase ON phrases(phrase)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_names_name ON character_names(name)')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized")
    
    def setup_jieba(self):
        """Setup jieba with custom dictionaries"""
        
        # Add custom words to jieba
        custom_dict_path = os.path.join(self.cache_dir, "custom_dict.txt")
        
        if not os.path.exists(custom_dict_path):
            self.create_custom_jieba_dict(custom_dict_path)
        
        jieba.load_userdict(custom_dict_path)
        logger.info("Jieba setup with custom dictionary")
    
    def create_custom_jieba_dict(self, dict_path: str):
        """Create custom dictionary for jieba"""
        
        custom_words = [
            # Common names
            "小明 10 nr",
            "小華 10 nr", 
            "小美 10 nr",
            "小強 10 nr",
            "小紅 10 nr",
            "小志 10 nr",
            "小文 10 nr",
            "小玲 10 nr",
            
            # Teacher titles
            "王老師 20 nr",
            "李老師 20 nr",
            "張老師 20 nr",
            "陳老師 20 nr",
            "林老師 20 nr",
            
            # Roles and titles
            "同學 15 n",
            "學生 15 n",
            "老師 20 n",
            "教師 15 n",
            "先生 10 n",
            "女士 10 n",
            
            # Actions
            "學習 20 v",
            "讀書 15 v",
            "上課 15 v",
            "討論 10 v",
            "思考 10 v",
        ]
        
        with open(dict_path, 'w', encoding='utf-8') as f:
            for word_info in custom_words:
                f.write(word_info + '\n')
        
        logger.info(f"Custom jieba dictionary created: {dict_path}")
    
    def load_dictionaries(self):
        """Load various Chinese dictionaries"""
        
        # Load built-in dictionaries
        self.load_common_names()
        self.load_common_words()
        self.load_sentence_patterns()
        
        # Try to load online dictionaries (with fallback)
        try:
            self.load_online_dictionary()
        except Exception as e:
            logger.warning(f"Failed to load online dictionary: {e}")
    
    def load_common_names(self):
        """Load common Chinese names"""
        
        names_data = {
            'surnames': [
                '王', '李', '張', '劉', '陳', '楊', '趙', '黃', '周', '吳',
                '徐', '孫', '胡', '朱', '高', '林', '何', '郭', '馬', '羅',
                '梁', '宋', '鄭', '謝', '韓', '唐', '馮', '於', '董', '蕭'
            ],
            'given_names_male': [
                '明', '華', '強', '志', '偉', '勇', '軍', '磊', '濤', '亮',
                '建', '國', '文', '武', '東', '南', '西', '北', '中', '正'
            ],
            'given_names_female': [
                '美', '麗', '紅', '玲', '芳', '燕', '娟', '霞', '雲', '鳳',
                '萍', '月', '英', '秀', '珍', '莉', '敏', '靜', '婷', '慧'
            ],
            'prefixes': ['小', '老', '大', '阿'],
            'titles': ['老師', '先生', '女士', '小姐', '同學', '朋友']
        }
        
        self.dictionaries['names'] = names_data
        
        # Store in database
        self.store_names_in_db(names_data)
    
    def store_names_in_db(self, names_data: Dict):
        """Store names in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store surnames and given names
        for surname in names_data['surnames']:
            cursor.execute('''
                INSERT OR IGNORE INTO character_names (name, gender, meaning, popularity, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (surname, 'unisex', 'surname', 100, 'builtin'))
        
        for name in names_data['given_names_male']:
            cursor.execute('''
                INSERT OR IGNORE INTO character_names (name, gender, meaning, popularity, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, 'male', 'given_name', 80, 'builtin'))
        
        for name in names_data['given_names_female']:
            cursor.execute('''
                INSERT OR IGNORE INTO character_names (name, gender, meaning, popularity, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, 'female', 'given_name', 80, 'builtin'))
        
        conn.commit()
        conn.close()
    
    def load_common_words(self):
        """Load common Chinese words with POS tags"""
        
        words_data = {
            'verbs': {
                '說': 'speak/say',
                '問': 'ask',
                '回答': 'answer',
                '去': 'go',
                '來': 'come',
                '做': 'do',
                '看': 'look/see',
                '想': 'think',
                '學習': 'study/learn',
                '讀書': 'read books',
                '寫': 'write',
                '聽': 'listen',
                '玩': 'play',
                '跑': 'run',
                '走': 'walk'
            },
            'nouns': {
                '學生': 'student',
                '老師': 'teacher',
                '同學': 'classmate',
                '朋友': 'friend',
                '家人': 'family',
                '父母': 'parents',
                '學校': 'school',
                '教室': 'classroom',
                '圖書館': 'library',
                '家': 'home',
                '書': 'book',
                '功課': 'homework'
            },
            'adjectives': {
                '好': 'good',
                '聰明': 'smart',
                '努力': 'hardworking',
                '認真': 'serious',
                '友善': 'friendly',
                '開朗': 'cheerful',
                '安靜': 'quiet',
                '活潑': 'lively',
                '善良': 'kind',
                '勇敢': 'brave'
            }
        }
        
        self.dictionaries['words'] = words_data
        
        # Store in database
        self.store_words_in_db(words_data)
    
    def store_words_in_db(self, words_data: Dict):
        """Store words in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pos_tag, words in words_data.items():
            for word, definition in words.items():
                cursor.execute('''
                    INSERT OR IGNORE INTO words (word, definition, pos_tags, frequency, source)
                    VALUES (?, ?, ?, ?, ?)
                ''', (word, definition, pos_tag[:-1], 100, 'builtin'))  # Remove 's' from pos_tag
        
        conn.commit()
        conn.close()
    
    def load_sentence_patterns(self):
        """Load common sentence patterns"""
        
        patterns = {
            'character_action': [
                r'([一-龥]{2,4})(說|問|回答|講|談)',
                r'([一-龥]{2,4})(去|來|走|跑|跳)',
                r'([一-龥]{2,4})(做|作|寫|畫|讀|學習)',
                r'([一-龥]{2,4})(看|見|注意|觀察)',
                r'([一-龥]{2,4})(想|思考|考慮|決定)'
            ],
            'character_description': [
                r'([一-龥]{2,4})是(一個|一位)?([學生|老師|同學|朋友])',
                r'([一-龥]{2,4})很([聰明|努力|認真|友善|開朗])',
                r'([一-龥]{2,4})的([朋友|同學|老師|學生])'
            ],
            'interaction': [
                r'([一-龥]{2,4})和([一-龥]{2,4})(一起|共同)([一-龥]+)',
                r'([一-龥]{2,4})跟([一-龥]{2,4})(說|談|討論)',
                r'([一-龥]{2,4})告訴([一-龥]{2,4})'
            ]
        }
        
        self.dictionaries['patterns'] = patterns
    
    def load_online_dictionary(self):
        """Load additional dictionary data from online sources"""
        
        # This would connect to online Chinese dictionary APIs
        # For now, we'll simulate with placeholder data
        
        online_cache = os.path.join(self.cache_dir, 'online_dict_cache.json')
        
        if os.path.exists(online_cache):
            with open(online_cache, 'r', encoding='utf-8') as f:
                online_data = json.load(f)
                self.dictionaries['online'] = online_data
        else:
            # Placeholder for online dictionary
            placeholder_data = {
                'extended_names': [],
                'idioms': [],
                'modern_words': []
            }
            
            with open(online_cache, 'w', encoding='utf-8') as f:
                json.dump(placeholder_data, f, ensure_ascii=False, indent=2)
            
            self.dictionaries['online'] = placeholder_data
    
    def is_valid_name(self, text: str) -> Tuple[bool, float]:
        """Check if text is a valid Chinese name"""
        
        if len(text) < 2 or len(text) > 4:
            return False, 0.0
        
        confidence = 0.0
        
        # Check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check exact match
        cursor.execute('SELECT popularity FROM character_names WHERE name = ?', (text,))
        result = cursor.fetchone()
        
        if result:
            confidence += 0.9
        else:
            # Check component analysis
            if len(text) >= 2:
                # Check if first character is common surname
                first_char = text[0]
                cursor.execute('SELECT popularity FROM character_names WHERE name = ? AND meaning = ?', 
                             (first_char, 'surname'))
                if cursor.fetchone():
                    confidence += 0.5
                
                # Check remaining characters as given names
                remaining = text[1:]
                for char in remaining:
                    cursor.execute('SELECT popularity FROM character_names WHERE name = ?', (char,))
                    if cursor.fetchone():
                        confidence += 0.3
        
        conn.close()
        
        # Pattern-based validation
        if text.startswith('小') and len(text) == 3:
            confidence += 0.4
        
        if text.endswith('老師'):
            confidence += 0.8
        
        # Character frequency analysis
        common_name_chars = '明華強志偉勇軍磊濤亮建國文武東南西北中正美麗紅玲芳燕娟霞雲鳳萍月英秀珍莉敏靜婷慧'
        
        for char in text:
            if char in common_name_chars:
                confidence += 0.1
        
        is_valid = confidence >= 0.4
        return is_valid, min(confidence, 1.0)
    
    def segment_with_dictionary(self, text: str) -> List[Dict]:
        """Segment text using dictionary knowledge"""
        
        # Use jieba for initial segmentation
        seg_result = list(pseg.cut(text))
        
        # Enhance with dictionary knowledge
        enhanced_segments = []
        
        for word, flag in seg_result:
            segment_info = {
                'word': word,
                'pos': flag,
                'is_name': self.is_potential_character_name(word, flag),
                'confidence': self.get_word_confidence(word, flag),
                'dictionary_info': self.get_dictionary_info(word)
            }
            
            enhanced_segments.append(segment_info)
        
        return enhanced_segments
    
    def is_potential_character_name(self, word: str, pos_tag: str) -> bool:
        """Check if word is potentially a character name"""
        
        # Check POS tag
        if pos_tag in ['nr', 'nrf', 'nrj']:  # Person name tags
            return True
        
        # Check dictionary
        is_valid, confidence = self.is_valid_name(word)
        return is_valid and confidence > 0.5
    
    def get_word_confidence(self, word: str, pos_tag: str) -> float:
        """Get confidence score for word segmentation"""
        
        # Check in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT frequency FROM words WHERE word = ?', (word,))
        result = cursor.fetchone()
        
        base_confidence = 0.6
        
        if result:
            frequency = result[0]
            base_confidence += min(frequency / 100, 0.3)
        
        # POS tag confidence adjustment
        high_confidence_pos = ['nr', 'nrf', 'nrj', 'n', 'v', 'a']
        if pos_tag in high_confidence_pos:
            base_confidence += 0.1
        
        conn.close()
        return min(base_confidence, 1.0)
    
    def get_dictionary_info(self, word: str) -> Dict:
        """Get comprehensive dictionary information for word"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        info = {'found': False}
        
        # Check words table
        cursor.execute('''
            SELECT definition, pos_tags, frequency, source 
            FROM words WHERE word = ?
        ''', (word,))
        
        result = cursor.fetchone()
        if result:
            info.update({
                'found': True,
                'definition': result[0],
                'pos_tags': result[1],
                'frequency': result[2],
                'source': result[3],
                'type': 'word'
            })
        else:
            # Check character names table
            cursor.execute('''
                SELECT gender, meaning, popularity, source 
                FROM character_names WHERE name = ?
            ''', (word,))
            
            result = cursor.fetchone()
            if result:
                info.update({
                    'found': True,
                    'gender': result[0],
                    'meaning': result[1],
                    'popularity': result[2],
                    'source': result[3],
                    'type': 'name'
                })
        
        conn.close()
        return info
    
    def enhance_character_extraction(self, text: str, initial_characters: List[Dict]) -> List[Dict]:
        """Enhance character extraction using dictionary knowledge"""
        
        enhanced_characters = []
        
        # Segment text with dictionary
        segments = self.segment_with_dictionary(text)
        
        # Find potential names in segments
        dictionary_names = []
        for segment in segments:
            if segment['is_name']:
                is_valid, confidence = self.is_valid_name(segment['word'])
                if is_valid:
                    dictionary_names.append({
                        'name': segment['word'],
                        'confidence': confidence,
                        'source': 'dictionary',
                        'pos_tag': segment['pos'],
                        'dictionary_info': segment['dictionary_info']
                    })
        
        # Merge with initial characters
        all_names = {}
        
        # Add initial characters
        for char in initial_characters:
            name = char['name']
            all_names[name] = char
        
        # Add/enhance with dictionary findings
        for dict_char in dictionary_names:
            name = dict_char['name']
            if name in all_names:
                # Enhance existing character
                existing = all_names[name]
                existing['confidence'] = max(existing.get('confidence', 0), dict_char['confidence'])
                existing['sources'] = existing.get('sources', []) + ['dictionary']
                existing['dictionary_enhanced'] = True
                existing['pos_tag'] = dict_char['pos_tag']
            else:
                # Add new character from dictionary
                new_char = {
                    'name': name,
                    'type': self.classify_character_from_dict(dict_char),
                    'confidence': dict_char['confidence'],
                    'source': 'dictionary',
                    'dictionary_enhanced': True,
                    'pos_tag': dict_char['pos_tag'],
                    'description': f"{name} - {self.get_character_description(dict_char)}"
                }
                all_names[name] = new_char
        
        enhanced_characters = list(all_names.values())
        
        # Sort by confidence
        enhanced_characters.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return enhanced_characters
    
    def classify_character_from_dict(self, dict_char: Dict) -> str:
        """Classify character type based on dictionary information"""
        
        name = dict_char['name']
        dict_info = dict_char.get('dictionary_info', {})
        
        if '老師' in name:
            return 'teacher'
        elif name.startswith('小') and len(name) <= 3:
            return 'student'
        elif dict_info.get('type') == 'name':
            gender = dict_info.get('gender', 'unisex')
            if gender == 'male':
                return 'male_person'
            elif gender == 'female':
                return 'female_person'
            else:
                return 'person'
        else:
            return 'person'
    
    def get_character_description(self, dict_char: Dict) -> str:
        """Generate character description from dictionary info"""
        
        dict_info = dict_char.get('dictionary_info', {})
        
        if dict_info.get('found'):
            if dict_info.get('type') == 'name':
                meaning = dict_info.get('meaning', '')
                gender = dict_info.get('gender', '')
                return f"{meaning} ({gender})"
            else:
                definition = dict_info.get('definition', '')
                return definition
        
        return dict_char['name']
    
    def validate_and_correct_segmentation(self, text: str, segments: List[str]) -> List[str]:
        """Validate and correct segmentation using dictionary"""
        
        corrected_segments = []
        
        for segment in segments:
            # Check if segment is in dictionary
            is_valid, confidence = self.is_valid_word(segment)
            
            if is_valid or confidence > 0.7:
                corrected_segments.append(segment)
            else:
                # Try to re-segment problematic segment
                sub_segments = self.re_segment_word(segment)
                corrected_segments.extend(sub_segments)
        
        return corrected_segments
    
    def is_valid_word(self, word: str) -> Tuple[bool, float]:
        """Check if word is valid according to dictionary"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check in words table
        cursor.execute('SELECT frequency FROM words WHERE word = ?', (word,))
        result = cursor.fetchone()
        
        if result:
            confidence = min(result[0] / 100, 1.0)
            conn.close()
            return True, confidence
        
        # Check in names table
        cursor.execute('SELECT popularity FROM character_names WHERE name = ?', (word,))
        result = cursor.fetchone()
        
        if result:
            confidence = min(result[0] / 100, 1.0)
            conn.close()
            return True, confidence
        
        conn.close()
        
        # Use jieba to check if it's a known word
        if word in jieba.dt.FREQ:
            confidence = min(jieba.dt.FREQ[word] / 1000, 1.0)
            return True, confidence
        
        return False, 0.0
    
    def re_segment_word(self, word: str) -> List[str]:
        """Re-segment a word that might be incorrectly segmented"""
        
        if len(word) <= 2:
            return [word]
        
        # Try different segmentation strategies
        best_segmentation = [word]
        best_score = 0.0
        
        # Strategy 1: Split into individual characters
        char_segments = list(word)
        char_score = sum(self.is_valid_word(char)[1] for char in char_segments)
        
        if char_score > best_score:
            best_score = char_score
            best_segmentation = char_segments
        
        # Strategy 2: Try prefix-suffix combinations
        for i in range(1, len(word)):
            prefix = word[:i]
            suffix = word[i:]
            
            prefix_valid, prefix_conf = self.is_valid_word(prefix)
            suffix_valid, suffix_conf = self.is_valid_word(suffix)
            
            if prefix_valid and suffix_valid:
                combined_score = prefix_conf + suffix_conf
                if combined_score > best_score:
                    best_score = combined_score
                    best_segmentation = [prefix, suffix]
        
        return best_segmentation
    
    def add_word_to_dictionary(self, word: str, definition: str, pos_tag: str, source: str = "user"):
        """Add new word to dictionary"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO words (word, definition, pos_tags, frequency, source)
            VALUES (?, ?, ?, ?, ?)
        ''', (word, definition, pos_tag, 1, source))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added word to dictionary: {word}")
    
    def add_name_to_dictionary(self, name: str, gender: str, meaning: str, source: str = "user"):
        """Add new name to dictionary"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO character_names (name, gender, meaning, popularity, source)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, gender, meaning, 1, source))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added name to dictionary: {name}")
    
    def get_dictionary_stats(self) -> Dict:
        """Get dictionary statistics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count words
        cursor.execute('SELECT COUNT(*) FROM words')
        word_count = cursor.fetchone()[0]
        
        # Count names
        cursor.execute('SELECT COUNT(*) FROM character_names')
        name_count = cursor.fetchone()[0]
        
        # Count phrases
        cursor.execute('SELECT COUNT(*) FROM phrases')
        phrase_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'words': word_count,
            'names': name_count,
            'phrases': phrase_count,
            'patterns': len(self.dictionaries.get('patterns', {})),
            'cache_size': len(os.listdir(self.cache_dir)) if os.path.exists(self.cache_dir) else 0
        }


# Dictionary-enhanced NLP pipeline
class DictionaryEnhancedNLP:
    """NLP pipeline enhanced with dictionary support"""
    
    def __init__(self, dictionary: ChineseDictionary):
        self.dictionary = dictionary
        
    def extract_characters_with_dictionary(self, text: str, initial_extraction_func) -> List[Dict]:
        """Extract characters using dictionary enhancement"""
        
        # Get initial extraction from other models
        initial_characters = initial_extraction_func(text)
        
        # Enhance with dictionary
        enhanced_characters = self.dictionary.enhance_character_extraction(text, initial_characters)
        
        # Additional validation
        validated_characters = self.validate_characters(enhanced_characters, text)
        
        return validated_characters
    
    def validate_characters(self, characters: List[Dict], text: str) -> List[Dict]:
        """Validate characters using dictionary knowledge"""
        
        validated = []
        
        for char in characters:
            name = char['name']
            
            # Dictionary validation
            is_valid, dict_confidence = self.dictionary.is_valid_name(name)
            
            # Context validation
            context_score = self.validate_character_context(name, text)
            
            # Combined confidence
            original_confidence = char.get('confidence', 0.5)
            enhanced_confidence = (original_confidence + dict_confidence + context_score) / 3
            
            if enhanced_confidence >= 0.4:  # Threshold for inclusion
                char['confidence'] = enhanced_confidence
                char['dictionary_validated'] = True
                char['context_score'] = context_score
                validated.append(char)
        
        return validated
    
    def validate_character_context(self, character: str, text: str) -> float:
        """Validate character based on context"""
        
        score = 0.0
        
        # Check for action patterns
        action_patterns = [
            f"{character}說",
            f"{character}問",
            f"{character}去",
            f"{character}來",
            f"{character}做",
            f"{character}看"
        ]
        
        for pattern in action_patterns:
            if pattern in text:
                score += 0.2
        
        # Check for descriptive patterns
        desc_patterns = [
            f"{character}是",
            f"{character}很",
            f"{character}的"
        ]
        
        for pattern in desc_patterns:
            if pattern in text:
                score += 0.1
        
        # Check for interaction patterns
        interaction_patterns = [
            f"{character}和",
            f"和{character}",
            f"{character}跟",
            f"跟{character}"
        ]
        
        for pattern in interaction_patterns:
            if pattern in text:
                score += 0.15
        
        return min(score, 1.0)


if __name__ == "__main__":
    # Test the dictionary system
    logging.basicConfig(level=logging.INFO)
    
    # Initialize dictionary
    dictionary = ChineseDictionary()
    
    # Test name validation
    test_names = ["小明", "王老師", "小華", "xyz", "李小美"]
    
    for name in test_names:
        is_valid, confidence = dictionary.is_valid_name(name)
        print(f"Name: {name}, Valid: {is_valid}, Confidence: {confidence:.2f}")
    
    # Test segmentation
    test_text = "小明是一個學生，他和王老師一起學習。"
    segments = dictionary.segment_with_dictionary(test_text)
    
    print("\nSegmentation result:")
    for segment in segments:
        print(f"Word: {segment['word']}, POS: {segment['pos']}, Is Name: {segment['is_name']}")
    
    # Test enhancement
    initial_chars = [{'name': '小明', 'confidence': 0.5}]
    enhanced = dictionary.enhance_character_extraction(test_text, initial_chars)
    
    print("\nEnhanced characters:")
    for char in enhanced:
        print(f"Name: {char['name']}, Confidence: {char['confidence']:.2f}")
    
    # Dictionary stats
    stats = dictionary.get_dictionary_stats()
    print(f"\nDictionary stats: {stats}")