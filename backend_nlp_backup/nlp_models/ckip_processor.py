"""
CKIP Transformers Integration for Chinese NLP
Provides advanced Chinese text processing capabilities
"""

import logging
from typing import List, Dict, Tuple, Optional
import torch
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import json
import re

logger = logging.getLogger(__name__)

class CKIPProcessor:
    """Advanced Chinese text processor using CKIP Transformers"""
    
    def __init__(self, device: str = "auto"):
        """Initialize CKIP models"""
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing CKIP models on device: {self.device}")
        
        try:
            # Initialize CKIP models
            self.ws = CkipWordSegmenter(device=self.device)
            self.pos = CkipPosTagger(device=self.device)
            self.ner = CkipNerChunker(device=self.device)
            
            logger.info("CKIP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CKIP models: {e}")
            # Fallback to CPU
            self.device = "cpu"
            self.ws = CkipWordSegmenter(device=self.device)
            self.pos = CkipPosTagger(device=self.device)
            self.ner = CkipNerChunker(device=self.device)
    
    def segment_words(self, texts: List[str]) -> List[List[str]]:
        """Segment Chinese text into words"""
        try:
            return self.ws(texts)
        except Exception as e:
            logger.error(f"Word segmentation failed: {e}")
            return [text.split() for text in texts]
    
    def pos_tag(self, word_segments: List[List[str]]) -> List[List[str]]:
        """POS tagging for segmented words"""
        try:
            return self.pos(word_segments)
        except Exception as e:
            logger.error(f"POS tagging failed: {e}")
            return [['N'] * len(segment) for segment in word_segments]
    
    def ner_tag(self, word_segments: List[List[str]]) -> List[Dict]:
        """Named Entity Recognition"""
        try:
            return self.ner(word_segments)
        except Exception as e:
            logger.error(f"NER tagging failed: {e}")
            return [[] for _ in word_segments]
    
    def process_texts(self, texts: List[str]) -> List[Dict]:
        """Complete NLP processing pipeline"""
        
        # Word segmentation
        word_segments = self.segment_words(texts)
        
        # POS tagging
        pos_tags = self.pos_tag(word_segments)
        
        # NER tagging
        ner_results = self.ner_tag(word_segments)
        
        # Combine results
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'words': word_segments[i] if i < len(word_segments) else [],
                'pos_tags': pos_tags[i] if i < len(pos_tags) else [],
                'ner_entities': ner_results[i] if i < len(ner_results) else [],
                'characters': self.extract_characters(
                    word_segments[i] if i < len(word_segments) else [],
                    pos_tags[i] if i < len(pos_tags) else [],
                    ner_results[i] if i < len(ner_results) else []
                )
            }
            results.append(result)
        
        return results
    
    def extract_characters(self, words: List[str], pos_tags: List[str], ner_entities: List[Dict]) -> List[Dict]:
        """Extract character entities from processed text"""
        
        characters = []
        
        # Process NER entities
        for entity in ner_entities:
            if entity.get('type') in ['PERSON', 'PER']:
                characters.append({
                    'name': entity.get('word', ''),
                    'type': 'person',
                    'source': 'ner',
                    'confidence': 0.9,
                    'start_pos': entity.get('idx', 0),
                    'end_pos': entity.get('idx', 0) + len(entity.get('word', '')),
                    'context': entity
                })
        
        # Process POS tagged words for potential characters
        for word, pos in zip(words, pos_tags):
            if self.is_potential_character(word, pos):
                # Check if not already found by NER
                if not any(char['name'] == word for char in characters):
                    characters.append({
                        'name': word,
                        'type': self.classify_character_type(word, pos),
                        'source': 'pos',
                        'confidence': 0.7,
                        'pos_tag': pos,
                        'context': {'word': word, 'pos': pos}
                    })
        
        # Additional pattern-based character detection
        pattern_chars = self.pattern_based_character_extraction(words, pos_tags)
        for char in pattern_chars:
            if not any(c['name'] == char['name'] for c in characters):
                characters.append(char)
        
        return characters
    
    def is_potential_character(self, word: str, pos: str) -> bool:
        """Check if word is potentially a character name"""
        
        # Chinese name patterns
        name_patterns = [
            r'^[一-龥]{2,4}$',  # 2-4 Chinese characters
            r'^小[一-龥]{1,2}$',  # Names starting with 小
            r'^[王李張劉陳楊趙黃周吳徐孫胡朱高林何郭馬羅梁宋鄭謝韓唐馮於董蕭程曹袁鄧許傅沈曾彭呂蘇盧蔣蔡賈丁魏薛葉閻余潘杜戴夏鍾汪田任姜崔范方石姚譚廖鄒熊金陸郝孔白崔康毛邱秦江史顧侯邵孟龍萬段漕錢湯尹黎易常武喬賀][一-龥]{1,2}$',  # Common surnames
        ]
        
        # POS tags indicating names
        name_pos_tags = ['Nb', 'Nc', 'Nep', 'Nes']  # CKIP name-related POS tags
        
        # Check patterns
        for pattern in name_patterns:
            if re.match(pattern, word):
                return True
        
        # Check POS tags
        if pos in name_pos_tags:
            return True
        
        return False
    
    def classify_character_type(self, word: str, pos: str) -> str:
        """Classify character type based on word and POS"""
        
        if '老師' in word or '教師' in word:
            return 'teacher'
        elif word.startswith('小') and len(word) <= 3:
            return 'student'
        elif '先生' in word or '女士' in word:
            return 'adult'
        elif any(title in word for title in ['博士', '教授', '醫生', '工程師']):
            return 'professional'
        else:
            return 'person'
    
    def pattern_based_character_extraction(self, words: List[str], pos_tags: List[str]) -> List[Dict]:
        """Extract characters using linguistic patterns"""
        
        characters = []
        text = ' '.join(words)
        
        # Chinese character name patterns
        patterns = [
            # Direct speech patterns
            (r'([一-龥]{2,4})說', 'speaker'),
            (r'([一-龥]{2,4})問', 'speaker'),
            (r'([一-龥]{2,4})回答', 'speaker'),
            
            # Action patterns
            (r'([一-龥]{2,4})去', 'actor'),
            (r'([一-龥]{2,4})來', 'actor'),
            (r'([一-龥]{2,4})做', 'actor'),
            
            # Relationship patterns
            (r'和([一-龥]{2,4})一起', 'companion'),
            (r'跟([一-龥]{2,4})一起', 'companion'),
            (r'([一-龥]{2,4})的[朋友|同學|老師|學生]', 'relational'),
            
            # Title patterns
            (r'([一-龥]+)老師', 'teacher'),
            (r'([一-龥]+)同學', 'student'),
            (r'([一-龥]+)先生', 'adult_male'),
            (r'([一-龥]+)女士', 'adult_female'),
        ]
        
        for pattern, char_type in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    name = match[0]
                else:
                    name = match
                
                if len(name) >= 2 and len(name) <= 4:
                    characters.append({
                        'name': name,
                        'type': char_type,
                        'source': 'pattern',
                        'confidence': 0.8,
                        'pattern': pattern,
                        'context': {'matched_text': text}
                    })
        
        return characters
    
    def analyze_text_structure(self, text: str) -> Dict:
        """Analyze text structure and discourse elements"""
        
        results = self.process_texts([text])[0]
        
        # Discourse analysis
        discourse_markers = self.identify_discourse_markers(results['words'], results['pos_tags'])
        
        # Sentence relationships
        sentence_relations = self.analyze_sentence_relations(text)
        
        # Character interactions
        character_interactions = self.analyze_character_interactions(results['characters'], text)
        
        return {
            'basic_analysis': results,
            'discourse_markers': discourse_markers,
            'sentence_relations': sentence_relations,
            'character_interactions': character_interactions,
            'text_complexity': self.calculate_text_complexity(results)
        }
    
    def identify_discourse_markers(self, words: List[str], pos_tags: List[str]) -> List[Dict]:
        """Identify discourse markers in text"""
        
        discourse_markers = []
        
        # Common Chinese discourse markers
        markers = {
            '但是': 'contrast',
            '然而': 'contrast',
            '不過': 'contrast',
            '因此': 'result',
            '所以': 'result',
            '然後': 'sequence',
            '接著': 'sequence',
            '首先': 'order',
            '其次': 'order',
            '最後': 'order',
            '另外': 'addition',
            '此外': 'addition',
            '例如': 'example',
            '比如': 'example'
        }
        
        for i, word in enumerate(words):
            if word in markers:
                discourse_markers.append({
                    'marker': word,
                    'type': markers[word],
                    'position': i,
                    'function': self.get_marker_function(markers[word])
                })
        
        return discourse_markers
    
    def get_marker_function(self, marker_type: str) -> str:
        """Get the functional description of discourse marker"""
        
        functions = {
            'contrast': 'Indicates contrast or opposition',
            'result': 'Shows cause-effect relationship',
            'sequence': 'Indicates temporal sequence',
            'order': 'Shows logical order',
            'addition': 'Adds information',
            'example': 'Provides examples'
        }
        
        return functions.get(marker_type, 'Unknown function')
    
    def analyze_sentence_relations(self, text: str) -> List[Dict]:
        """Analyze relationships between sentences"""
        
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        relations = []
        
        for i in range(len(sentences) - 1):
            current = sentences[i]
            next_sent = sentences[i + 1]
            
            relation = self.determine_sentence_relation(current, next_sent)
            
            relations.append({
                'sentence1': current,
                'sentence2': next_sent,
                'relation': relation,
                'confidence': 0.7
            })
        
        return relations
    
    def determine_sentence_relation(self, sent1: str, sent2: str) -> str:
        """Determine semantic relation between two sentences"""
        
        # Simple heuristics for sentence relations
        if any(word in sent2 for word in ['但是', '然而', '不過']):
            return 'contrast'
        elif any(word in sent2 for word in ['因此', '所以', '因為']):
            return 'causation'
        elif any(word in sent2 for word in ['然後', '接著', '後來']):
            return 'sequence'
        elif any(word in sent2 for word in ['同時', '另外', '還有']):
            return 'addition'
        else:
            return 'continuation'
    
    def analyze_character_interactions(self, characters: List[Dict], text: str) -> List[Dict]:
        """Analyze interactions between characters"""
        
        interactions = []
        
        # Find character co-occurrences
        for i, char1 in enumerate(characters):
            for j, char2 in enumerate(characters[i+1:], i+1):
                
                name1 = char1['name']
                name2 = char2['name']
                
                # Check if characters appear in same context
                interaction_patterns = [
                    f"{name1}和{name2}",
                    f"{name1}跟{name2}",
                    f"{name2}和{name1}",
                    f"{name2}跟{name1}",
                ]
                
                for pattern in interaction_patterns:
                    if pattern in text:
                        interactions.append({
                            'character1': name1,
                            'character2': name2,
                            'interaction_type': 'collaboration',
                            'evidence': pattern,
                            'context': self.extract_interaction_context(text, pattern)
                        })
                        break
        
        return interactions
    
    def extract_interaction_context(self, text: str, pattern: str) -> str:
        """Extract context around character interaction"""
        
        # Find the sentence containing the pattern
        sentences = re.split(r'[。！？]', text)
        
        for sentence in sentences:
            if pattern in sentence:
                return sentence.strip()
        
        return ""
    
    def calculate_text_complexity(self, analysis_result: Dict) -> Dict:
        """Calculate text complexity metrics"""
        
        words = analysis_result['words']
        pos_tags = analysis_result['pos_tags']
        
        # Basic metrics
        total_words = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        
        # POS diversity
        pos_diversity = len(set(pos_tags)) / len(pos_tags) if pos_tags else 0
        
        # Character count
        character_count = len(analysis_result['characters'])
        
        complexity_score = (
            unique_words / total_words * 0.3 +
            avg_word_length / 4 * 0.2 +
            pos_diversity * 0.3 +
            min(character_count / 5, 1) * 0.2
        )
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'lexical_diversity': unique_words / total_words if total_words > 0 else 0,
            'avg_word_length': avg_word_length,
            'pos_diversity': pos_diversity,
            'character_count': character_count,
            'complexity_score': complexity_score,
            'complexity_level': self.get_complexity_level(complexity_score)
        }
    
    def get_complexity_level(self, score: float) -> str:
        """Determine text complexity level"""
        
        if score < 0.3:
            return 'Simple'
        elif score < 0.6:
            return 'Moderate'
        elif score < 0.8:
            return 'Complex'
        else:
            return 'Very Complex'


class CKIPCharacterExtractor:
    """Specialized character extractor using CKIP"""
    
    def __init__(self):
        self.processor = CKIPProcessor()
        
    def extract_characters_advanced(self, text: str) -> List[Dict]:
        """Advanced character extraction with CKIP analysis"""
        
        # Full CKIP analysis
        analysis = self.processor.analyze_text_structure(text)
        
        characters = analysis['basic_analysis']['characters']
        
        # Enhance character information
        enhanced_characters = []
        
        for char in characters:
            enhanced_char = char.copy()
            
            # Add interaction information
            enhanced_char['interactions'] = [
                interaction for interaction in analysis['character_interactions']
                if char['name'] in [interaction['character1'], interaction['character2']]
            ]
            
            # Add behavioral analysis
            enhanced_char['behaviors'] = self.analyze_character_behavior(
                char['name'], text, analysis
            )
            
            # Add role classification
            enhanced_char['role'] = self.classify_character_role(
                char, analysis
            )
            
            enhanced_characters.append(enhanced_char)
        
        return enhanced_characters
    
    def analyze_character_behavior(self, character_name: str, text: str, analysis: Dict) -> List[Dict]:
        """Analyze character behaviors from text"""
        
        behaviors = []
        
        # Action verbs associated with character
        action_patterns = [
            f"{character_name}([說問回答講述談論])",
            f"{character_name}([去來走跑跳])",
            f"{character_name}([做作寫畫讀學習])",
            f"{character_name}([看見注意觀察發現])",
            f"{character_name}([想思考考慮決定])",
        ]
        
        behavior_categories = {
            '說問回答講述談論': 'communication',
            '去來走跑跳': 'movement',
            '做作寫畫讀學習': 'activity',
            '看見注意觀察發現': 'perception',
            '想思考考慮決定': 'cognition'
        }
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                for category, behavior_type in behavior_categories.items():
                    if match in category:
                        behaviors.append({
                            'action': match,
                            'category': behavior_type,
                            'context': self.extract_action_context(text, character_name, match),
                            'confidence': 0.8
                        })
                        break
        
        return behaviors
    
    def extract_action_context(self, text: str, character: str, action: str) -> str:
        """Extract context around character action"""
        
        # Find sentence containing the action
        pattern = f"{character}{action}"
        sentences = re.split(r'[。！？]', text)
        
        for sentence in sentences:
            if pattern in sentence:
                return sentence.strip()
        
        return ""
    
    def classify_character_role(self, character: Dict, analysis: Dict) -> str:
        """Classify character role in the narrative"""
        
        char_name = character['name']
        
        # Count interactions and behaviors
        interaction_count = len([
            i for i in analysis['character_interactions']
            if char_name in [i['character1'], i['character2']]
        ])
        
        behavior_count = len(character.get('behaviors', []))
        
        # Role classification logic
        if interaction_count >= 2 and behavior_count >= 3:
            return 'protagonist'
        elif interaction_count >= 1 or behavior_count >= 2:
            return 'supporting'
        else:
            return 'minor'


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize CKIP processor
    processor = CKIPProcessor()
    
    # Test text
    test_text = "小明是一個學生，他每天上課學習。今天小明和同學小華一起去圖書館讀書。王老師看到他們很滿意。"
    
    # Process text
    results = processor.analyze_text_structure(test_text)
    
    print("Analysis Results:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # Character extraction
    extractor = CKIPCharacterExtractor()
    characters = extractor.extract_characters_advanced(test_text)
    
    print("\nExtracted Characters:")
    for char in characters:
        print(f"Name: {char['name']}, Type: {char['type']}, Role: {char.get('role', 'unknown')}")
        print(f"Behaviors: {[b['action'] for b in char.get('behaviors', [])]}")
        print(f"Interactions: {len(char.get('interactions', []))}")
        print("---")