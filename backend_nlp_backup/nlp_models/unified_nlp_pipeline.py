"""
Unified NLP Pipeline for Advanced Chinese Character Analysis
Integrates BERT+CRF, CKIP, and fine-tuned models
"""

import logging
import json
import os
from typing import List, Dict, Optional, Union, Tuple
import torch
import numpy as np
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle

# Import custom modules
from .bert_crf_segmenter import BertCrfSegmenter, ChineseCorpusCollector, SegmentationTrainer
from .ckip_processor import CKIPProcessor, CKIPCharacterExtractor
from .bert_finetuning import MultiTaskBertForNER, BertFineTuner, CharacterExtractionPipeline

logger = logging.getLogger(__name__)

class UnifiedNLPPipeline:
    """Advanced Chinese NLP pipeline combining multiple models"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 device: str = "auto",
                 cache_dir: str = "./nlp_cache"):
        
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize models
        self.models = {}
        self.model_weights = self.config.get('model_weights', {
            'bert_crf': 0.3,
            'ckip': 0.4,
            'finetuned_bert': 0.3
        })
        
        # Initialize components
        self.initialize_models()
        
        # Cache for expensive operations
        self.cache = {}
        
        logger.info(f"Unified NLP Pipeline initialized on device: {self.device}")
    
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load pipeline configuration"""
        
        default_config = {
            "models": {
                "bert_crf": {
                    "enabled": True,
                    "model_name": "bert-base-chinese",
                    "model_path": None
                },
                "ckip": {
                    "enabled": True,
                    "device": self.device
                },
                "finetuned_bert": {
                    "enabled": True,
                    "model_path": "./chinese_ner_finetuned",
                    "auto_train": True
                }
            },
            "ensemble": {
                "method": "weighted_voting",
                "confidence_threshold": 0.6,
                "min_models_agreement": 2
            },
            "preprocessing": {
                "text_cleaning": True,
                "sentence_segmentation": True,
                "max_length": 512
            },
            "model_weights": {
                "bert_crf": 0.3,
                "ckip": 0.4,
                "finetuned_bert": 0.3
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def initialize_models(self):
        """Initialize all NLP models"""
        
        logger.info("Initializing NLP models...")
        
        # BERT+CRF Segmenter
        if self.config['models']['bert_crf']['enabled']:
            try:
                if self.config['models']['bert_crf']['model_path']:
                    self.models['bert_crf'] = BertCrfSegmenter.load_model(
                        self.config['models']['bert_crf']['model_path']
                    )
                else:
                    self.models['bert_crf'] = BertCrfSegmenter(
                        model_name=self.config['models']['bert_crf']['model_name']
                    )
                logger.info("BERT+CRF segmenter loaded")
            except Exception as e:
                logger.error(f"Failed to load BERT+CRF model: {e}")
        
        # CKIP Processor
        if self.config['models']['ckip']['enabled']:
            try:
                self.models['ckip_processor'] = CKIPProcessor(device=self.device)
                self.models['ckip_extractor'] = CKIPCharacterExtractor()
                logger.info("CKIP models loaded")
            except Exception as e:
                logger.error(f"Failed to load CKIP models: {e}")
        
        # Fine-tuned BERT
        if self.config['models']['finetuned_bert']['enabled']:
            try:
                model_path = self.config['models']['finetuned_bert']['model_path']
                if os.path.exists(model_path):
                    self.models['finetuned_bert'] = CharacterExtractionPipeline(model_path)
                    logger.info("Fine-tuned BERT model loaded")
                elif self.config['models']['finetuned_bert']['auto_train']:
                    logger.info("Fine-tuned model not found, training new model...")
                    self.train_finetuned_model()
                else:
                    logger.warning("Fine-tuned BERT model not available")
            except Exception as e:
                logger.error(f"Failed to load fine-tuned BERT model: {e}")
    
    def train_finetuned_model(self):
        """Train a new fine-tuned model if needed"""
        
        try:
            model_path = self.config['models']['finetuned_bert']['model_path']
            
            # Initialize trainer
            tuner = BertFineTuner(
                model_name="bert-base-chinese",
                output_dir=model_path,
                use_wandb=False
            )
            
            # Quick training with small dataset
            logger.info("Starting quick training session...")
            trainer = tuner.train(num_epochs=2, batch_size=4)
            
            # Load trained model
            self.models['finetuned_bert'] = CharacterExtractionPipeline(model_path)
            logger.info("Fine-tuned model trained and loaded")
            
        except Exception as e:
            logger.error(f"Failed to train fine-tuned model: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        
        if not self.config['preprocessing']['text_cleaning']:
            return text
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long
        max_length = self.config['preprocessing']['max_length']
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        """Advanced sentence segmentation"""
        
        if 'bert_crf' in self.models:
            try:
                sentences = self.models['bert_crf'].segment_text(text)
                return sentences
            except Exception as e:
                logger.error(f"BERT+CRF segmentation failed: {e}")
        
        # Fallback to simple segmentation
        import re
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def extract_characters_single_model(self, text: str, model_name: str) -> List[Dict]:
        """Extract characters using a single model"""
        
        characters = []
        
        try:
            if model_name == 'ckip' and 'ckip_extractor' in self.models:
                characters = self.models['ckip_extractor'].extract_characters_advanced(text)
                
            elif model_name == 'finetuned_bert' and 'finetuned_bert' in self.models:
                characters = self.models['finetuned_bert'].extract_characters(text)
                
            elif model_name == 'bert_crf' and 'bert_crf' in self.models:
                # Use BERT+CRF for pattern-based extraction
                sentences = self.models['bert_crf'].segment_text(text)
                characters = self.extract_from_segmented_sentences(sentences)
            
            # Add model source information
            for char in characters:
                char['model_source'] = model_name
                
        except Exception as e:
            logger.error(f"Character extraction failed for {model_name}: {e}")
        
        return characters
    
    def extract_from_segmented_sentences(self, sentences: List[str]) -> List[Dict]:
        """Extract characters from segmented sentences using patterns"""
        
        characters = []
        
        # Enhanced patterns for character extraction
        patterns = {
            'names': [
                r'([一-龥]{2,4})說',
                r'([一-龥]{2,4})問',
                r'([一-龥]{2,4})回答',
                r'小([一-龥]{1,2})',
                r'([一-龥]+)老師',
                r'([一-龥]+)同學',
            ],
            'roles': [
                r'(老師|教師|學生|同學)',
                r'([一-龥]+)(先生|女士|小姐)',
            ],
            'actions': [
                r'([一-龥]{2,4})(說|問|答|去|來|做|看|想)',
            ]
        }
        
        full_text = '。'.join(sentences)
        
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                import re
                matches = re.findall(pattern, full_text)
                
                for match in matches:
                    if isinstance(match, tuple):
                        name = ''.join(match)
                    else:
                        name = match
                    
                    if len(name) >= 2 and len(name) <= 4:
                        characters.append({
                            'name': name,
                            'type': self.classify_character_type(name),
                            'source': 'bert_crf_pattern',
                            'confidence': 0.7,
                            'pattern_type': pattern_type,
                            'context': self.find_character_context(full_text, name)
                        })
        
        return characters
    
    def classify_character_type(self, name: str) -> str:
        """Classify character type based on name"""
        
        if '老師' in name:
            return 'teacher'
        elif name.startswith('小'):
            return 'student'
        elif any(title in name for title in ['先生', '女士', '小姐']):
            return 'adult'
        else:
            return 'person'
    
    def find_character_context(self, text: str, character: str) -> str:
        """Find context around character mention"""
        
        import re
        
        # Find sentence containing the character
        sentences = re.split(r'[。！？]', text)
        
        for sentence in sentences:
            if character in sentence:
                return sentence.strip()
        
        return ""
    
    def ensemble_character_extraction(self, text: str) -> List[Dict]:
        """Extract characters using ensemble of models"""
        
        all_extractions = {}
        
        # Extract from each model
        for model_name in ['ckip', 'finetuned_bert', 'bert_crf']:
            if model_name in self.config['models'] and self.config['models'][model_name]['enabled']:
                extractions = self.extract_characters_single_model(text, model_name)
                all_extractions[model_name] = extractions
        
        # Ensemble results
        if self.config['ensemble']['method'] == 'weighted_voting':
            final_characters = self.weighted_voting_ensemble(all_extractions)
        else:
            final_characters = self.simple_merge_ensemble(all_extractions)
        
        return final_characters
    
    def weighted_voting_ensemble(self, all_extractions: Dict[str, List[Dict]]) -> List[Dict]:
        """Ensemble using weighted voting"""
        
        character_votes = {}
        
        # Collect votes from each model
        for model_name, characters in all_extractions.items():
            weight = self.model_weights.get(model_name, 1.0)
            
            for char in characters:
                name = char['name']
                confidence = char.get('confidence', 0.5)
                
                if name not in character_votes:
                    character_votes[name] = {
                        'votes': 0,
                        'weighted_confidence': 0,
                        'details': [],
                        'sources': set()
                    }
                
                character_votes[name]['votes'] += weight
                character_votes[name]['weighted_confidence'] += weight * confidence
                character_votes[name]['details'].append(char)
                character_votes[name]['sources'].add(model_name)
        
        # Filter and rank characters
        final_characters = []
        min_models = self.config['ensemble']['min_models_agreement']
        confidence_threshold = self.config['ensemble']['confidence_threshold']
        
        for name, vote_data in character_votes.items():
            if len(vote_data['sources']) >= min_models:
                avg_confidence = vote_data['weighted_confidence'] / vote_data['votes']
                
                if avg_confidence >= confidence_threshold:
                    # Create consensus character
                    consensus_char = self.create_consensus_character(name, vote_data)
                    final_characters.append(consensus_char)
        
        # Sort by confidence
        final_characters.sort(key=lambda x: x['confidence'], reverse=True)
        
        return final_characters
    
    def create_consensus_character(self, name: str, vote_data: Dict) -> Dict:
        """Create consensus character from multiple model outputs"""
        
        # Get most common type
        types = [char.get('type', 'person') for char in vote_data['details']]
        most_common_type = max(set(types), key=types.count)
        
        # Average confidence
        avg_confidence = vote_data['weighted_confidence'] / vote_data['votes']
        
        # Merge behaviors and attributes
        all_behaviors = []
        all_attributes = []
        
        for char in vote_data['details']:
            all_behaviors.extend(char.get('behaviors', []))
            all_attributes.extend(char.get('attributes', []))
        
        # Remove duplicates
        unique_behaviors = []
        seen_behaviors = set()
        for behavior in all_behaviors:
            behavior_key = behavior.get('action', '') + behavior.get('category', '')
            if behavior_key not in seen_behaviors:
                unique_behaviors.append(behavior)
                seen_behaviors.add(behavior_key)
        
        consensus_char = {
            'name': name,
            'type': most_common_type,
            'confidence': avg_confidence,
            'source': 'ensemble',
            'model_sources': list(vote_data['sources']),
            'vote_count': vote_data['votes'],
            'behaviors': unique_behaviors[:5],  # Top 5 behaviors
            'attributes': all_attributes[:3],   # Top 3 attributes
            'context': self.find_character_context(' '.join([
                char.get('context', '') for char in vote_data['details']
            ]), name),
            'description': f"{name} - {most_common_type}"
        }
        
        return consensus_char
    
    def simple_merge_ensemble(self, all_extractions: Dict[str, List[Dict]]) -> List[Dict]:
        """Simple merge ensemble method"""
        
        all_characters = []
        seen_names = set()
        
        for model_name, characters in all_extractions.items():
            for char in characters:
                if char['name'] not in seen_names:
                    char['model_source'] = model_name
                    all_characters.append(char)
                    seen_names.add(char['name'])
        
        return all_characters
    
    def analyze_text_comprehensive(self, text: str) -> Dict:
        """Comprehensive text analysis"""
        
        # Cache key
        cache_key = f"analysis_{hash(text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Preprocess
        text = self.preprocess_text(text)
        
        # Sentence segmentation
        sentences = self.segment_sentences(text)
        
        # Character extraction
        characters = self.ensemble_character_extraction(text)
        
        # Additional analysis
        analysis_result = {
            'text': text,
            'sentences': sentences,
            'characters': characters,
            'character_count': len(characters),
            'sentence_count': len(sentences),
            'text_length': len(text),
            'analysis_timestamp': datetime.now().isoformat(),
            'models_used': list(self.models.keys()),
            'pipeline_version': "2.0.0"
        }
        
        # Add character interactions if CKIP is available
        if 'ckip_processor' in self.models:
            try:
                ckip_analysis = self.models['ckip_processor'].analyze_text_structure(text)
                analysis_result['character_interactions'] = ckip_analysis.get('character_interactions', [])
                analysis_result['text_complexity'] = ckip_analysis.get('text_complexity', {})
            except Exception as e:
                logger.error(f"CKIP analysis failed: {e}")
        
        # Cache result
        self.cache[cache_key] = analysis_result
        
        return analysis_result
    
    async def analyze_text_async(self, text: str) -> Dict:
        """Asynchronous text analysis"""
        
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Run different models in parallel
            futures = []
            
            if 'ckip_extractor' in self.models:
                futures.append(
                    loop.run_in_executor(
                        executor, 
                        self.extract_characters_single_model, 
                        text, 'ckip'
                    )
                )
            
            if 'finetuned_bert' in self.models:
                futures.append(
                    loop.run_in_executor(
                        executor, 
                        self.extract_characters_single_model, 
                        text, 'finetuned_bert'
                    )
                )
            
            if 'bert_crf' in self.models:
                futures.append(
                    loop.run_in_executor(
                        executor, 
                        self.extract_characters_single_model, 
                        text, 'bert_crf'
                    )
                )
            
            # Wait for all extractions
            results = await asyncio.gather(*futures)
        
        # Combine results
        all_extractions = {}
        model_names = ['ckip', 'finetuned_bert', 'bert_crf']
        
        for i, result in enumerate(results):
            if i < len(model_names):
                all_extractions[model_names[i]] = result
        
        # Ensemble
        characters = self.weighted_voting_ensemble(all_extractions)
        
        return {
            'text': text,
            'characters': characters,
            'analysis_timestamp': datetime.now().isoformat(),
            'async_processing': True
        }
    
    def save_pipeline_state(self, filepath: str):
        """Save pipeline state for later loading"""
        
        state = {
            'config': self.config,
            'model_weights': self.model_weights,
            'cache': self.cache,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Pipeline state saved to {filepath}")
    
    def load_pipeline_state(self, filepath: str):
        """Load pipeline state"""
        
        if not os.path.exists(filepath):
            logger.warning(f"State file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.config.update(state.get('config', {}))
            self.model_weights.update(state.get('model_weights', {}))
            self.cache.update(state.get('cache', {}))
            
            logger.info(f"Pipeline state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline state: {e}")
    
    def get_pipeline_info(self) -> Dict:
        """Get pipeline information and status"""
        
        return {
            'models_loaded': list(self.models.keys()),
            'device': self.device,
            'cache_size': len(self.cache),
            'config': self.config,
            'model_weights': self.model_weights,
            'pipeline_version': "2.0.0"
        }


# Convenience function for easy usage
def create_advanced_nlp_pipeline(config_path: Optional[str] = None) -> UnifiedNLPPipeline:
    """Create and initialize the advanced NLP pipeline"""
    
    return UnifiedNLPPipeline(config_path=config_path)


# Test function
def test_pipeline():
    """Test the unified pipeline"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = create_advanced_nlp_pipeline()
    
    # Test text
    test_text = "小明是一個學生，他每天上課學習。今天小明和同學小華一起去圖書館讀書。王老師看到他們很滿意，誇獎了他們的努力。"
    
    # Analyze
    result = pipeline.analyze_text_comprehensive(test_text)
    
    print("Analysis Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Print characters
    print("\nExtracted Characters:")
    for char in result['characters']:
        print(f"- {char['name']} ({char['type']}) - Confidence: {char['confidence']:.2f}")
        if 'model_sources' in char:
            print(f"  Sources: {', '.join(char['model_sources'])}")
        print(f"  Description: {char.get('description', 'N/A')}")
        print()


if __name__ == "__main__":
    test_pipeline()