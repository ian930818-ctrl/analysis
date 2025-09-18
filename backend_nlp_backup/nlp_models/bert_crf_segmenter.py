"""
BERT+CRF Model for Chinese Sentence Segmentation
Combines BERT contextual embeddings with CRF for sequence labeling
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from torchcrf import CRF
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)

class BertCrfSegmenter(nn.Module):
    """BERT + CRF model for Chinese sentence segmentation"""
    
    def __init__(self, 
                 model_name: str = "bert-base-chinese",
                 num_labels: int = 4,  # B-SENT, I-SENT, E-SENT, S-SENT
                 dropout: float = 0.1,
                 freeze_bert: bool = False):
        super().__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load BERT model
        try:
            self.bert = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            logger.warning(f"Failed to load {model_name}, using bert-base-chinese")
            self.bert = BertModel.from_pretrained("bert-base-chinese")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
        # Label mapping
        self.label2id = {
            'B-SENT': 0,  # Beginning of sentence
            'I-SENT': 1,  # Inside sentence
            'E-SENT': 2,  # End of sentence
            'S-SENT': 3   # Single character sentence
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through BERT+CRF"""
        
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Classification layer
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            # Training mode - compute loss
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss, logits
        else:
            # Inference mode - decode best path
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
            return logits, predictions
    
    def segment_text(self, text: str, max_length: int = 512) -> List[str]:
        """Segment Chinese text into sentences"""
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Get predictions
        with torch.no_grad():
            self.eval()
            logits, predictions = self(input_ids, attention_mask)
        
        # Convert predictions to sentences
        sentences = self._predictions_to_sentences(text, predictions[0])
        return sentences
    
    def _predictions_to_sentences(self, text: str, predictions: List[int]) -> List[str]:
        """Convert BIO predictions to sentence list"""
        
        sentences = []
        current_sentence = ""
        
        # Tokenize to align with predictions
        tokens = self.tokenizer.tokenize(text)
        
        # Handle subword tokens and sentence boundaries
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if token.startswith('##'):
                # Subword token - continue current sentence
                current_sentence += token[2:]
            else:
                current_sentence += token
            
            # Check for sentence boundary
            label = self.id2label.get(pred_id, 'I-SENT')
            
            if label in ['E-SENT', 'S-SENT']:
                # End of sentence
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add remaining text as final sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def save_model(self, save_path: str):
        """Save model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        
        with open(os.path.join(save_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, model_path: str):
        """Load trained model"""
        
        # Load config
        with open(os.path.join(model_path, "config.json"), 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(
            model_name=config['model_name'],
            num_labels=config['num_labels']
        )
        
        # Load state dict
        model.load_state_dict(torch.load(
            os.path.join(model_path, "model.pt"),
            map_location='cpu'
        ))
        
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"Model loaded from {model_path}")
        return model


class ChineseCorpusCollector:
    """Collect and prepare Chinese text corpus for training"""
    
    def __init__(self):
        self.sentences_data = []
    
    def collect_news_corpus(self) -> List[str]:
        """Collect Chinese news corpus for sentence segmentation"""
        
        # Sample Chinese sentences for training
        sample_sentences = [
            "小明是一個學生，他每天上課學習。",
            "今天小明和同學小華一起去圖書館讀書。",
            "王老師看到他們很滿意，誇獎了他們的努力。",
            "小華說：「我們應該更加努力學習！」",
            "老師回答：「是的，學習是很重要的事情。」",
            "他們決定每天都要來圖書館讀書。",
            "小明的媽媽為他準備了營養豐富的午餐。",
            "下午，小華和小明一起踢足球。",
            "晚上，他們各自回家做功課。",
            "這是充實而愉快的一天。"
        ]
        
        # Expand with common Chinese patterns
        common_patterns = [
            "我今天去了學校。老師教我們新的知識。",
            "春天到了，花開了。鳥兒在樹上唱歌。",
            "媽媽做飯，爸爸看報紙。我在房間裡讀書。",
            "朋友們一起踢球。大家都很開心。",
            "圖書館裡很安靜。學生們認真地學習。",
            "今天天氣很好。我們決定去公園散步。",
            "老師說：「同學們要好好學習！」我們都點頭同意。",
            "小貓在院子裡玩耍。牠看起來很可愛。",
            "週末到了。家人們聚在一起吃飯。",
            "書店裡有很多好書。我買了幾本回家看。"
        ]
        
        all_sentences = sample_sentences + common_patterns
        
        # Generate BIO labels for each sentence
        labeled_data = []
        for text in all_sentences:
            sentences = text.split('。')
            sentences = [s.strip() + '。' for s in sentences if s.strip()]
            
            # Create character-level labels
            char_labels = []
            for sentence in sentences:
                if len(sentence) == 1:
                    char_labels.append('S-SENT')
                elif len(sentence) > 1:
                    char_labels.append('B-SENT')
                    for _ in range(len(sentence) - 2):
                        char_labels.append('I-SENT')
                    char_labels.append('E-SENT')
            
            labeled_data.append({
                'text': text,
                'labels': char_labels,
                'sentences': sentences
            })
        
        self.sentences_data = labeled_data
        logger.info(f"Collected {len(labeled_data)} training samples")
        
        return labeled_data
    
    def save_corpus(self, filepath: str):
        """Save collected corpus to file"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.sentences_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Corpus saved to {filepath}")
    
    def load_corpus(self, filepath: str):
        """Load corpus from file"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.sentences_data = json.load(f)
        
        logger.info(f"Corpus loaded from {filepath}")
        return self.sentences_data


# Training utilities
class SegmentationTrainer:
    """Trainer for BERT+CRF segmentation model"""
    
    def __init__(self, model: BertCrfSegmenter, learning_rate: float = 2e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = None
    
    def prepare_training_data(self, corpus_data: List[Dict]) -> List[Dict]:
        """Prepare training data from corpus"""
        
        training_samples = []
        
        for sample in corpus_data:
            text = sample['text']
            labels = sample['labels']
            
            # Tokenize and align labels
            encoded = self.model.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Align labels with subword tokens
            aligned_labels = self._align_labels_with_tokens(text, labels, encoded)
            
            training_samples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(aligned_labels, dtype=torch.long)
            })
        
        return training_samples
    
    def _align_labels_with_tokens(self, text: str, labels: List[str], encoded) -> List[int]:
        """Align character-level labels with subword tokens"""
        
        tokens = self.model.tokenizer.tokenize(text)
        aligned_labels = []
        
        char_idx = 0
        for token in tokens:
            if token.startswith('##'):
                # Subword token - use same label as previous
                if aligned_labels:
                    aligned_labels.append(aligned_labels[-1])
                else:
                    aligned_labels.append(self.model.label2id['I-SENT'])
            else:
                # Regular token
                if char_idx < len(labels):
                    label = labels[char_idx]
                    aligned_labels.append(self.model.label2id.get(label, self.model.label2id['I-SENT']))
                    char_idx += 1
                else:
                    aligned_labels.append(self.model.label2id['I-SENT'])
        
        # Pad or truncate to match sequence length
        seq_len = encoded['input_ids'].size(1)
        while len(aligned_labels) < seq_len:
            aligned_labels.append(self.model.label2id['I-SENT'])
        
        return aligned_labels[:seq_len]
    
    def train_epoch(self, training_data: List[Dict]) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        
        for batch in training_data:
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].unsqueeze(0)
            attention_mask = batch['attention_mask'].unsqueeze(0)
            labels = batch['labels'].unsqueeze(0)
            
            loss, _ = self.model(input_ids, attention_mask, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(training_data)
    
    def train(self, corpus_data: List[Dict], epochs: int = 10) -> List[float]:
        """Full training loop"""
        
        training_data = self.prepare_training_data(corpus_data)
        losses = []
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = self.train_epoch(training_data)
            losses.append(epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        logger.info("Training completed!")
        return losses


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Collect corpus
    collector = ChineseCorpusCollector()
    corpus_data = collector.collect_news_corpus()
    
    # Initialize model
    model = BertCrfSegmenter()
    
    # Train model
    trainer = SegmentationTrainer(model)
    losses = trainer.train(corpus_data, epochs=5)
    
    # Test segmentation
    test_text = "小明去學校上課。他很認真地學習。老師誇獎了他。"
    sentences = model.segment_text(test_text)
    
    print("Original text:", test_text)
    print("Segmented sentences:", sentences)