"""
Chinese BERT/RoBERTa Fine-tuning for Character Annotation
Advanced NER model with multi-task learning
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
import logging
from sklearn.metrics import classification_report, f1_score
from datasets import Dataset as HFDataset
import wandb
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

class ChineseCharacterDataset(Dataset):
    """Dataset for Chinese character annotation"""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[List[str]], 
                 tokenizer,
                 max_length: int = 512,
                 label_to_id: Dict[str, int] = None):
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default label mapping for character annotation
        if label_to_id is None:
            self.label_to_id = {
                'O': 0,           # Outside
                'B-PER': 1,       # Beginning of person
                'I-PER': 2,       # Inside person
                'B-ROLE': 3,      # Beginning of role (teacher, student)
                'I-ROLE': 4,      # Inside role
                'B-ACTION': 5,    # Beginning of action
                'I-ACTION': 6,    # Inside action
                'B-ATTR': 7,      # Beginning of attribute
                'I-ATTR': 8,      # Inside attribute
            }
        else:
            self.label_to_id = label_to_id
        
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokens
        aligned_labels = self.align_labels_with_tokens(text, labels, encoding)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def align_labels_with_tokens(self, text: str, labels: List[str], encoding) -> List[int]:
        """Align character-level labels with subword tokens"""
        
        tokens = self.tokenizer.tokenize(text)
        aligned_labels = []
        
        # Handle special tokens
        aligned_labels.append(self.label_to_id['O'])  # [CLS]
        
        char_idx = 0
        for token in tokens:
            if token.startswith('##'):
                # Subword token - use same label as previous
                if aligned_labels:
                    aligned_labels.append(aligned_labels[-1])
                else:
                    aligned_labels.append(self.label_to_id['O'])
            else:
                # Regular token
                if char_idx < len(labels):
                    label = labels[char_idx]
                    aligned_labels.append(self.label_to_id.get(label, self.label_to_id['O']))
                    char_idx += 1
                else:
                    aligned_labels.append(self.label_to_id['O'])
        
        # Handle [SEP] token
        aligned_labels.append(self.label_to_id['O'])
        
        # Pad or truncate to max_length
        while len(aligned_labels) < self.max_length:
            aligned_labels.append(self.label_to_id['O'])
        
        return aligned_labels[:self.max_length]


class MultiTaskBertForNER(nn.Module):
    """Multi-task BERT model for character annotation"""
    
    def __init__(self, 
                 model_name: str = "bert-base-chinese",
                 num_labels: int = 9,
                 dropout: float = 0.1,
                 use_crf: bool = True,
                 task_weights: Dict[str, float] = None):
        
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_crf = use_crf
        
        # Load base model
        if 'roberta' in model_name.lower():
            self.bert = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        
        # Main NER head
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Additional task heads
        self.role_classifier = nn.Linear(self.bert.config.hidden_size, 4)  # student, teacher, adult, other
        self.action_classifier = nn.Linear(self.bert.config.hidden_size, 6)  # speak, move, think, see, do, other
        
        # CRF layer for sequence labeling
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_labels, batch_first=True)
        
        # Task weights for multi-task learning
        if task_weights is None:
            self.task_weights = {'ner': 1.0, 'role': 0.3, 'action': 0.3}
        else:
            self.task_weights = task_weights
    
    def forward(self, input_ids, attention_mask, labels=None, role_labels=None, action_labels=None):
        """Forward pass with multi-task outputs"""
        
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # NER predictions
        ner_logits = self.ner_classifier(sequence_output)
        
        # Additional task predictions
        role_logits = self.role_classifier(sequence_output)
        action_logits = self.action_classifier(sequence_output)
        
        total_loss = 0
        
        if labels is not None:
            if self.use_crf:
                # CRF loss
                ner_loss = -self.crf(ner_logits, labels, mask=attention_mask.bool())
            else:
                # Standard cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                ner_loss = loss_fct(ner_logits.view(-1, self.num_labels), labels.view(-1))
            
            total_loss += self.task_weights['ner'] * ner_loss
            
            # Additional task losses
            if role_labels is not None:
                role_loss_fct = nn.CrossEntropyLoss()
                role_loss = role_loss_fct(role_logits.view(-1, 4), role_labels.view(-1))
                total_loss += self.task_weights['role'] * role_loss
            
            if action_labels is not None:
                action_loss_fct = nn.CrossEntropyLoss()
                action_loss = action_loss_fct(action_logits.view(-1, 6), action_labels.view(-1))
                total_loss += self.task_weights['action'] * action_loss
        
        if self.use_crf and labels is None:
            # Inference mode - decode best path
            predictions = self.crf.decode(ner_logits, mask=attention_mask.bool())
            return {
                'loss': total_loss,
                'ner_logits': ner_logits,
                'predictions': predictions,
                'role_logits': role_logits,
                'action_logits': action_logits
            }
        
        return {
            'loss': total_loss,
            'ner_logits': ner_logits,
            'role_logits': role_logits,
            'action_logits': action_logits
        }


class ChineseCorpusGenerator:
    """Generate training corpus for Chinese character annotation"""
    
    def __init__(self):
        self.character_patterns = {
            'students': ['小明', '小華', '小美', '小強', '小紅', '小志', '小文', '小玲'],
            'teachers': ['王老師', '李老師', '張老師', '陳老師', '林老師'],
            'adults': ['王先生', '李女士', '媽媽', '爸爸', '阿姨', '叔叔'],
            'actions': ['說', '問', '回答', '去', '來', '做', '看', '想', '學習', '玩耍'],
            'roles': ['學生', '老師', '教師', '同學', '朋友'],
            'attributes': ['聰明', '努力', '認真', '友善', '開朗', '安靜']
        }
    
    def generate_training_data(self, num_samples: int = 1000) -> Tuple[List[str], List[List[str]]]:
        """Generate labeled training data"""
        
        texts = []
        labels = []
        
        # Template-based generation
        templates = [
            "{student}是一個{role}，他很{attr}。",
            "{teacher}教{student}學習。",
            "{student}和{student2}一起{action}。",
            "{student}{action}：「我要努力學習！」",
            "{teacher}對{student}{action}：「你很{attr}。」",
            "今天{student}去學校{action}。",
            "{student}的{attr}讓{teacher}很滿意。",
            "{student}跟{student2}說：「我們一起{action}吧！」"
        ]
        
        for _ in range(num_samples):
            template = np.random.choice(templates)
            
            # Fill template with random choices
            filled_text = template.format(
                student=np.random.choice(self.character_patterns['students']),
                student2=np.random.choice(self.character_patterns['students']),
                teacher=np.random.choice(self.character_patterns['teachers']),
                role=np.random.choice(self.character_patterns['roles']),
                action=np.random.choice(self.character_patterns['actions']),
                attr=np.random.choice(self.character_patterns['attributes'])
            )
            
            # Generate labels for the text
            char_labels = self.generate_labels_for_text(filled_text)
            
            texts.append(filled_text)
            labels.append(char_labels)
        
        # Add real-world examples
        real_examples = self.get_real_world_examples()
        texts.extend([ex['text'] for ex in real_examples])
        labels.extend([ex['labels'] for ex in real_examples])
        
        return texts, labels
    
    def generate_labels_for_text(self, text: str) -> List[str]:
        """Generate BIO labels for text"""
        
        labels = ['O'] * len(text)
        
        # Label characters (people)
        for student in self.character_patterns['students']:
            if student in text:
                start_idx = text.find(student)
                if start_idx != -1:
                    labels[start_idx] = 'B-PER'
                    for i in range(start_idx + 1, start_idx + len(student)):
                        if i < len(labels):
                            labels[i] = 'I-PER'
        
        for teacher in self.character_patterns['teachers']:
            if teacher in text:
                start_idx = text.find(teacher)
                if start_idx != -1:
                    labels[start_idx] = 'B-PER'
                    for i in range(start_idx + 1, start_idx + len(teacher)):
                        if i < len(labels):
                            labels[i] = 'I-PER'
        
        # Label roles
        for role in self.character_patterns['roles']:
            if role in text:
                start_idx = text.find(role)
                if start_idx != -1:
                    labels[start_idx] = 'B-ROLE'
                    for i in range(start_idx + 1, start_idx + len(role)):
                        if i < len(labels):
                            labels[i] = 'I-ROLE'
        
        # Label actions
        for action in self.character_patterns['actions']:
            if action in text:
                start_idx = text.find(action)
                if start_idx != -1:
                    labels[start_idx] = 'B-ACTION'
                    for i in range(start_idx + 1, start_idx + len(action)):
                        if i < len(labels):
                            labels[i] = 'I-ACTION'
        
        # Label attributes
        for attr in self.character_patterns['attributes']:
            if attr in text:
                start_idx = text.find(attr)
                if start_idx != -1:
                    labels[start_idx] = 'B-ATTR'
                    for i in range(start_idx + 1, start_idx + len(attr)):
                        if i < len(labels):
                            labels[i] = 'I-ATTR'
        
        return labels
    
    def get_real_world_examples(self) -> List[Dict]:
        """Get manually annotated real-world examples"""
        
        examples = [
            {
                'text': '小明在課堂上積極發言，老師很欣賞他的表現。',
                'labels': ['B-PER', 'I-PER', 'O', 'B-ROLE', 'I-ROLE', 'O', 'O', 'O', 'B-ACTION', 'I-ACTION', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
            },
            {
                'text': '王老師教導學生要誠實守信。',
                'labels': ['B-PER', 'I-PER', 'I-PER', 'B-ACTION', 'I-ACTION', 'B-ROLE', 'I-ROLE', 'O', 'B-ATTR', 'I-ATTR', 'B-ATTR', 'I-ATTR', 'O']
            },
            {
                'text': '小華和小美一起完成作業。',
                'labels': ['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-ACTION', 'I-ACTION', 'O', 'O', 'O']
            }
        ]
        
        return examples


class BertFineTuner:
    """Fine-tuning manager for Chinese BERT models"""
    
    def __init__(self, 
                 model_name: str = "bert-base-chinese",
                 output_dir: str = "./chinese_ner_model",
                 use_wandb: bool = False):
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        # Initialize model
        self.model = MultiTaskBertForNER(model_name=model_name)
        self.tokenizer = self.model.tokenizer
        
        # Data generator
        self.corpus_generator = ChineseCorpusGenerator()
        
        if use_wandb:
            wandb.init(project="chinese-character-ner")
    
    def prepare_datasets(self, 
                        train_size: int = 800, 
                        val_size: int = 200) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        
        # Generate data
        texts, labels = self.corpus_generator.generate_training_data(train_size + val_size)
        
        # Split train/val
        train_texts = texts[:train_size]
        train_labels = labels[:train_size]
        val_texts = texts[train_size:]
        val_labels = labels[train_size:]
        
        # Create datasets
        train_dataset = ChineseCharacterDataset(
            train_texts, train_labels, self.tokenizer
        )
        val_dataset = ChineseCharacterDataset(
            val_texts, val_labels, self.tokenizer, 
            label_to_id=train_dataset.label_to_id
        )
        
        return train_dataset, val_dataset
    
    def train(self,
              num_epochs: int = 5,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500,
              save_steps: int = 500):
        """Fine-tune the model"""
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="wandb" if self.use_wandb else None,
            run_name="chinese-ner-finetuning" if self.use_wandb else None,
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (usually -100)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten for metrics
        flat_true_predictions = [item for sublist in true_predictions for item in sublist]
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        
        # Calculate F1 score
        f1 = f1_score(flat_true_labels, flat_true_predictions, average='weighted')
        
        return {
            "f1": f1,
        }
    
    def evaluate_model(self, test_texts: List[str], test_labels: List[List[str]]) -> Dict:
        """Evaluate trained model"""
        
        test_dataset = ChineseCharacterDataset(
            test_texts, test_labels, self.tokenizer
        )
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for i in range(len(test_dataset)):
                batch = test_dataset[i]
                
                # Add batch dimension
                input_ids = batch['input_ids'].unsqueeze(0)
                attention_mask = batch['attention_mask'].unsqueeze(0)
                
                # Get predictions
                outputs = self.model(input_ids, attention_mask)
                
                if 'predictions' in outputs:
                    pred = outputs['predictions'][0]
                else:
                    pred = torch.argmax(outputs['ner_logits'], dim=-1)[0].cpu().numpy()
                
                predictions.append(pred)
                true_labels.append(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        
        # Classification report
        label_names = list(test_dataset.id_to_label.values())
        report = classification_report(
            flat_true_labels, flat_predictions, 
            target_names=label_names, output_dict=True
        )
        
        return report


class CharacterExtractionPipeline:
    """Complete pipeline for character extraction using fine-tuned models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Load fine-tuned model
        self.model = MultiTaskBertForNER.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load label mappings
        with open(os.path.join(model_path, 'label_mapping.json'), 'r') as f:
            label_mapping = json.load(f)
            self.id_to_label = label_mapping['id_to_label']
    
    def extract_characters(self, text: str) -> List[Dict]:
        """Extract characters using fine-tuned model"""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get predictions
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(**encoding)
        
        # Process predictions
        if 'predictions' in outputs:
            predictions = outputs['predictions'][0]
        else:
            predictions = torch.argmax(outputs['ner_logits'], dim=-1)[0].cpu().numpy()
        
        # Convert to characters
        characters = self.predictions_to_characters(text, predictions)
        
        return characters
    
    def predictions_to_characters(self, text: str, predictions: List[int]) -> List[Dict]:
        """Convert predictions to character entities"""
        
        characters = []
        tokens = self.tokenizer.tokenize(text)
        
        current_entity = None
        current_text = ""
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            label = self.id_to_label.get(str(pred_id), 'O')
            
            if label.startswith('B-'):
                # Beginning of new entity
                if current_entity:
                    # Save previous entity
                    characters.append(current_entity)
                
                entity_type = label.split('-')[1]
                current_entity = {
                    'name': token.replace('##', ''),
                    'type': entity_type.lower(),
                    'start_pos': i,
                    'confidence': 0.9,
                    'source': 'bert_finetuned'
                }
                current_text = token.replace('##', '')
                
            elif label.startswith('I-') and current_entity:
                # Inside current entity
                current_text += token.replace('##', '')
                current_entity['name'] = current_text
                
            else:
                # Outside entity or different entity type
                if current_entity:
                    characters.append(current_entity)
                    current_entity = None
                    current_text = ""
        
        # Add final entity if exists
        if current_entity:
            characters.append(current_entity)
        
        return characters


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Fine-tune model
    tuner = BertFineTuner(
        model_name="bert-base-chinese",
        output_dir="./chinese_ner_finetuned",
        use_wandb=False
    )
    
    # Train model
    trainer = tuner.train(num_epochs=3, batch_size=8)
    
    # Test extraction
    pipeline = CharacterExtractionPipeline("./chinese_ner_finetuned")
    
    test_text = "小明是學生，王老師很欣賞他的努力學習態度。"
    characters = pipeline.extract_characters(test_text)
    
    print("Extracted characters:")
    for char in characters:
        print(f"Name: {char['name']}, Type: {char['type']}, Confidence: {char['confidence']}")