"""
Advanced ML Module: Transfer Learning + LoRA + Sequential Fine-tuning
實施高級機器學習技術來提升中文 NER 模型性能
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import optuna
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import jieba.posseg as pseg
import re

@dataclass
class NERConfig:
    """NER模型配置"""
    model_name: str = "hfl/chinese-bert-wwm-ext"  # 中文BERT預訓練模型
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

class ChineseNERModel:
    """中文NER模型 - 使用Transfer Learning + LoRA"""
    
    def __init__(self, config: NERConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.label2id = {"O": 0, "B-PER": 1, "I-PER": 2}
        self.id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initialize_model(self):
        """初始化預訓練模型並應用LoRA"""
        try:
            # 載入預訓練模型和tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            base_model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_name,
                num_labels=len(self.label2id),
                label2id=self.label2id,
                id2label=self.id2label
            )
            
            # 配置LoRA
            lora_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["query", "key", "value", "dense"]
            )
            
            # 應用LoRA適配器
            self.model = get_peft_model(base_model, lora_config)
            self.model.to(self.device)
            
            print(f"✅ 模型初始化完成 - 使用設備: {self.device}")
            print(f"📊 LoRA參數: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
            
        except Exception as e:
            print(f"❌ 模型初始化失敗: {e}")
            # Fallback to traditional approach
            self.use_traditional_approach = True
    
    def prepare_training_data(self, annotated_texts: List[Dict]) -> Dataset:
        """準備訓練數據"""
        tokenized_inputs = []
        labels = []
        
        for example in annotated_texts:
            text = example['text']
            entities = example['entities']
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            
            # Create BIO labels
            token_labels = self.create_bio_labels(text, entities, tokenized['offset_mapping'][0])
            
            tokenized_inputs.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': torch.tensor(token_labels)
            })
        
        return Dataset.from_list(tokenized_inputs)
    
    def create_bio_labels(self, text: str, entities: List[Dict], offset_mapping) -> List[int]:
        """創建BIO標籤序列"""
        labels = [0] * len(offset_mapping)  # 初始化為O
        
        for entity in entities:
            start, end, label = entity['start'], entity['end'], entity['label']
            if label == 'PERSON':
                # 找到對應的token位置
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start >= start and token_end <= end:
                        if token_start == start:
                            labels[i] = 1  # B-PER
                        else:
                            labels[i] = 2  # I-PER
        
        return labels
    
    def hyperparameter_search(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict:
        """使用Optuna進行超參數搜索"""
        def objective(trial):
            # 搜索超參數空間
            lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            lora_r = trial.suggest_int('lora_r', 4, 16)
            lora_alpha = trial.suggest_int('lora_alpha', 16, 64)
            
            # 更新配置
            self.config.learning_rate = lr
            self.config.batch_size = batch_size
            self.config.lora_r = lora_r
            self.config.lora_alpha = lora_alpha
            
            # 重新初始化模型
            self.initialize_model()
            
            # 訓練模型
            trainer = self.create_trainer(train_dataset, val_dataset)
            trainer.train()
            
            # 評估性能
            eval_results = trainer.evaluate()
            return eval_results['eval_f1']
        
        # 運行超參數搜索
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        return study.best_params
    
    def cross_validation_training(self, dataset: Dataset, k_folds: int = 5) -> Dict:
        """K折交叉驗證訓練"""
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"🔄 開始第 {fold + 1}/{k_folds} 折訓練...")
            
            # 分割數據
            train_subset = dataset.select(train_idx)
            val_subset = dataset.select(val_idx)
            
            # 重新初始化模型（避免權重污染）
            self.initialize_model()
            
            # 創建訓練器
            trainer = self.create_trainer(train_subset, val_subset)
            
            # 訓練
            trainer.train()
            
            # 評估
            eval_results = trainer.evaluate()
            cv_scores.append(eval_results['eval_f1'])
            
            print(f"✅ 第 {fold + 1} 折 F1 分數: {eval_results['eval_f1']:.4f}")
        
        return {
            'mean_f1': np.mean(cv_scores),
            'std_f1': np.std(cv_scores),
            'all_scores': cv_scores
        }
    
    def create_trainer(self, train_dataset: Dataset, val_dataset: Dataset) -> Trainer:
        """創建Trainer實例"""
        training_args = TrainingArguments(
            output_dir='./model_output',
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None  # 禁用wandb
        )
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
    
    def compute_metrics(self, eval_pred):
        """計算評估指標"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # 移除特殊token
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:  # 忽略填充token
                    true_predictions.append(self.id2label[pred_id])
                    true_labels.append(self.id2label[label_id])
        
        # 計算精確率、召回率、F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average='weighted'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def sequential_fine_tuning(self, datasets: List[Dataset]) -> None:
        """序列式微調 - 逐步適應不同領域"""
        for i, dataset in enumerate(datasets):
            print(f"🎯 開始第 {i + 1} 階段序列式微調...")
            
            # 調整學習率（遞減策略）
            self.config.learning_rate *= 0.8
            
            # 分割訓練/驗證集
            train_size = int(0.8 * len(dataset))
            train_subset = dataset.select(range(train_size))
            val_subset = dataset.select(range(train_size, len(dataset)))
            
            # 訓練
            trainer = self.create_trainer(train_subset, val_subset)
            trainer.train()
            
            print(f"✅ 第 {i + 1} 階段完成，學習率: {self.config.learning_rate}")
    
    def predict_entities(self, text: str) -> List[Dict]:
        """預測文本中的實體"""
        if not self.model or not self.tokenizer:
            return self.fallback_prediction(text)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_offsets_mapping=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'offset_mapping'}
            offset_mapping = inputs.pop('offset_mapping', None)
            
            # 預測
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # 解析實體
            entities = self.extract_entities_from_predictions(
                text, predictions[0], offset_mapping[0]
            )
            
            return entities
            
        except Exception as e:
            print(f"⚠️ ML預測失敗，使用後備方案: {e}")
            return self.fallback_prediction(text)
    
    def extract_entities_from_predictions(self, text: str, predictions, offset_mapping) -> List[Dict]:
        """從預測結果提取實體"""
        entities = []
        current_entity = None
        
        for i, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            label = self.id2label[pred_id.item()]
            
            if label == "B-PER":  # 開始新實體
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'start': start.item(),
                    'end': end.item(),
                    'text': text[start:end],
                    'label': 'PERSON',
                    'confidence': 0.9
                }
            elif label == "I-PER" and current_entity:  # 繼續實體
                current_entity['end'] = end.item()
                current_entity['text'] = text[current_entity['start']:current_entity['end']]
            else:  # O標籤
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最後一個實體
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def fallback_prediction(self, text: str) -> List[Dict]:
        """後備預測方案（基於規則）"""
        entities = []
        
        # 使用jieba進行詞性標註
        words = pseg.cut(text)
        pos = 0
        
        for word, flag in words:
            if flag in ['nr', 'nrt'] and len(word) >= 2:
                entities.append({
                    'start': pos,
                    'end': pos + len(word),
                    'text': word,
                    'label': 'PERSON',
                    'confidence': 0.7
                })
            pos += len(word)
        
        return entities

# 全局模型實例
ml_model = None

def initialize_ml_model():
    """初始化ML模型"""
    global ml_model
    try:
        config = NERConfig()
        ml_model = ChineseNERModel(config)
        ml_model.initialize_model()
        return True
    except Exception as e:
        print(f"❌ ML模型初始化失敗: {e}")
        return False

def get_ml_predictions(text: str) -> List[Dict]:
    """獲取ML模型預測結果"""
    global ml_model
    if ml_model:
        return ml_model.predict_entities(text)
    else:
        # 如果ML模型未初始化，返回空列表
        return []