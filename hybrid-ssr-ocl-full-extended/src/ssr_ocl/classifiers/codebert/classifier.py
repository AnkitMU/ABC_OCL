#!/usr/bin/env python3
"""
CodeBERT Classifier for OCL Pattern Classification

Standalone classifier using:
- CodeBERT ('microsoft/codebert-base') as encoder
- Fine-tuned classification head
- All 50 OCL patterns
- 100 examples per pattern
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class CodeBERTClassifier(nn.Module):
    """CodeBERT-based classifier for OCL patterns"""
    
    def __init__(self, num_labels: int):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)  # CodeBERT is 768-dim
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}


class CodeBERTOCLClassifier:
    """Inference wrapper for CodeBERT OCL classifier"""
    
    def __init__(self, model_dir: str = "models/codebert_ocl_classifier"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Load label mapping
        with open(self.model_dir / "label_mapping.json", 'r') as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
        self.label2id = {v: int(k) for k, v in mapping['id2label'].items()}
        
        # Initialize and load model
        num_labels = len(self.id2label)
        self.model = CodeBERTClassifier(num_labels=num_labels)
        
        # Load weights
        state_dict = torch.load(self.model_dir / "model.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, ocl_text: str, return_confidence: bool = True) -> Tuple[str, float, Optional[List]]:
        """
        Predict OCL pattern
        
        Args:
            ocl_text: OCL constraint text
            return_confidence: Whether to return confidence scores
        
        Returns:
            pattern_name: Predicted pattern name
            confidence: Confidence score
            top5: Top 5 predictions (if return_confidence=True)
        """
        # Tokenize
        encoding = self.tokenizer(
            ocl_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)['logits']
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        pattern_name = self.id2label[pred_idx]
        
        if return_confidence:
            # Get top-5 predictions
            top5_probs, top5_idxs = torch.topk(probs[0], k=min(5, len(self.id2label)))
            top5 = [
                (self.id2label[idx.item()], prob.item())
                for prob, idx in zip(top5_probs, top5_idxs)
            ]
            return pattern_name, confidence, top5
        
        return pattern_name, confidence, None
    
    def predict_batch(self, ocl_texts: List[str]) -> List[Dict]:
        """Predict patterns for multiple OCL constraints"""
        results = []
        for ocl_text in ocl_texts:
            pattern, confidence, top5 = self.predict(ocl_text)
            results.append({
                'ocl': ocl_text,
                'pattern': pattern,
                'confidence': confidence,
                'top5': top5
            })
        return results
