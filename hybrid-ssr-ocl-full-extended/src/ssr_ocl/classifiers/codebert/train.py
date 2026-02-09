#!/usr/bin/env python3
"""
Train CodeBERT Classifier on 5000 OCL Pattern Examples

Trains a CodeBERT-based classifier for all 50 OCL patterns
using 5000 training examples (100 per pattern)
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm

# Device selection
device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)


class OCLDataset(Dataset):
    """Dataset for OCL pattern classification"""
    
    def __init__(self, texts: List[str], pattern_labels: List[str], tokenizer, max_length=512):
        self.texts = texts
        self.pattern_labels = pattern_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        unique_labels = sorted(list(set(pattern_labels)))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.label_ids = [self.label2id[label] for label in pattern_labels]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_id = self.label_ids[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }


class CodeBERTClassifier(nn.Module):
    """CodeBERT-based classifier for OCL patterns"""
    
    def __init__(self, num_labels: int):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    return accuracy


def load_training_data_from_json(json_file: str) -> List[Tuple[str, str]]:
    """
    Load training data from JSON file
    
    Args:
        json_file: Path to JSON file with training examples
    
    Returns:
        List of (ocl_text, pattern) tuples
    """
    import json
    
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Training data file not found: {json_file}")
    
    print(f"📂 Loading training data from {json_file}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract examples
    examples = [
        (item['ocl_text'], item['pattern'])
        for item in data['examples']
    ]
    
    print(f" Loaded {len(examples)} examples from {json_file}")
    print(f"   Source: {data['metadata'].get('source', 'unknown')}")
    
    return examples


def train_codebert_classifier(training_data: List[Tuple[str, str]] = None, 
                             json_file: str = None,
                             output_dir: str = "models/codebert_ocl_classifier"):
    """
    Train CodeBERT classifier
    
    Args:
        training_data: List of (ocl_text, pattern) tuples (or None to use json_file)
        json_file: Path to JSON file with training examples (alternative to training_data)
        output_dir: Where to save the trained model
    """
    print("\n" + "="*80)
    print("🧠 CodeBERT OCL Pattern Classifier Training")
    print("="*80)
    
    # Load training data
    if json_file is not None:
        training_data = load_training_data_from_json(json_file)
        print(f"\n Loaded training data from JSON file")
    elif training_data is None:
        raise ValueError("Either training_data or json_file must be provided")
    else:
        print(f"\n Using provided training data")
    
    # Configuration
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512
    
    print(f"\n Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Max length: {MAX_LENGTH}")
    print(f"   Device: {device}")
    
    # Load tokenizer
    print("\n Loading CodeBERT...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    print(f" Tokenizer loaded")
    
    # Extract texts and labels
    texts = [ocl_text for ocl_text, _ in training_data]
    pattern_labels = [pattern for _, pattern in training_data]
    
    print(f"\n Training data:")
    print(f"   Examples: {len(texts)}")
    print(f"   Patterns: {len(set(pattern_labels))}")
    
    # Create dataset
    print(f"\n🔄 Creating dataset...")
    dataset = OCLDataset(texts, pattern_labels, tokenizer, max_length=MAX_LENGTH)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f" Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print(f"\n🤖 Initializing CodeBERT classifier...")
    num_labels = len(dataset.label2id)
    model = CodeBERTClassifier(num_labels=num_labels)
    model = model.to(device)
    print(f" Model ready for {num_labels} patterns")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n🚀 Starting training...\n")
    
    best_accuracy = 0
    training_history = []
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")
        
        val_accuracy = evaluate(model, val_dataloader, device)
        print(f"Val accuracy: {val_accuracy:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy
        })
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f" New best accuracy: {best_accuracy:.4f}")
            
            model_dir = Path(output_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), model_dir / "model.pt")
            
            # Save tokenizer
            tokenizer.save_pretrained(model_dir)
            
            # Save label mappings
            label_mapping = {
                'label2id': dataset.label2id,
                'id2label': dataset.id2label
            }
            with open(model_dir / "label_mapping.json", 'w') as f:
                json.dump(label_mapping, f, indent=2)
            
            print(f"💾 Saved to {model_dir}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("📈 Final Evaluation")
    print(f"{'='*60}")
    
    final_accuracy = evaluate(model, val_dataloader, device)
    print(f"Final validation accuracy: {final_accuracy:.4f}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Training time: {training_time:.2f}s ({training_time/60:.2f}m)")
    
    # Save summary
    model_dir = Path(output_dir)
    summary = {
        "model": "CodeBERT ('microsoft/codebert-base')",
        "task": "OCL Pattern Classification",
        "num_patterns": num_labels,
        "training_examples": len(texts),
        "examples_per_pattern": 100,
        "train_set_size": len(train_dataset),
        "val_set_size": len(val_dataset),
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "device": str(device),
        "final_validation_accuracy": float(final_accuracy),
        "best_validation_accuracy": float(best_accuracy),
        "training_time_seconds": float(training_time),
        "training_history": training_history,
        "model_directory": str(model_dir)
    }
    
    summary_file = model_dir / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Model: CodeBERT ('microsoft/codebert-base')")
    print(f"Patterns: {num_labels}")
    print(f"Validation Accuracy: {final_accuracy:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Model Location: {model_dir}")
    print(f"{'='*80}\n")
    
    return model_dir
