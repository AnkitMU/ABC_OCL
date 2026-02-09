#!/usr/bin/env python3
"""
CodeBERT Training Runner
Generates 5000 OCL examples and trains CodeBERT classifier
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ssr_ocl.classifiers.codebert import (
    save_training_data,
    train_codebert_classifier
)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧠 CodeBERT OCL Pattern Classifier Training")
    print("="*80)
    
    # Step 1: Generate training data
    print("\nSTEP 1: Generating 5000 OCL training examples...")
    json_file = "ocl_training_data.json"
    save_training_data(json_file)
    
    # Step 2: Train CodeBERT
    print("\nSTEP 2: Training CodeBERT classifier...")
    train_codebert_classifier(
        json_file=json_file,
        output_dir="models/codebert_ocl_classifier"
    )
    
    print("\n CodeBERT training complete!")
    print(f"Model saved to: models/codebert_ocl_classifier")
