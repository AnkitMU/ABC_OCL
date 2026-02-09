#!/usr/bin/env python3
"""
SentenceTransformer-based OCL Pattern Classifier
Uses all-MiniLM-L6-v2 model for embedding and LogisticRegression for classification
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
from enum import Enum


class OCLPatternType(Enum):
    """All 50 OCL pattern types"""
    # Basic patterns (1-9)
    PAIRWISE_UNIQUENESS = "pairwise_uniqueness"
    EXACT_COUNT_SELECTION = "exact_count_selection"
    GLOBAL_COLLECTION = "global_collection"
    SET_INTERSECTION = "set_intersection"
    SIZE_CONSTRAINT = "size_constraint"
    UNIQUENESS_CONSTRAINT = "uniqueness_constraint"
    COLLECTION_MEMBERSHIP = "collection_membership"
    NULL_CHECK = "null_check"
    NUMERIC_COMPARISON = "numeric_comparison"
    
    # Advanced patterns (10-19)
    EXACTLY_ONE = "exactly_one"
    CLOSURE = "closure_transitive"
    ACYCLICITY = "acyclicity"
    ITERATE = "aggregation_iterate"
    IMPLIES = "boolean_guard_implies"
    SAFE_NAVIGATION = "safe_navigation"
    TYPE_CHECK = "type_check_casting"
    SUBSET_DISJOINT = "subset_disjointness"
    ORDERING = "ordering_ranking"
    CONTRACTUAL = "contractual_temporal"
    
    # Collection Operations (20-27)
    SELECT_REJECT = "select_reject"
    COLLECT_FLATTEN = "collect_flatten"
    ANY_OPERATION = "any_operation"
    FOR_ALL_NESTED = "forall_nested"
    EXISTS_NESTED = "exists_nested"
    COLLECT_NESTED = "collect_nested"
    AS_SET_AS_BAG = "as_set_as_bag"
    SUM_PRODUCT = "sum_product"
    
    # String Operations (28-31)
    STRING_CONCAT = "string_concat"
    STRING_OPERATIONS = "string_operations"
    STRING_COMPARISON = "string_comparison"
    STRING_PATTERN = "string_pattern"
    
    # Arithmetic & Logic (32-36)
    ARITHMETIC_EXPRESSION = "arithmetic_expression"
    DIV_MOD_OPERATIONS = "div_mod_operations"
    ABS_MIN_MAX = "abs_min_max"
    BOOLEAN_OPERATIONS = "boolean_operations"
    IF_THEN_ELSE = "if_then_else"
    
    # Tuple & Let (37-39)
    TUPLE_LITERAL = "tuple_literal"
    LET_EXPRESSION = "let_expression"
    LET_NESTED = "let_nested"
    
    # Set Operations (40-43)
    UNION_INTERSECTION = "union_intersection"
    SYMMETRIC_DIFFERENCE = "symmetric_difference"
    INCLUDING_EXCLUDING = "including_excluding"
    FLATTEN_OPERATION = "flatten_operation"
    
    # Navigation & Property (44-47)
    NAVIGATION_CHAIN = "navigation_chain"
    OPTIONAL_NAVIGATION = "optional_navigation"
    COLLECTION_NAVIGATION = "collection_navigation"
    SHORTHAND_NOTATION = "shorthand_notation"
    
    # OCL Standard Library (48-50)
    OCL_IS_UNDEFINED = "ocl_is_undefined"
    OCL_IS_INVALID = "ocl_is_invalid"
    OCL_AS_TYPE = "ocl_as_type"
    
    UNKNOWN = "unknown"


class SentenceTransformerClassifier:
    """SentenceTransformer-based OCL pattern classifier"""
    
    def __init__(self, model_dir: str = "models/sentence_transformer_classifier"):
        self.model_dir = model_dir
        self.encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = LogisticRegression(
            C=100.0,  # Higher C = less regularization = better fitting for large dataset
            solver='lbfgs',  # Better for multi-class
            max_iter=10000,  # More iterations for convergence
            random_state=42,
            multi_class='multinomial',
            verbose=0,
            class_weight='balanced',  # Handle any class imbalance
            warm_start=False
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Try to load pre-trained model
        self._load_model()
    
    def train(self, training_data: List[Tuple[str, str]] = None):
        """Train classifier on OCL examples"""
        if training_data is None:
            training_data = self._create_default_training_data()
        
        texts = [ocl_text for ocl_text, _ in training_data]
        patterns = [pattern for _, pattern in training_data]
        
        print(f"🧠 SentenceTransformer Classifier Training")
        print(f"{'='*60}")
        print(f"Examples: {len(texts)}")
        print(f"Patterns: {len(set(patterns))}")
        
        # Generate embeddings
        print(f"\n Generating embeddings with all-MiniLM-L6-v2...")
        embeddings = self.encoder_model.encode(texts, show_progress_bar=True)
        print(f" Embedding shape: {embeddings.shape}")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(patterns)
        
        # Train classifier
        print(f"\n🎯 Training LogisticRegression...")
        self.classifier.fit(embeddings, encoded_labels)
        
        # Evaluate
        accuracy = self.classifier.score(embeddings, encoded_labels)
        print(f" Training accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        self._save_model()
        
        print(f"\n{'='*60}")
        print(f"🎉 Training complete!")
        print(f"Model saved to: {self.model_dir}")
    
    def predict(self, ocl_text: str) -> Tuple[str, float]:
        """Predict pattern for OCL constraint"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        embedding = self.encoder_model.encode([ocl_text])
        prediction = self.classifier.predict(embedding)[0]
        probabilities = self.classifier.predict_proba(embedding)[0]
        
        max_prob = np.max(probabilities)
        pattern = self.label_encoder.inverse_transform([prediction])[0]
        
        return pattern, max_prob
    
    def _create_default_training_data(self) -> List[Tuple[str, str]]:
        """Default training data if none provided"""
        return [
            ("self.students->forAll(x, y | x <> y implies x.id <> y.id)", "pairwise_uniqueness"),
            ("self.thelib.holdings->select(x | x.id = self.id)->size() = 1", "exact_count_selection"),
            ("Copy.allInstances()->collect(id)->select(i | i = self.id)->size() = 1", "global_collection"),
        ]
    
    def _save_model(self):
        """Save trained model"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        with open(os.path.join(self.model_dir, "classifier.pkl"), "wb") as f:
            pickle.dump(self.classifier, f)
        
        with open(os.path.join(self.model_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        metadata = {
            "model": "all-MiniLM-L6-v2",
            "patterns": len(self.label_encoder.classes_)
        }
        with open(os.path.join(self.model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_model(self):
        """Load pre-trained model if available"""
        try:
            clf_path = os.path.join(self.model_dir, "classifier.pkl")
            enc_path = os.path.join(self.model_dir, "label_encoder.pkl")
            
            if os.path.exists(clf_path) and os.path.exists(enc_path):
                with open(clf_path, "rb") as f:
                    self.classifier = pickle.load(f)
                with open(enc_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                self.is_trained = True
                print(f" Loaded SentenceTransformer model from {self.model_dir}")
        except Exception as e:
            print(f"  Could not load model: {e}")
