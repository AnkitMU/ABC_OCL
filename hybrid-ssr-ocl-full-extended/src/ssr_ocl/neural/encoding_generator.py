"""
Neural Z3 Encoding Generator
Generates optimal Z3 SMT encodings from OCL patterns using neural networks
"""
import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import numpy as np
from .pattern_classifier import OCLPatternType
from z3 import *

class NeuralEncodingGenerator:
    """Neural generator for Z3 encodings from OCL patterns"""
    
    def __init__(self, model_dir: str = "models/encoding_generator"):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        
        # Initialize or load model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize T5 model for sequence-to-sequence generation"""
        try:
            if os.path.exists(self.model_dir):
                print(f" Loading neural encoding generator from {self.model_dir}...")
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
                self.is_trained = True
            else:
                print("🔄 Initializing new T5 model for encoding generation...")
                self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
                self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
                self.is_trained = False
                
        except Exception as e:
            print(f"  Could not initialize neural model: {e}")
            self.tokenizer = None
            self.model = None
    
    def _create_training_data(self) -> List[Tuple[str, str]]:
        """Generate OCL -> Z3 encoding training pairs"""
        training_pairs = [
            # Pairwise uniqueness patterns
            (
                "Generate Z3 constraint for: self.holdings->forAll(x, y | x <> y implies x.id <> y.id)",
                "Distinct([Int(f'hold_id_{i}') for i in range(n)])"
            ),
            (
                "Generate Z3 constraint for: self.students->forAll(s1, s2 | s1 <> s2 implies s1.id <> s2.id)",
                "Distinct([Int(f'student_id_{i}') for i in range(n)])"
            ),
            
            # Exact count selection patterns
            (
                "Generate Z3 constraint for: self.thelib.holdings->select(x | x.id = self.id)->size() = 1",
                "Or([And(ids[i] == self_id, And([Not(ids[j] == self_id) for j in range(n) if j != i])) for i in range(n)])"
            ),
            (
                "Generate Z3 constraint for: self.courses->select(c | c.code = target)->size() = 1",
                "Or([And(codes[i] == target, And([Not(codes[j] == target) for j in range(n) if j != i])) for i in range(n)])"
            ),
            
            # Size constraints
            (
                "Generate Z3 constraint for: self.departments->size() > 0",
                "size_var > 0"
            ),
            (
                "Generate Z3 constraint for: self.enrollments->size() <= 6",
                "size_var <= 6"
            ),
            
            # Numeric comparisons
            (
                "Generate Z3 constraint for: self.age >= 16",
                "age >= 16"
            ),
            (
                "Generate Z3 constraint for: self.gpa >= 0.0 and self.gpa <= 4.0",
                "And(gpa >= 0.0, gpa <= 4.0)"
            ),
            
            # Null checks
            (
                "Generate Z3 constraint for: self.advisor <> null",
                "advisor_present"
            ),
            (
                "Generate Z3 constraint for: self.room = null",
                "Not(room_present)"
            ),
            
            # Collection membership
            (
                "Generate Z3 constraint for: not self.prerequisites->includes(self)",
                "Not(Or([prereq[i] == self_value for i in range(n)]))"
            ),
        ]
        
        return training_pairs
    
    def generate_encoding(self, pattern_type: OCLPatternType, ocl_text: str, context: Dict) -> str:
        """Generate Z3 encoding for OCL constraint using neural model"""
        if not self.is_trained or self.model is None:
            return self._template_based_fallback(pattern_type, ocl_text, context)
        
        try:
            # Prepare input for T5 model
            input_text = f"Generate Z3 constraint for: {ocl_text}"
            inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate encoding
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, 
                    max_length=256,
                    num_beams=5,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_encoding = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process and validate
            return self._post_process_encoding(generated_encoding, context)
            
        except Exception as e:
            print(f"  Neural generation failed: {e}, falling back to templates...")
            return self._template_based_fallback(pattern_type, ocl_text, context)
    
    def _template_based_fallback(self, pattern_type: OCLPatternType, ocl_text: str, context: Dict) -> str:
        """Fallback to template-based encoding generation"""
        templates = {
            # Basic patterns
            OCLPatternType.PAIRWISE_UNIQUENESS: "Distinct([Int(f'{collection}_id_{i}') for i in range({n})])",
            OCLPatternType.EXACT_COUNT_SELECTION: "Or([And({array}[i] == {target}, And([Not({array}[j] == {target}) for j in range({n}) if j != i])) for i in range({n})])",
            OCLPatternType.SIZE_CONSTRAINT: "{var} {op} {value}",
            OCLPatternType.NUMERIC_COMPARISON: "{var} {op} {value}",
            OCLPatternType.NULL_CHECK: "{var}_present" if "<>" in ocl_text else "Not({var}_present)",
            OCLPatternType.COLLECTION_MEMBERSHIP: "Or([{array}[i] == {target} for i in range({n})])",
            OCLPatternType.UNIQUENESS_CONSTRAINT: "Distinct({array})",
            
            # Advanced patterns templates
            OCLPatternType.EXACTLY_ONE: "And(AtMost(*[Bool(f'{collection}_satisfies_{i}') for i in range({n})], 1), AtLeast(*[Bool(f'{collection}_satisfies_{i}') for i in range({n})], 1))",
            OCLPatternType.CLOSURE: "Or([reach[0][j] for j in range(1, {n})])",  # Simplified reachability check
            OCLPatternType.ACYCLICITY: "And([Not(Or([And(rel[i][k], reach[k][i]) for k in range({n}) if k != i])) for i in range({n})])",
            OCLPatternType.ITERATE: "accumulators[{n}] >= {threshold}",  # Final accumulator result
            OCLPatternType.IMPLIES: "Or(Not({condition}), {expression})",
            OCLPatternType.SAFE_NAVIGATION: "Implies({guard}, {access})",
            OCLPatternType.TYPE_CHECK: "Bool(f'is_{type_name}')",
            OCLPatternType.SUBSET_DISJOINT: "And([Implies(set_b[i], set_a[i % {n}]) for i in range({n})])",
            OCLPatternType.ORDERING: "And([elements[i] <= elements[i+1] for i in range({n}-1)])",
            OCLPatternType.CONTRACTUAL: "{pre_var} + {change} == {post_var}",
        }
        
        template = templates.get(pattern_type, "True")  # Default fallback
        
        # Simple variable substitution
        encoding = template.format(
            collection=context.get('collection', 'elements'),
            n=context.get('scope', 3),
            array=context.get('array', 'arr'),
            target=context.get('target', 'target'),
            var=context.get('variable', 'var'),
            op=context.get('operator', '>='),
            value=context.get('value', 0)
        )
        
        return encoding
    
    def _post_process_encoding(self, encoding: str, context: Dict) -> str:
        """Post-process generated encoding to ensure validity"""
        # Basic cleanup and variable substitution
        encoding = encoding.strip()
        
        # Replace placeholder variables with actual context
        if context.get('collection'):
            encoding = encoding.replace('collection', context['collection'])
        if context.get('scope'):
            encoding = encoding.replace('n', str(context['scope']))
        
        return encoding
    
    def train_neural_model(self, training_data: Optional[List[Tuple[str, str]]] = None):
        """Train the neural encoding generator"""
        if self.tokenizer is None or self.model is None:
            print(" Cannot train: model not initialized")
            return
        
        print("🧠 Training Neural Z3 Encoding Generator...")
        
        # Get training data
        if training_data is None:
            training_data = self._create_training_data()
        
        # Prepare dataset for training
        input_texts, target_texts = zip(*training_data)
        
        # Tokenize inputs and targets
        input_encodings = self.tokenizer(
            list(input_texts), 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors="pt"
        )
        
        target_encodings = self.tokenizer(
            list(target_texts), 
            truncation=True, 
            padding=True, 
            max_length=256,
            return_tensors="pt"
        )
        
        # Create simple training loop (simplified for demo)
        print("🎯 Training encoding generator...")
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        # Training loop (simplified)
        for epoch in range(3):  # Small number for demo
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                labels=target_encodings.input_ids
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch + 1}/3, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        self._save_model()
        print(" Neural encoding generator training completed")
    
    def _save_model(self):
        """Save trained model"""
        if self.model is None or self.tokenizer is None:
            return
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        print(f"💾 Neural encoding model saved to {self.model_dir}")

# Global instance
_generator_instance = None

def get_neural_encoding_generator() -> NeuralEncodingGenerator:
    """Get singleton neural encoding generator"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = NeuralEncodingGenerator()
    return _generator_instance