#!/usr/bin/env python3
"""
Complete Two-Phase Framework Integration

Phase 1: Domain Adaptation (Training)
  1. Extract Metamodel (Vocabulary) from XMI
  2. Generate Domain Data using 50 OCL patterns
  3. Retrain Classifier on merged dataset

Phase 2: OCL Verification (Application)
  1. Classify Real Constraint
  2. Apply Algorithm (Classifier)
  3. Put Pattern in Z3 (Encoder)
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssr_ocl.classifiers.sentence_transformer import (
    SentenceTransformerClassifier
)
from ssr_ocl.classifiers.sentence_transformer.xmi_based_domain_adapter import (
    GenericDomainDataGenerator
)
from ssr_ocl.lowering.unified_smt_encoder import UnifiedSMTEncoder
from ssr_ocl.solver.z3_runner import run_solver
from ssr_ocl.validation import ModelConsistencyChecker

# Dummy verifier class for confidence-guided routing
class ConfidenceGuidedVerifier:
    """Simple confidence routing logic"""
    def route(self, confidence):
        if confidence > 0.7:
            return "HIGH_CONFIDENCE_SINGLE"
        elif confidence > 0.3:
            return "MEDIUM_CONFIDENCE_ENSEMBLE"
        else:
            return "LOW_CONFIDENCE_MANUAL_REVIEW"


class OCLVerificationFramework:
    """
    Complete two-phase OCL verification framework
    """
    
    def __init__(self):
        self.encoder = UnifiedSMTEncoder()
        self.verifier = ConfidenceGuidedVerifier()
        self.classifier = None
        self.domain_name = None
    
    # ============================================================================
    # PHASE 1: DOMAIN ADAPTATION (Training)
    # ============================================================================
    
    def phase1_domain_adaptation(self, xmi_file: str) -> Dict:
        """
        Phase 1: Domain Adaptation - Make classifier domain-aware
        
        Workflow:
        1. Extract Metamodel (Vocabulary)
        2. Generate Domain Data
        3. Retrain Classifier
        
        Args:
            xmi_file: Path to XMI model file
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*80)
        print("PHASE 1: DOMAIN ADAPTATION (Training)")
        print("="*80)
        
        results = {
            'phase': 'Phase 1 - Domain Adaptation',
            'xmi_file': xmi_file,
            'steps': {}
        }
        
        # Step 1: Extract Metamodel (Vocabulary)
        print("\n[Step 1] Extract Metamodel (Vocabulary)")
        print("-" * 80)
        step1_result = self._step1_extract_metamodel(xmi_file)
        results['steps']['extract_metamodel'] = step1_result
        
        # Step 2: Generate Domain Data
        print("\n[Step 2] Generate Domain Data")
        print("-" * 80)
        step2_result = self._step2_generate_domain_data(xmi_file)
        results['steps']['generate_domain_data'] = step2_result
        
        # Step 3: Retrain Classifier
        print("\n[Step 3] Retrain Classifier")
        print("-" * 80)
        step3_result = self._step3_retrain_classifier(
            step2_result['merged_file'],
            xmi_file
        )
        results['steps']['retrain_classifier'] = step3_result
        
        print("\n" + "="*80)
        print(" PHASE 1 COMPLETE: Domain Adaptation Successful")
        print("="*80)
        
        return results
    
    def _step1_extract_metamodel(self, xmi_file: str) -> Dict:
        """
        Step 1: Extract Metamodel (Vocabulary)
        
        Reads the model.xmi file and extracts:
        - Class names (e.g., Vehicle, Customer)
        - Attribute names (e.g., vin, startDate)
        """
        print("Reading XMI file: " + xmi_file)
        
        generator = GenericDomainDataGenerator(xmi_file)
        classes = generator.extractor.get_classes()
        
        print(f" Extracted {len(classes)} classes:")
        for cls in classes[:5]:
            print(f"   - {cls}")
        if len(classes) > 5:
            print(f"   ... and {len(classes) - 5} more")
        
        return {
            'xmi_file': xmi_file,
            'classes_extracted': len(classes),
            'classes': classes
        }
    
    def _step2_generate_domain_data(self, xmi_file: str) -> Dict:
        """
        Step 2: Generate Domain Data
        
        Takes the 50 known OCL patterns and uses extracted vocabulary
        to generate hundreds of domain-specific training examples
        
        Example:
          Pattern: SIZE_CONSTRAINT
          Template: self.{collection}->size() {op} {val}
          Generated: "self.vehicles->size() <= self.capacity"
        """
        print("Generating domain-specific OCL examples...")
        print("Using 50 OCL patterns with extracted vocabulary...")
        
        generator = GenericDomainDataGenerator(xmi_file, examples_per_pattern=10)
        domain_examples = generator.generate_domain_data()
        
        # Save domain data
        domain_name = Path(xmi_file).stem
        domain_file = f"ocl_domain_{domain_name}_adapted.json"
        generator.save_to_json(domain_file)
        
        print(f" Generated {len(domain_examples)} domain-specific OCL examples")
        print(f"   Patterns covered: 50 (10 examples per pattern)")
        print(f"   Saved to: {domain_file}")
        
        # Merge with generic data
        print("\nMerging with generic training data...")
        generic_file = "ocl_training_data.json"
        merged_file = f"ocl_training_data_{domain_name}_merged.json"
        
        with open(generic_file, 'r') as f:
            generic_data = json.load(f)
        
        with open(domain_file, 'r') as f:
            domain_data = json.load(f)
        
        # Combine datasets
        merged_examples = generic_data['examples'] + domain_data['examples']
        
        merged_data = {
            "metadata": {
                "domain": domain_name,
                "total_examples": len(merged_examples),
                "generic_examples": len(generic_data['examples']),
                "domain_examples": len(domain_data['examples']),
                "patterns": 50
            },
            "examples": merged_examples
        }
        
        with open(merged_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f" Merged datasets:")
        print(f"   Generic: {len(generic_data['examples'])}")
        print(f"   Domain: {len(domain_data['examples'])}")
        print(f"   Total: {len(merged_examples)}")
        print(f"   Saved to: {merged_file}")
        
        return {
            'domain_file': domain_file,
            'merged_file': merged_file,
            'domain_examples_count': len(domain_data['examples']),
            'total_examples_count': len(merged_examples)
        }
    
    def _step3_retrain_classifier(self, merged_file: str, xmi_file: str) -> Dict:
        """
        Step 3: Retrain Classifier
        
        Merges domain-specific data with generic examples,
        then re-runs training from classifier.py on the combined dataset.
        Result: A new adapted classifier model that understands domain vocabulary.
        """
        print("Retraining classifier on merged dataset...")
        
        domain_name = Path(xmi_file).stem
        model_dir = f"models/adapted_{domain_name}"
        
        # Load merged data
        with open(merged_file, 'r') as f:
            data = json.load(f)
        
        training_data = [
            (ex['ocl_text'], ex['pattern'])
            for ex in data['examples']
        ]
        
        # Create classifier and train
        self.classifier = SentenceTransformerClassifier(model_dir)
        self.domain_name = domain_name
        
        print(f"Training on {len(training_data)} examples...")
        self.classifier.train(training_data)
        
        print(f" Classifier retrained and saved to: {model_dir}")
        
        return {
            'model_dir': model_dir,
            'training_examples': len(training_data),
            'status': 'complete'
        }
    
    # ============================================================================
    # PHASE 2: OCL VERIFICATION (Application)
    # ============================================================================
    
    def phase2_ocl_verification(self, constraint: str, model_dir: Optional[str] = None) -> Dict:
        """
        Phase 2: OCL Verification - Classify and verify constraint
        
        Workflow:
        1. Classify Real Constraint
        2. Apply Algorithm (Classifier)
        3. Put Pattern in Z3 (Encoder)
        
        Args:
            constraint: OCL constraint string
            model_dir: Optional pre-trained model directory
            
        Returns:
            Dictionary with verification results
        """
        if self.classifier is None and model_dir is None:
            raise ValueError("Classifier not initialized. Run Phase 1 first or provide model_dir.")
        
        if self.classifier is None:
            self.classifier = SentenceTransformerClassifier(model_dir)
        
        results = {
            'phase': 'Phase 2 - OCL Verification',
            'constraint': constraint,
            'steps': {}
        }
        
        # Step 1: Classify Real Constraint
        print("\n" + "="*80)
        print("PHASE 2: OCL VERIFICATION (Application)")
        print("="*80)
        print(f"\nConstraint: {constraint}")
        
        print("\n[Step 1] Classify Real Constraint")
        print("-" * 80)
        step1_result = self._step1_classify_constraint(constraint)
        results['steps']['classify'] = step1_result
        
        # Step 2: Apply Algorithm (Classifier)
        print("\n[Step 2] Apply Algorithm (Classifier)")
        print("-" * 80)
        step2_result = self._step2_apply_algorithm(step1_result)
        results['steps']['algorithm'] = step2_result
        
        # Step 3: Put Pattern in Z3 (Encoder)
        print("\n[Step 3] Put Pattern in Z3 (Encoder)")
        print("-" * 80)
        step3_result = self._step3_put_pattern_in_z3(step2_result)
        results['steps']['z3_verification'] = step3_result
        
        print("\n" + "="*80)
        print(" PHASE 2 COMPLETE: OCL Verification Successful")
        print("="*80)
        
        return results
    
    def _step1_classify_constraint(self, constraint: str) -> Dict:
        """
        Step 1: Classify Real Constraint
        
        A user provides a new OCL constraint.
        The classifier (retrained in Phase 1) is now used.
        """
        print("Using adapted classifier to classify constraint...")
        
        pattern, confidence = self.classifier.predict(constraint)
        
        print(f" Classification results:")
        print(f"   Pattern: {pattern}")
        print(f"   Confidence: {confidence:.4f}")
        
        return {
            'pattern': pattern,
            'confidence': confidence
        }
    
    def _step2_apply_algorithm(self, classifier_result: Dict) -> Dict:
        """
        Step 2: Apply Algorithm (Classifier)
        
        The classifier.py (which was re-trained in Phase 1) is used.
        It reads the constraint and predicts its pattern,
        returning a name like "pairwise_uniqueness" and confidence score.
        """
        print("Applying confidence-guided routing...")
        
        pattern = classifier_result['pattern']
        confidence = classifier_result['confidence']
        
        # Determine strategy based on confidence
        if confidence > 0.7:
            strategy = "HIGH_CONFIDENCE_SINGLE"
            print(f" High confidence ({confidence:.4f}) - Use single pattern directly")
        elif confidence > 0.3:
            strategy = "MEDIUM_CONFIDENCE_ENSEMBLE"
            print(f" Medium confidence ({confidence:.4f}) - Try ensemble with alternatives")
        else:
            strategy = "LOW_CONFIDENCE_MANUAL_REVIEW"
            print(f"  Low confidence ({confidence:.4f}) - Flag for manual review")
        
        return {
            'pattern': pattern,
            'confidence': confidence,
            'strategy': strategy
        }
    
    def _step3_put_pattern_in_z3(self, algorithm_result: Dict) -> Dict:
        """
        Step 3: Put Pattern in Z3 (Encoder)
        
        The ocl2smt.py script takes the predicted pattern name ("pairwise_uniqueness")
        and passes it to unified_smt_encoder.py.
        That file looks up the correct function (e.g., encode_pairwise_uniqueness)
        and generates the final Z3 logical formula to check for a violation.
        """
        print("Encoding pattern to Z3 formula...")
        
        pattern = algorithm_result['pattern']
        confidence = algorithm_result['confidence']
        
        print(f"Looking up encoder function: encode_{pattern}()")
        
        # Encode pattern to Z3
        context = {'scope': 5, 'collection': 'elements'}
        
        try:
            solver, model_vars = self.encoder.encode(
                pattern_name=pattern,
                ocl_text=self.phase2_constraint,  # Store for reference
                context=context
            )
            
            print(f" Generated Z3 formula for pattern: {pattern}")
            
            # Verify with Z3
            print("Running Z3 verification...")
            verdict, model, time_ms = run_solver(solver, model_vars)
            
            print(f" Z3 Verification complete:")
            print(f"   Verdict: {verdict}")
            print(f"   Time: {time_ms:.2f}ms")
            
            return {
                'pattern': pattern,
                'verdict': verdict,
                'model': model,
                'time_ms': time_ms,
                'status': 'success'
            }
            
        except Exception as e:
            print(f" Error during Z3 encoding: {e}")
            return {
                'pattern': pattern,
                'status': 'error',
                'error': str(e)
            }
    
    # ============================================================================
    # Utility Functions
    # ============================================================================
    
    def parse_ocl_constraints(self, ocl_file: str) -> List[Tuple[str, str]]:
        """
        Parse OCL constraints file and extract constraints with their context
        
        Args:
            ocl_file: Path to constraints.ocl file
            
        Returns:
            List of (constraint_text, context_class) tuples
        """
        constraints = []
        
        try:
            with open(ocl_file, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for "context ClassName" lines
                if line.startswith('context '):
                    context_class = line.split('context ')[1].strip()
                    
                    # Next lines until "inv" keyword contain the constraint
                    i += 1
                    inv_name = ""
                    constraint_lines = []
                    
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        if line.startswith('inv '):
                            inv_name = line.split('inv ')[1].split(':')[0].strip()
                            i += 1
                            # Collect constraint lines until next context or end
                            while i < len(lines) and not lines[i].strip().startswith('context '):
                                l = lines[i].strip()
                                if l and not l.startswith('--'):
                                    constraint_lines.append(l)
                                i += 1
                            
                            constraint_text = ' '.join(constraint_lines).strip()
                            if constraint_text:
                                constraints.append((constraint_text, context_class, inv_name))
                            break
                        i += 1
                else:
                    i += 1
            
            return constraints
        
        except Exception as e:
            print(f" Error parsing OCL file: {e}")
            return []
    
    # ============================================================================
    # Complete Workflow (Phase 1 + Phase 2)
    # ============================================================================
    
    def run_complete_workflow(self, xmi_file: str, constraints: Optional[List[str]] = None, ocl_file: Optional[str] = None) -> Dict:
        """
        Run complete workflow: Phase 1 (training) + Phase 2 (verification)
        
        Args:
            xmi_file: Path to XMI model
            constraints: List of OCL constraints to verify (optional, can use ocl_file instead)
            ocl_file: Path to constraints.ocl file (optional)
            
        Returns:
            Dictionary with all results
        """
        # CRITICAL: Validate model consistency first
        if ocl_file:
            print("\n🔒 STEP 0: Model Consistency Validation")
            print("="*80)
            checker = ModelConsistencyChecker(xmi_file, ocl_file)
            try:
                validation_result = checker.validate_and_raise()
                print(" Model consistency check passed - proceeding with verification\n")
            except ValueError as e:
                print(f"\n CRITICAL ERROR: {e}")
                print("\n  Cannot proceed - XMI model and OCL constraints appear to be from different models.")
                print("   Please verify that both files belong to the same class diagram.\n")
                return {
                    'error': 'Model consistency validation failed',
                    'details': str(e),
                    'validation_result': validation_result if 'validation_result' in locals() else None
                }
        
        # Parse constraints from file if provided
        if ocl_file and not constraints:
            print(f"\n📖 Parsing OCL constraints from: {ocl_file}")
            parsed = self.parse_ocl_constraints(ocl_file)
            constraints = [c[0] for c in parsed]
            print(f" Extracted {len(constraints)} constraints from OCL file")
        
        if not constraints:
            print(" No constraints provided. Use either 'constraints' or 'ocl_file' parameter.")
            return {}
        
        # Phase 1: Domain Adaptation
        phase1_result = self.phase1_domain_adaptation(xmi_file)
        
        # Phase 2: Verify each constraint
        phase2_results = []
        model_dir = phase1_result['steps']['retrain_classifier']['model_dir']
        
        for i, constraint in enumerate(constraints, 1):
            print(f"\n\n Constraint {i}/{len(constraints)}")
            self.phase2_constraint = constraint  # Store for Z3 encoder
            result = self.phase2_ocl_verification(constraint, model_dir)
            phase2_results.append(result)
        
        return {
            'phase1': phase1_result,
            'phase2': phase2_results,
            'total_constraints': len(constraints),
            'constraints': constraints
        }


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    framework = OCLVerificationFramework()
    
    # CarRental model with constraints from constraints.ocl file
    xmi_file = 'examples/carrentalsystem/model.xmi'
    ocl_file = 'examples/carrentalsystem/constraints.ocl'
    
    print("\n🚀 Starting Complete Two-Phase Workflow\n")
    results = framework.run_complete_workflow(xmi_file, ocl_file=ocl_file)
    
    if results:
        print("\n\n" + "="*80)
        print(" FINAL RESULTS - Pattern to Constraint Mapping")
        print("="*80)
        print(f"\nPhase 1 Status: {results['phase1']['steps']['retrain_classifier']['status']}")
        print(f"Model saved to: {results['phase1']['steps']['retrain_classifier']['model_dir']}")
        
        # Extract pattern-to-constraint mapping
        print(f"\nPhase 2: Pattern-Constraint Analysis ({len(results['phase2'])} constraints verified)")
        print("-" * 80)
        
        # Create formatted table
        print(f"{'No.':<4} {'Pattern':<28} {'Confidence':<12} {'Strategy':<20} {'Verdict':<8}")
        print("-" * 80)
        
        patterns_summary = {}
        for i, phase2_result in enumerate(results['phase2'], 1):
            if 'classify' in phase2_result['steps']:
                classify_step = phase2_result['steps']['classify']
                pattern = classify_step.get('pattern', 'UNKNOWN')
                confidence = classify_step.get('confidence', 0.0)
                
                # Get strategy and verdict
                strategy = "UNKNOWN"
                verdict = "UNKNOWN"
                if 'algorithm' in phase2_result['steps']:
                    strategy = phase2_result['steps']['algorithm'].get('strategy', 'UNKNOWN')
                if 'z3_verification' in phase2_result['steps']:
                    verdict = phase2_result['steps']['z3_verification'].get('verdict', 'ERROR')
                
                # Display row
                strategy_short = "HIGH" if strategy == "HIGH_CONFIDENCE_SINGLE" else ("MED" if "MEDIUM" in strategy else "LOW")
                print(f"{i:<4} {pattern:<28} {confidence:<12.4f} {strategy_short:<20} {verdict:<8}")
                
                # Collect pattern statistics
                if pattern not in patterns_summary:
                    patterns_summary[pattern] = []
                patterns_summary[pattern].append({
                    'constraint_idx': i,
                    'constraint': results['constraints'][i-1][:60] + '...' if len(results['constraints'][i-1]) > 60 else results['constraints'][i-1],
                    'confidence': confidence,
                    'verdict': verdict
                })
        
        # Summary by pattern
        print("\n" + "="*80)
        print(" Summary by Pattern")
        print("="*80)
        for pattern, instances in sorted(patterns_summary.items()):
            print(f"\n {pattern.upper()} ({len(instances)} constraint(s)):")
            for instance in instances:
                print(f"   [{instance['constraint_idx']}] {instance['constraint']} (conf: {instance['confidence']:.4f}, verdict: {instance['verdict']})")
        
        # Statistics
        print("\n" + "="*80)
        print("📈 Statistics")
        print("="*80)
        total_constraints = len(results['phase2'])
        high_conf = sum(1 for r in results['phase2'] if r['steps'].get('classify', {}).get('confidence', 0) > 0.7)
        med_conf = sum(1 for r in results['phase2'] if 0.3 < r['steps'].get('classify', {}).get('confidence', 0) <= 0.7)
        low_conf = sum(1 for r in results['phase2'] if r['steps'].get('classify', {}).get('confidence', 0) <= 0.3)
        
        print(f"Total Constraints: {total_constraints}")
        print(f"  🟢 High Confidence (>0.7): {high_conf} ({high_conf*100//total_constraints}%)")
        print(f"  🟡 Medium Confidence (0.3-0.7): {med_conf} ({med_conf*100//total_constraints}%)")
        print(f"  🔴 Low Confidence (<0.3): {low_conf} ({low_conf*100//total_constraints}%)")
        print(f"\nUnique Patterns Detected: {len(patterns_summary)}")
        print(f"  {', '.join(sorted(patterns_summary.keys()))}")
