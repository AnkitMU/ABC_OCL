#!/usr/bin/env python3
"""
Complete Framework Test Suite
Tests all 50 OCL patterns with association metadata and multiplicity verification

Workflow:
1. Load XMI metadata (classes, associations, multiplicities)
2. Parse OCL constraints from constraints.ocl
3. Detect OCL pattern type
4. Route to enhanced pattern encoder
5. Generate Z3 constraints with domain relationships
6. Verify satisfiability and generate counterexamples
test_enchanced_framework.py

"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Fix imports for both module and direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Try relative imports first (when run as module)
    from .enhanced_smt_encoder import EnhancedSMTEncoder
    from .comprehensive_pattern_detector import ComprehensivePatternDetector
    from .generic_global_consistency_checker import GenericGlobalConsistencyChecker
    from ..lowering.association_backed_encoder import XMIMetadataExtractor
    from ..validation import ModelConsistencyChecker
    from ..classifiers.sentence_transformer import SentenceTransformerClassifier
    from ..classifiers.sentence_transformer.xmi_based_domain_adapter import GenericDomainDataGenerator
    from ..classifiers.sentence_transformer.classifier import OCLPatternType
except ImportError:
    # Fallback to absolute imports (when run as script)
    from ssr_ocl.super_encoder.enhanced_smt_encoder import EnhancedSMTEncoder
    from ssr_ocl.super_encoder.comprehensive_pattern_detector import ComprehensivePatternDetector
    from ssr_ocl.super_encoder.generic_global_consistency_checker import GenericGlobalConsistencyChecker
    from ssr_ocl.lowering.association_backed_encoder import XMIMetadataExtractor
    from ssr_ocl.validation import ModelConsistencyChecker
    from ssr_ocl.classifiers.sentence_transformer import SentenceTransformerClassifier
    from ssr_ocl.classifiers.sentence_transformer.xmi_based_domain_adapter import GenericDomainDataGenerator
    from ssr_ocl.classifiers.sentence_transformer.classifier import OCLPatternType


class FrameworkTestSuite:
    """Complete test suite for hybrid framework with neural classifier"""
    
    def __init__(self, xmi_file: str, ocl_file: str, use_neural_classifier: bool = True):
        """Initialize framework test suite
        
        Args:
            xmi_file: Path to XMI model file
            ocl_file: Path to OCL constraints file
            use_neural_classifier: If True, uses neural network classifier (default)
                                   If False, falls back to regex-based detector
        """
        self.xmi_file = xmi_file
        self.ocl_file = ocl_file
        self.use_neural_classifier = use_neural_classifier
        
        # Core components
        self.extractor = XMIMetadataExtractor(xmi_file)
        self.encoder = EnhancedSMTEncoder(xmi_file)
        self.validator = ModelConsistencyChecker(xmi_file, ocl_file)
        
        # Pattern detection: Neural classifier or regex fallback
        if use_neural_classifier:
            self.classifier = None  # Will be initialized in domain adaptation phase
            self.detector = None
            domain_name = Path(xmi_file).stem
            self.model_dir = f"models/adapted_{domain_name}"
        else:
            # Fallback to regex-based detection
            self.detector = ComprehensivePatternDetector()
            self.classifier = None
            self.model_dir = None
        
        self.results = []
    
    # ========== Helpers ==========
    
    def parse_ocl_file(self) -> List[Dict]:
        """Parse OCL file and extract constraints"""
        constraints = []
        
        with open(self.ocl_file, 'r') as f:
            content = f.read()
        
        # Parse context blocks
        pattern = r'context\s+(\w+)\s+inv\s+(\w+):(.*?)(?=context|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            context_class = match.group(1)
            constraint_name = match.group(2)
            constraint_text = match.group(3).strip()
            
            constraints.append({
                'context': context_class,
                'name': constraint_name,
                'text': constraint_text
            })
        
        return constraints
    
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}\n")
    
    def print_separator(self):
        """Print separator"""
        print(f"{'-'*80}")
    
    # ========== Phase 0: Model Consistency Validation ==========
    
    def test_phase_0_model_validation(self) -> Dict:
        """Phase 0: Validate XMI and OCL consistency"""
        self.print_header("PHASE 0: MODEL CONSISTENCY VALIDATION")
        
        print(f" Validating XMI and OCL consistency...")
        print(f"   XMI: {Path(self.xmi_file).name}")
        print(f"   OCL: {Path(self.ocl_file).name}")
        print()
        
        try:
            validation_result = self.validator.validate()
            
            if validation_result.is_valid:
                print("\n Validation PASSED - Models are consistent")
                print(f"   • All {validation_result.stats['ocl_context_classes_count']} context classes found")
                print(f"   • {validation_result.stats['total_constraints']} constraints validated")
                
                if len(validation_result.warnings) > 0:
                    print(f"     {len(validation_result.warnings)} warnings (non-critical)")
                
                return {
                    'phase': 'Model Validation',
                    'status': 'PASSED',
                    'is_valid': True,
                    'issues': [],
                    'warnings': validation_result.warnings,
                    'stats': validation_result.stats
                }
            else:
                print("\n Validation FAILED - Models may be inconsistent")
                print(f"   • {len(validation_result.issues)} critical issues found")
                for issue in validation_result.issues:
                    print(f"     - {issue}")
                
                print("\n  Cannot proceed with verification - please fix model consistency issues")
                
                return {
                    'phase': 'Model Validation',
                    'status': 'FAILED',
                    'is_valid': False,
                    'issues': validation_result.issues,
                    'warnings': validation_result.warnings,
                    'stats': validation_result.stats
                }
        
        except Exception as e:
            print(f"\n Error during validation: {e}")
            return {
                'phase': 'Model Validation',
                'status': 'ERROR',
                'is_valid': False,
                'error': str(e)
            }
    
    # ========== Phase 1a: Domain Adaptation (Neural Classifier Training) ==========
    
    def test_phase_1a_domain_adaptation(self) -> Dict:
        """Phase 1a: Train neural classifier on domain-specific data"""
        if not self.use_neural_classifier:
            return {
                'phase': 'Domain Adaptation',
                'status': 'SKIPPED',
                'reason': 'Using regex-based detector instead'
            }
        
        self.print_header("PHASE 1A: DOMAIN ADAPTATION (Neural Classifier Training)")
        
        print(f"🧠 Training neural classifier for domain: {Path(self.xmi_file).stem}")
        print(f"   Model will be saved to: {self.model_dir}\n")
        
        try:
            # Step 1: Generate domain-specific training data
            print("[Step 1] Generating domain-specific training data...")
            generator = GenericDomainDataGenerator(self.xmi_file, examples_per_pattern=100)  # Increased to 100 for better training
            domain_examples = generator.generate_domain_data()
            
            domain_name = Path(self.xmi_file).stem
            domain_file = f"ocl_domain_{domain_name}_adapted.json"
            generator.save_to_json(domain_file)
            
            print(f" Generated {len(domain_examples)} domain examples")
            print(f"   Saved to: {domain_file}\n")
            
            # Step 2: Merge with generic training data
            print("[Step 2] Merging with generic training data...")
            import json
            generic_file = "ocl_training_data.json"
            
            try:
                with open(generic_file, 'r') as f:
                    generic_data = json.load(f)
                    generic_count = len(generic_data['examples'])
            except FileNotFoundError:
                print(f"  Generic training data not found: {generic_file}")
                print(f"   Using domain data only")
                generic_count = 0
                generic_data = {'examples': []}
            
            with open(domain_file, 'r') as f:
                domain_data = json.load(f)
            
            merged_examples = generic_data['examples'] + domain_data['examples']
            print(f" Merged: {generic_count} generic + {len(domain_examples)} domain = {len(merged_examples)} total\n")
            
            # Step 3: Train classifier
            print("[Step 3] Training SentenceTransformer classifier...")
            training_data = [(ex['ocl_text'], ex['pattern']) for ex in merged_examples]
            
            self.classifier = SentenceTransformerClassifier(self.model_dir)
            self.classifier.train(training_data)
            
            print(f"\n Classifier trained and saved to: {self.model_dir}")
            
            return {
                'phase': 'Domain Adaptation',
                'domain_examples': len(domain_examples),
                'total_training_examples': len(merged_examples),
                'model_dir': self.model_dir,
                'status': 'COMPLETE'
            }
            
        except Exception as e:
            print(f"\n Error during domain adaptation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'phase': 'Domain Adaptation',
                'status': 'FAILED',
                'error': str(e)
            }
    
    # ========== Phase 1: Metadata Extraction ==========
    
    def test_phase_1_metadata_extraction(self) -> Dict:
        """Phase 1: Extract and verify XMI metadata"""
        self.print_header("PHASE 1: METADATA EXTRACTION")
        
        print(f" Domain Model: {Path(self.xmi_file).stem}")
        print(f"   Classes: {len(self.extractor.classes)}")
        print(f"   Associations: {len(self.extractor.get_associations())}")
        
        print(f"\n Classes extracted:")
        for cls in sorted(self.extractor.classes):
            print(f"   • {cls}")
        
        print(f"\n🔗 Associations extracted:")
        for assoc in self.extractor.get_associations()[:15]:
            print(f"   • {assoc}")
        
        if len(self.extractor.get_associations()) > 15:
            print(f"   ... and {len(self.extractor.get_associations()) - 15} more")
        
        return {
            'phase': 'Metadata Extraction',
            'classes': len(self.extractor.classes),
            'associations': len(self.extractor.get_associations()),
            'status': 'COMPLETE'
        }
    
    # ========== Phase 2: Constraint Parsing ==========
    
    def test_phase_2_constraint_parsing(self) -> Dict:
        """Phase 2: Parse and analyze constraints"""
        self.print_header("PHASE 2: CONSTRAINT PARSING")
        
        constraints = self.parse_ocl_file()
        print(f" Constraints loaded: {len(constraints)}")
        
        for i, c in enumerate(constraints[:5], 1):
            print(f"\n   {i}. {c['name']} (Context: {c['context']})")
            print(f"      {c['text'][:70]}...")
        
        if len(constraints) > 5:
            print(f"\n   ... and {len(constraints) - 5} more constraints")
        
        return {
            'phase': 'Constraint Parsing',
            'constraints': len(constraints),
            'status': 'COMPLETE'
        }
    
    # ========== Phase 3: Pattern Detection ==========
    
    def test_phase_3_pattern_detection(self) -> Dict:
        """Phase 3: Detect OCL patterns using hybrid approach (neural + regex fallback)"""
        self.print_header("PHASE 3: PATTERN DETECTION (HYBRID)")
        
        constraints = self.parse_ocl_file()
        pattern_counts = {}
        confidences = []
        fallback_used = 0
        confidence_threshold = 0.5  # Use regex fallback if confidence < 0.5
        
        # Initialize OCL normalizer for preprocessing
        try:
            from .ocl_normalizer import OCLNormalizer
        except ImportError:
            from ssr_ocl.super_encoder.ocl_normalizer import OCLNormalizer
        normalizer = OCLNormalizer(enable_logging=False)
        
        if self.use_neural_classifier:
            print(f"🧠 Using Hybrid Approach: Neural Classifier + Comprehensive Regex Fallback")
            print(f"   Confidence Threshold: {confidence_threshold} (below this → comprehensive regex)")
            print(f"   Regex Patterns: 617 comprehensive patterns across 50 types")
            print(f"   ⚙️  OCL Normalization: ENABLED")
            if self.classifier is None:
                print(" Error: Classifier not initialized. Run domain adaptation first.")
                return {'phase': 'Pattern Detection', 'status': 'FAILED', 'error': 'Classifier not initialized'}
            # Initialize comprehensive regex detector for fallback
            if self.detector is None:
                try:
                    from .comprehensive_pattern_detector import ComprehensivePatternDetector
                except ImportError:
                    from ssr_ocl.super_encoder.comprehensive_pattern_detector import ComprehensivePatternDetector
                self.detector = ComprehensivePatternDetector()
        else:
            print(f" Using Comprehensive Regex-based Pattern Detector")
            print(f"   Regex Patterns: 617 comprehensive patterns")
            print(f"   ⚙️  OCL Normalization: ENABLED")
        
        print(f"\n🎯 Detecting patterns in {len(constraints)} constraints:\n")
        
        for i, constraint in enumerate(constraints, 1):
            # Apply OCL normalization first
            normalized_text = normalizer.normalize(constraint['text'])
            
            if self.use_neural_classifier:
                # Use neural classifier on normalized text
                pattern_name, confidence = self.classifier.predict(normalized_text)
                confidences.append(confidence)
                
                # Use regex fallback if confidence is low
                if confidence < confidence_threshold:
                    regex_pattern = self.detector.detect_pattern(normalized_text)
                    fallback_used += 1
                    confidence_str = f" (conf: {confidence:.3f}  → regex: {regex_pattern.value})"
                    # Use regex pattern instead
                    pattern_name = regex_pattern.value
                else:
                    confidence_str = f" (conf: {confidence:.3f})"
            else:
                # Use regex detector on normalized text
                pattern = self.detector.detect_pattern(normalized_text)
                pattern_name = pattern.value
                confidence_str = ""
            
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
            print(f"   {i:2}. {constraint['name']:30} → {pattern_name}{confidence_str}")
        
        print(f"\n Pattern Distribution:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"   • {pattern:30} : {count} constraint(s)")
        
        result = {
            'phase': 'Pattern Detection',
            'method': 'hybrid' if self.use_neural_classifier else 'regex_detector',
            'patterns_detected': len(pattern_counts),
            'pattern_distribution': pattern_counts,
            'status': 'COMPLETE'
        }
        
        if self.use_neural_classifier and confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"\n Neural Classifier Stats:")
            print(f"   Average Confidence: {avg_confidence:.3f}")
            print(f"   High Confidence (>={confidence_threshold}): {len(constraints) - fallback_used}/{len(constraints)}")
            print(f"   Regex Fallback Used: {fallback_used}/{len(constraints)} ({fallback_used*100//len(constraints)}%)")
            result['average_confidence'] = avg_confidence
            result['confidences'] = confidences
            result['fallback_used'] = fallback_used
            result['confidence_threshold'] = confidence_threshold
        
        return result
    
    # ========== Phase 4: Association Resolution ==========
    
    def test_phase_4_association_resolution(self) -> Dict:
        """Phase 4: Resolve constraints to associations"""
        self.print_header("PHASE 4: ASSOCIATION RESOLUTION")
        
        constraints = self.parse_ocl_file()
        resolved = 0
        unresolved = 0
        
        print(f"🔗 Resolving constraints to domain associations:\n")
        
        for i, constraint in enumerate(constraints[:10], 1):
            # Extract reference name
            match = re.search(r'self\.(\w+)', constraint['text'])
            if match:
                ref_name = match.group(1)
                assoc = self.extractor.get_association_by_ref(constraint['context'], ref_name)
                
                if assoc:
                    print(f"   {i:2}. {constraint['name']:30}")
                    print(f"        Association: {assoc}")
                    print(f"        Multiplicity: {assoc.multiplicity_str()}")
                    resolved += 1
                else:
                    # Check if it's an attribute before reporting as unresolved
                    attr_metadata = self.extractor.get_attribute_by_name(constraint['context'], ref_name)
                    
                    if attr_metadata:
                        print(f"   {i:2}. {constraint['name']:30}")
                        print(f"         Attribute (not association): {attr_metadata}")
                        resolved += 1  # Count as resolved - it's a valid field
                    else:
                        print(f"   {i:2}. {constraint['name']:30}")
                        print(f"         No association found for: {constraint['context']}.{ref_name}")
                        unresolved += 1
            else:
                print(f"   {i:2}. {constraint['name']:30}")
                print(f"         No reference found in constraint")
                unresolved += 1
        
        print(f"\n   Resolved: {resolved}, Unresolved: {unresolved}")
        
        return {
            'phase': 'Association Resolution',
            'resolved': resolved,
            'unresolved': unresolved,
            'status': 'COMPLETE'
        }
    
    # ========== Phase 5: SMT Encoding ==========
    
    def test_phase_5_smt_encoding(self) -> Dict:
        """Phase 5: Encode constraints to Z3"""
        self.print_header("PHASE 5: SMT ENCODING")
        
        constraints = self.parse_ocl_file()
        encoding_results = []
        
        print(f" Encoding {len(constraints)} constraints to Z3:\n")
        
        for i, constraint in enumerate(constraints[:8], 1):
            # Get pattern using classifier or detector
            if self.use_neural_classifier:
                pattern_name, confidence = self.classifier.predict(constraint['text'])
            else:
                pattern_name = self.detector.detect_pattern(constraint['text']).value
                confidence = None
            
            context = constraint['context']
            
            print(f"   {i}. {constraint['name']}")
            print(f"      Pattern: {pattern_name}")
            
            try:
                # Prepare context for encoder
                encoder_context = {
                    'context_class': context,
                    'scope': 5,
                    'source_scope': 2,
                    'target_scope': 5
                }
                
                # Encode constraint
                solver, model_vars = self.encoder.encode(
                    pattern_name,
                    constraint['text'],
                    encoder_context
                )
                
                # Check satisfiability
                z3_result = solver.check()
                print(f"      Z3 Result: {z3_result}")
                print(f"      Variables: {len(model_vars)}")
                
                encoding_results.append({
                    'name': constraint['name'],
                    'pattern': pattern_name,
                    'z3_result': str(z3_result),
                    'variables': len(model_vars),
                    'status': 'SUCCESS'
                })
                
            except Exception as e:
                print(f"       Error: {str(e)[:60]}")
                encoding_results.append({
                    'name': constraint['name'],
                    'pattern': pattern_name,
                    'error': str(e),
                    'status': 'FAILED'
                })
            
            print()
        
        successful = sum(1 for r in encoding_results if r['status'] == 'SUCCESS')
        failed = sum(1 for r in encoding_results if r['status'] == 'FAILED')
        
        print(f"   Successful: {successful}, Failed: {failed}")
        
        return {
            'phase': 'SMT Encoding',
            'encoded': len(encoding_results),
            'successful': successful,
            'failed': failed,
            'details': encoding_results,
            'status': 'COMPLETE'
        }

    # ========== Phase 6: Verification ==========
    
    def test_phase_6_verification(self) -> Dict:
        """Phase 6: Verify constraints and generate counterexamples"""
        self.print_header("PHASE 6: VERIFICATION")
        
        constraints = self.parse_ocl_file()
        verification_results = []
        
        print(f" Verifying {len(constraints)} constraints:\n")
        
        for i, constraint in enumerate(constraints[:5], 1):
            # Get pattern using classifier or detector
            if self.use_neural_classifier:
                pattern_name, confidence = self.classifier.predict(constraint['text'])
            else:
                pattern_name = self.detector.detect_pattern(constraint['text']).value
                confidence = None
            
            print(f"   {i}. {constraint['name']}")
            print(f"      Constraint: {constraint['text'][:60]}...")
            
            try:
                encoder_context = {
                    'context_class': constraint['context'],
                    'scope': 5,
                    'source_scope': 2,
                    'target_scope': 5
                }
                
                solver, model_vars = self.encoder.encode(
                    pattern_name,
                    constraint['text'],
                    encoder_context
                )
                
                z3_result = solver.check()
                
                if str(z3_result) == 'sat':
                    print(f"        SAT - Counterexample found (violation possible)")
                    model = solver.model()
                    print(f"      Sample assignment:")
                    
                    # Print first few variables (skip metadata starting with _)
                    count = 0
                    for key in model_vars.keys():
                        if count >= 3:  # Limit to 3 variables
                            break
                        if key.startswith('_'):  # Skip metadata variables
                            continue
                        try:
                            val = model.evaluate(model_vars[key])
                            print(f"         {key} = {val}")
                            count += 1
                        except Exception as e:
                            # Skip variables that can't be evaluated
                            continue
                    
                    status = 'VIOLATION'
                else:
                    print(f"       {z3_result} - Constraint verified")
                    status = 'VERIFIED'
                
                verification_results.append({
                    'name': constraint['name'],
                    'pattern': pattern_name,
                    'status': status,
                    'z3_result': str(z3_result)
                })
                
            except Exception as e:
                print(f"       Error: {str(e)[:60]}")
                verification_results.append({
                    'name': constraint['name'],
                    'status': 'ERROR',
                    'error': str(e)
                })
            
            print()
        
        verified = sum(1 for r in verification_results if r['status'] == 'VERIFIED')
        violations = sum(1 for r in verification_results if r['status'] == 'VIOLATION')
        
        print(f"   Verified: {verified}, Violations: {violations}")
        
        return {
            'phase': 'Verification',
            'verified': verified,
            'violations': violations,
            'details': verification_results,
            'status': 'COMPLETE'
        }

    # ========== Phase 6b: Global Consistency ==========
    
    def test_phase_6b_global_consistency(self) -> Dict:
        """Phase 6b: Verify all constraints can coexist simultaneously"""
        self.print_header("PHASE 6B: GLOBAL CONSISTENCY VERIFICATION")
        
        constraints = self.parse_ocl_file()
        
        # Prepare constraint list with pattern info
        constraint_list = []
        for constraint in constraints:
            # Get pattern using classifier or detector
            if self.use_neural_classifier:
                pattern_name, confidence = self.classifier.predict(constraint['text'])
            else:
                pattern_name = self.detector.detect_pattern(constraint['text']).value
            
            constraint_list.append({
                'name': constraint['name'],
                'pattern': pattern_name,
                'context': constraint['context'],
                'text': constraint['text']
            })
        
        # Auto-generate scope from XMI metadata (generic approach)
        scope = {}
        for class_name in self.extractor.classes:
            # Use singular form as key (convention: nClassName without plural)
            scope[f'n{class_name}'] = 1  # Minimal scope of 1 per class
        
        print(f" Auto-generated scope from XMI metadata:")
        for key, val in scope.items():
            print(f"   {key}: {val}")
        print()
        
        # Create global consistency checker with rich instances for realistic output
        # Using NEW 50-pattern generic checker!
        checker = GenericGlobalConsistencyChecker(
            self.xmi_file,
            rich_instances=True,      # Force realistic instances
            timeout_ms=60000,         # 60 second timeout
            show_raw_values=False      # Show Z3 raw values for transparency
        )
        
        try:
            # Verify all constraints together
            result, model = checker.verify_all_constraints(constraint_list, scope)
            
            if result == 'sat':
                return {
                    'phase': 'Global Consistency',
                    'status': 'CONSISTENT',
                    'result': result,
                    'message': 'All constraints can be satisfied simultaneously',
                    'model_exists': True
                }
            elif result == 'unsat':
                return {
                    'phase': 'Global Consistency',
                    'status': 'INCONSISTENT',
                    'result': result,
                    'message': 'Constraints are contradictory - no valid instance exists',
                    'model_exists': False
                }
            else:  # unknown
                return {
                    'phase': 'Global Consistency',
                    'status': 'UNKNOWN',
                    'result': result,
                    'message': 'Solver timeout or inconclusive',
                    'model_exists': False
                }
        
        except Exception as e:
            print(f"\n Error during global consistency check: {e}")
            return {
                'phase': 'Global Consistency',
                'status': 'ERROR',
                'error': str(e),
                'model_exists': False
            }
    
    # ========== Main Test Runner ==========
    
    def run_complete_test_suite(self) -> Dict:
        """Run complete test suite"""
        self.print_header("HYBRID FRAMEWORK COMPLETE TEST SUITE")
        
        print(f"📁 XMI Model: {self.xmi_file}")
        print(f"📄 OCL File: {self.ocl_file}")
        print(f"🕐 Timestamp: {Path(self.ocl_file).stat().st_mtime}")
        
        # Phase 0: Model Consistency Validation (CRITICAL)
        phase0 = self.test_phase_0_model_validation()
        
        # Check validation result before proceeding
        if not phase0.get('is_valid', False):
            print("\n" + "="*80)
            print(" TEST SUITE ABORTED: Model validation failed")
            print("="*80)
            print("\n  Please fix model consistency issues before running tests.")
            print("   XMI and OCL files must belong to the same model.\n")
            return {
                'phase0': phase0,
                'status': 'ABORTED',
                'reason': 'Model validation failed'
            }
        
        # Run all phases
        phase1a = self.test_phase_1a_domain_adaptation()  # Neural classifier training
        phase1 = self.test_phase_1_metadata_extraction()
        phase2 = self.test_phase_2_constraint_parsing()
        phase3 = self.test_phase_3_pattern_detection()
        phase4 = self.test_phase_4_association_resolution()
        phase5 = self.test_phase_5_smt_encoding()
        phase6 = self.test_phase_6_verification()
        phase6b = self.test_phase_6b_global_consistency()
        

        ''''
        # Final summary
        self.print_header("TEST SUITE SUMMARY")
        
        print(f" Phase 0: Model Validation - {phase0['status']}")
        print(f"   Context Classes: {phase0['stats']['ocl_context_classes_count']}, Issues: {len(phase0['issues'])}")
        
        if phase1a['status'] != 'SKIPPED':
            print(f"\n Phase 1a: Domain Adaptation - {phase1a['status']}")
            if phase1a['status'] == 'COMPLETE':
                print(f"   Training Examples: {phase1a.get('total_training_examples', 'N/A')}")
                print(f"   Model: {phase1a.get('model_dir', 'N/A')}")
        
        print(f"\n Phase 1: Metadata Extraction - {phase1['status']}")
        print(f"   Classes: {phase1['classes']}, Associations: {phase1['associations']}")
        
        print(f"\n Phase 2: Constraint Parsing - {phase2['status']}")
        print(f"   Constraints: {phase2['constraints']}")
        
        print(f"\n Phase 3: Pattern Detection - {phase3['status']}")
        print(f"   Method: {phase3.get('method', 'unknown')}")
        print(f"   Patterns: {phase3['patterns_detected']}")
        if 'average_confidence' in phase3:
            print(f"   Avg Confidence: {phase3['average_confidence']:.3f}")
        
        print(f"\n Phase 4: Association Resolution - {phase4['status']}")
        print(f"   Resolved: {phase4['resolved']}, Unresolved: {phase4['unresolved']}")
        
        print(f"\n Phase 5: SMT Encoding - {phase5['status']}")
        print(f"   Encoded: {phase5['encoded']}, Successful: {phase5['successful']}, Failed: {phase5['failed']}")
        
        print(f"\n Phase 6: Verification - {phase6['status']}")
        print(f"   Verified: {phase6['verified']}, Violations: {phase6['violations']}")
        
        print(f"\n Phase 6b: Global Consistency - {phase6b['status']}")
        if phase6b['status'] == 'CONSISTENT':
            print(f"   🎉 Model is CONSISTENT - all constraints can coexist!")
        elif phase6b['status'] == 'INCONSISTENT':
            print(f"     Model is INCONSISTENT - constraints are contradictory")
        else:
            print(f"   {phase6b.get('message', 'Status unknown')}")
        
        print(f"\n{'='*80}")
        print(" COMPLETE FRAMEWORK TEST SUITE FINISHED")
        print("="*80 + "\n")
        
        return {
            'phase0': phase0,
            'phase1a': phase1a,
            'phase1': phase1,
            'phase2': phase2,
            'phase3': phase3,
            'phase4': phase4,
            'phase5': phase5,
            'phase6': phase6,
            'phase6b': phase6b,
            'status': 'COMPLETE'
        }
'''

def main():
    """Main entry point - configurable for any model"""
    import sys
    
    # Allow command-line arguments to read data
    if len(sys.argv) >= 3:
        xmi_path = sys.argv[1]
        ocl_path = sys.argv[2]
    else:
        # Model example(Ecore)
        xmi_path = "examples/carrentalsystem/model.xmi"
        ocl_path = "examples/carrentalsystem/constraints.ocl"
        print(f"   Usage: python {sys.argv[0]} <xmi_path> <ocl_path>")
        print()
    
    from pathlib import Path
    if not Path(xmi_path).exists():
        print(f" XMI file not found: {xmi_path}")
        return 1
    
    if not Path(ocl_path).exists():
        print(f" OCL file not found: {ocl_path}")
        return 1
    
    # Run test suite
    test_suite = FrameworkTestSuite(xmi_path, ocl_path)
    result = test_suite.run_complete_test_suite()
    
    return 0


if __name__ == '__main__':
    exit(main())
