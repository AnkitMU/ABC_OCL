"""
Advanced benchmark engine with coverage tracking, diversity filtering, and adaptive generation.
"""
from __future__ import annotations
import random
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from modules.core.models import Metamodel, OCLConstraint
from modules.synthesis.pattern_engine.pattern_registry import PatternRegistry
from modules.generation.composer.ocl_generator import OCLGenerator

from .bench_config import BenchmarkProfile, FAMILY_KEYS, OPERATORS, TYPES
from .coverage_tracker import compute_coverage, count_operators, nav_hops, quantifier_depth
from .metadata_enricher import similarity, difficulty_score


def classify_family(pattern_id: str, category: str) -> str:
    """Classify pattern into family."""
    pid = pattern_id.lower()
    cat = (category or "").lower()
    
    if cat == "string" or pid.startswith("string_") or "regex" in pid:
        return "string"
    if cat == "arithmetic" or any(k in pid for k in ["numeric", "range", "division", "abs_min_max"]):
        return "arithmetic"
    if any(k in pid for k in ["forall", "exists", "select_operation", "collect_operation", "one_operation", "any_operation"]):
        return "quantified"
    if cat == "navigation" or "navigation" in pid:
        return "navigation"
    if "unique" in pid or pid in ("uniqueness_constraint", "all_different"):
        return "uniqueness"
    if any(k in pid for k in ["size", "isempty", "notempty", "includes", "excludes", "cardinality"]):
        return "cardinality"
    if "enum" in pid or "forbiddenliteral" in pid or "subset" in pid:
        return "enum"
    if any(k in pid for k in ["ocliskindof", "oclastype", "ocl_is", "type_check"]):
        return "type_checks"
    if cat in ("collection", "set_operations"):
        return "cardinality"
    if cat == "ocl_library":
        return "type_checks"
    return "cardinality"


class CoverageState:
    """Live coverage tracking during generation."""
    def __init__(self, metamodel: Metamodel, targets: 'BenchmarkProfile'):
        self.metamodel = metamodel
        self.targets = targets.coverage
        self.constraints: List[OCLConstraint] = []
        
        # Track usage
        self.classes_used: Set[str] = set()
        self.attributes_used: Set[Tuple[str, str]] = set()
        self.associations_used: Set[Tuple[str, str]] = set()
        self.operator_counts: Dict[str, int] = {op: 0 for op in OPERATORS}
        self.hop_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self.depth_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self.type_counts: Dict[str, int] = {t: 0 for t in TYPES}
        self.difficulty_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        
    def add_constraint(self, c: OCLConstraint):
        """Add constraint and update coverage."""
        self.constraints.append(c)
        self.classes_used.add(c.context)
        
        ocl = c.ocl
        # Operators
        ops = count_operators(ocl)
        for k, v in ops.items():
            self.operator_counts[k] += v
        
        # Hops
        hops = nav_hops(ocl)
        bucket = 0 if hops == 0 else (1 if hops == 1 else 2)
        self.hop_counts[bucket] += 1
        
        # Depth
        depth = quantifier_depth(ocl)
        bucket = 0 if depth == 0 else (1 if depth == 1 else 2)
        self.depth_counts[bucket] += 1
        
        # Difficulty
        diff = difficulty_score(ocl)
        diff_key = ["easy", "medium", "hard"][diff]
        self.difficulty_counts[diff_key] += 1
    
    def score(self) -> float:
        """Compute overall coverage score (0-1)."""
        total_classes = len(self.metamodel.classes)
        total_attrs = sum(len(c.attributes) for c in self.metamodel.classes.values())
        total_assocs = len(self.metamodel.get_all_associations())
        
        scores = []
        
        # Class coverage
        if total_classes > 0:
            target_classes = self.targets.class_context_pct / 100.0 * total_classes
            scores.append(min(1.0, len(self.classes_used) / max(1, target_classes)))
        
        # Operator coverage
        for op in OPERATORS:
            target = self.targets.operator_mins.get(op, 0)
            if target > 0:
                scores.append(min(1.0, self.operator_counts[op] / target))
        
        # Hop coverage
        for k in [0, 1, 2]:
            key = str(k) if k < 2 else "2plus"
            target = self.targets.nav_hops.get(key, 0)
            if target > 0:
                scores.append(min(1.0, self.hop_counts[k] / target))
        
        # Depth coverage
        for k in [0, 1, 2]:
            key = str(k) if k < 2 else "2plus"
            target = self.targets.quantifier_depth.get(key, 0)
            if target > 0:
                scores.append(min(1.0, self.depth_counts[k] / target))
        
        # Difficulty mix
        total = sum(self.difficulty_counts.values()) or 1
        for key in ["easy", "medium", "hard"]:
            target_pct = self.targets.difficulty_mix.get(key, 0) / 100.0
            actual_pct = self.difficulty_counts[key] / total
            scores.append(1.0 - abs(target_pct - actual_pct))
        
        return sum(scores) / max(1, len(scores)) if scores else 0.0
    
    def deficits(self) -> List[Tuple[str, int, int]]:
        """Return list of (target_name, achieved, needed) for unmet targets."""
        gaps = []
        
        # Operators
        for op in OPERATORS:
            target = self.targets.operator_mins.get(op, 0)
            if target > 0 and self.operator_counts[op] < target:
                gaps.append((f"op:{op}", self.operator_counts[op], target))
        
        # Hops
        for k in [0, 1, 2]:
            key = str(k) if k < 2 else "2plus"
            target = self.targets.nav_hops.get(key, 0)
            if target > 0 and self.hop_counts[k] < target:
                gaps.append((f"hops:{key}", self.hop_counts[k], target))
        
        # Depth
        for k in [0, 1, 2]:
            key = str(k) if k < 2 else "2plus"
            target = self.targets.quantifier_depth.get(key, 0)
            if target > 0 and self.depth_counts[k] < target:
                gaps.append((f"depth:{key}", self.depth_counts[k], target))
        
        return gaps


class BenchmarkEngineV2:
    """Advanced benchmark generator with full coverage and diversity."""
    
    # C: CENTRALIZED BLACKLIST - Single source of truth for all phases
    BLACKLIST = {
        # Tautologies (always true/false)
        'isEmpty_notEmpty',                 # Always true: isEmpty() or notEmpty()
        'self_not_null',                    # Always true in OCL
        
        # Not a constraint (expression without assertion)
        'set_difference',                   # Just returns a set
        'closure_operation',                # Just returns a closure
        'product_operation',                # Just returns a product
        'string_concat',                    # Just concatenates
        'abs_min_max',                      # Expression without comparison
        
        # Not implemented in encoder
        'collection_indexOf',               # Not supported
        'oclAsType',                        # Causes type errors
        'oclAsType_cast',                   # Causes type errors
        
        # Type check issues
        'oclIsTypeOf_check',                # Often wrong types
        'oclIsKindOf_check',                # Often wrong types
        'allInstances_check',               # Encoder errors
        
        # Patterns that commonly generate tautologies
        'two_attributes_equal',             # Often picks same attribute twice
        'string_comparison',                # Often picks same attribute twice
        
        # String operations not supported by Z3
        'string_to_upper_equals',           # toUpper() not encodable
        'string_to_lower_equals',           # toLower() not encodable
        'string_operation',                 # Generates syntax errors like size()(value)
        
        # Collection operations with encoding issues
        'collection_sortedBy',              # sortedBy() causes type errors with booleans
        
        # Broken pattern templates
        'at_least_one_defined',             # Generates malformed OCL
        'null_check',                       # Uses oclIsUndefined() - low value
    }
    
    # B: SAT/UNSAT PATTERN TAGS - Patterns likely to generate SAT vs UNSAT
    SAT_LIKELY_PATTERNS = {
        # Lenient constraints - usually satisfiable
        'numeric_non_negative', 'numeric_positive', 'string_min_length',
        'collection_not_empty_simple', 'boolean_is_true', 'boolean_is_false',
        'attribute_not_null_simple', 'implies_simple',
    }
    
    UNSAT_LIKELY_PATTERNS = {
        # Strict/conflicting constraints - harder to satisfy
        'numeric_bounded',  # Can conflict with other bounds
        'xor_condition',  # Exclusive conditions
        'acyclicity',  # Graph constraints
        'all_different',  # Uniqueness across collection
    }
    
    # D: SMT-COMPATIBLE PATTERNS - Patterns that map cleanly to Z3
    SMT_VERIFIED_PATTERNS = {
        # Numeric
        'numeric_positive', 'numeric_non_negative', 'numeric_bounded',
        'numeric_comparison', 'range_constraint', 'numeric_sum_constraint',
        'numeric_difference_constraint', 'numeric_even', 'numeric_odd',
        # Boolean
        'boolean_guard', 'boolean_is_true', 'boolean_is_false',
        # Collection
        'size_constraint', 'collection_not_empty_simple', 'collection_min_size',
        'collection_max_size', 'collection_has_size',
        # String (basic)
        'string_min_length', 'string_max_length', 'string_exact_length',
        # Logic
        'implies_simple', 'all_attributes_defined', 'at_least_one_defined',
    }

    NUMERIC_TYPES = {
        'Integer', 'Real', 'Double', 'Float', 'EInt', 'EDouble', 'EFloat', 'ELong', 'EShort'
    }
    INTEGER_TYPES = {
        'Integer', 'int', 'EInt', 'Long', 'long', 'ELong', 'Short', 'short', 'EShort'
    }
    BOOLEAN_TYPES = {'Boolean', 'EBoolean'}
    STRING_TYPES = {'String', 'EString'}

    def _normalize_type(self, t: Optional[str]) -> str:
        if not t:
            return 'unknown'
        t_lower = t.lower()
        if t in self.NUMERIC_TYPES:
            return 'numeric'
        if t in self.BOOLEAN_TYPES or t_lower == 'boolean':
            return 'boolean'
        if t in self.STRING_TYPES or t_lower == 'string':
            return 'string'
        if 'date' in t_lower or 'time' in t_lower or 'timestamp' in t_lower:
            return 'date'
        return 'other'

    def _types_compatible(self, t1: Optional[str], t2: Optional[str]) -> bool:
        g1 = self._normalize_type(t1)
        g2 = self._normalize_type(t2)
        if g1 == 'unknown' or g2 == 'unknown':
            return True
        if g1 == g2:
            return True
        # Allow numeric mixing (Integer/Real/etc.)
        if g1 == 'numeric' and g2 == 'numeric':
            return True
        return False

    def _is_integer_type(self, t: Optional[str]) -> bool:
        if not t:
            return False
        if t in self.INTEGER_TYPES:
            return True
        t_lower = t.lower()
        if t_lower in {x.lower() for x in self.INTEGER_TYPES}:
            return True
        return any(x in t_lower for x in ['int', 'long', 'short'])
    
    def _attribute_names_compatible(self, first_attr: str, second_attr: str) -> bool:
        """Heuristic compatibility check based on attribute names."""
        first_lower = first_attr.lower()
        second_lower = second_attr.lower()

        # Boolean-like naming conventions
        is_first_bool = any(x in first_lower for x in ['is', 'has', 'can', 'should'])
        is_second_bool = any(x in second_lower for x in ['is', 'has', 'can', 'should'])
        if is_first_bool != is_second_bool:
            return False

        # Date/time naming conventions
        is_first_date = any(x in first_lower for x in ['date', 'time', 'at', 'when'])
        is_second_date = any(x in second_lower for x in ['date', 'time', 'at', 'when'])
        if is_first_date != is_second_date:
            return False

        # Money/price naming conventions
        is_first_money = any(x in first_lower for x in ['amount', 'price', 'cost', 'fee', 'rate', 'payment'])
        is_second_money = any(x in second_lower for x in ['amount', 'price', 'cost', 'fee', 'rate', 'payment'])
        if is_first_money != is_second_money:
            return False

        # Distance/mileage naming conventions
        is_first_distance = any(x in first_lower for x in ['mileage', 'distance', 'kilometer', 'mile', 'odometer'])
        is_second_distance = any(x in second_lower for x in ['mileage', 'distance', 'kilometer', 'mile', 'odometer'])
        if is_first_distance != is_second_distance:
            return False

        return True

    def _get_attribute_type(self, context: str, attr_name: str) -> Optional[str]:
        attr = next((a for a in self.metamodel.get_attributes_for(context) if a.name == attr_name), None)
        return attr.type if attr else None

    def _get_association(self, context: str, ref_name: str):
        return next((a for a in self.metamodel.get_associations_for(context) if a.ref_name == ref_name), None)
    
    def __init__(self, metamodel: Metamodel, enable_semantic_validation: bool = True,
                 verification_enabled: bool = False):
        logger.info("="*80)
        logger.info("Initializing BenchmarkEngineV2")
        logger.info(f"Metamodel: {len(metamodel.classes)} classes")
        logger.info(f"Semantic validation: {'ENABLED' if enable_semantic_validation else 'DISABLED'}")
        
        self.metamodel = metamodel
        self.verification_enabled = verification_enabled
        self.registry = PatternRegistry()
        self.generator = OCLGenerator(pattern_registry=self.registry, metamodel=metamodel)
        logger.debug(f"Pattern registry loaded: {len(self.registry.get_all_patterns())} patterns")
        
        # A: FAILURE TRACKING - Track pattern/context failures for adaptive weighting
        self.pattern_fail_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.pattern_success_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        # Track string attributes already constrained to avoid null-check conflicts
        self._string_constrained_attrs: Dict[str, set] = defaultdict(set)
        
        # Index patterns by family
        logger.debug("Indexing patterns by family...")
        self.patterns_by_family: Dict[str, List] = {k: [] for k in FAMILY_KEYS}
        self.all_patterns = self.registry.get_all_patterns()
        for p in self.all_patterns:
            fam = classify_family(p.id, getattr(p.category, "value", str(p.category)))
            if fam not in self.patterns_by_family:
                self.patterns_by_family[fam] = []
            self.patterns_by_family[fam].append(p)
        
        logger.debug(f"Pattern families: {', '.join(f'{k}={len(v)}' for k, v in self.patterns_by_family.items() if v)}")
        
        # TIER 2: Initialize semantic validator (ENABLED BY DEFAULT)
        logger.debug("Initializing semantic validator...")
        self.semantic_validator = None
        self.pattern_template_validator = None
        if enable_semantic_validation:
            try:
                logger.debug("Loading semantic_rules module...")
                import sys
                from pathlib import Path
                config_path = Path(__file__).parent.parent.parent.parent / 'config'
                if str(config_path) not in sys.path:
                    sys.path.insert(0, str(config_path))
                from semantic_rules import SemanticValidator, is_valid_pattern_template
                self.semantic_validator = SemanticValidator()
                self.pattern_template_validator = is_valid_pattern_template
                
                # Validate all loaded patterns
                invalid_count = 0
                for pattern in self.all_patterns:
                    template = getattr(pattern, 'template', '')
                    if template:
                        is_valid, reason = self.pattern_template_validator(pattern.id, template)
                        if not is_valid:
                            invalid_count += 1
                            print(f"Pattern '{pattern.id}' has invalid template: {reason}")
                
                print(f"Semantic validation enabled (Tier 2)")
                if invalid_count > 0:
                    print(f"   Found {invalid_count} patterns with invalid templates (will be skipped during generation)")
            except ImportError as e:
                print(f"Semantic validator not available: {e}")
                print("   Falling back to basic validation (Tier 1 only)")
        
        # TIER 3: Initialize advanced semantic analyzers
        from modules.semantic.metamodel.structure_analyzer import StructureAnalyzer
        from modules.semantic.metamodel.pattern_suggester import PatternSuggester
        from modules.semantic.metamodel.dependency_graph import DependencyGraph
        from modules.semantic.metamodel.invariant_detector import InvariantDetector
        
        self.structure_analyzer = StructureAnalyzer(metamodel)
        self.pattern_suggester = PatternSuggester(metamodel)
        self.dependency_graph = DependencyGraph(metamodel)
        self.invariant_detector = InvariantDetector(metamodel)
        print("Advanced semantic analysis enabled (Tier 3): Structure, Patterns, Dependencies, Invariants")
    
    def generate(self, profile: BenchmarkProfile, progress_callback=None) -> List[OCLConstraint]:
        """Generate constraints with full coverage tracking and diversity."""
        logger.info("\n" + "="*80)
        logger.info("GENERATE() - Starting constraint generation")
        logger.info("="*80)
        
        # RNG seeding is handled by the suite controller using the profile seed
        # Reset per-run string constraint tracking
        self._string_constrained_attrs = defaultdict(set)
        
        coverage = CoverageState(self.metamodel, profile)
        total = profile.quantities.invariants
        
        logger.info(f"Target: {total} invariants")
        logger.info(f"Classes: {len(self.metamodel.get_class_names())}")
        logger.info(f"Enabled patterns: {len(profile.library.enabled) if profile.library.enabled else len(self.all_patterns)}")
        logger.info(f"Blacklist size: {len(self.BLACKLIST)} patterns")
        
        print(f"\n=== BENCHMARK GENERATION START ===")
        print(f"Target: {total} invariants")
        print(f"Classes: {len(self.metamodel.get_class_names())}")
        print(f"Enabled patterns: {len(profile.library.enabled) if profile.library.enabled else len(self.all_patterns)}")
        
        # Phase 0: DISABLED - Skip metamodel-driven invariants
        # Using template-based generation only for better semantic quality
        print(f"\n--- Phase 0: DISABLED (template-based generation only) ---")
        
        # Initialize empty constraint list and class quota
        constraints = []
        class_quota = {c: 0 for c in self.metamodel.get_class_names()}
        
        # Phase 1: Family-based generation
        logger.info("\nPHASE 1: Family-based generation")
        plan = self._plan_families(profile)
        logger.debug(f"Family plan: {plan}")
        print(f"\nFamily plan: {plan}")
        
        for family, count in sorted(plan.items(), key=lambda x: -x[1]):
            logger.debug(f"\n--- Processing family '{family}' (need {count} constraints) ---")
            patterns = self._get_enabled_patterns(family, profile)
            if not patterns:
                logger.warning(f"Skipping family {family}: no enabled patterns")
                print(f"  Skipping family {family}: no enabled patterns")
                continue
            
            logger.info(f"Generating {count} constraints for family '{family}' ({len(patterns)} patterns available)")
            print(f"\n  Generating {count} constraints for family '{family}' ({len(patterns)} patterns available)")
            
            for i in range(count):
                if len(constraints) >= total:
                    break
                
                # Select context with room (enhanced with complexity weighting)
                logger.debug(f"  Attempt {i+1}/{count}: Selecting context...")
                context = self._select_context(class_quota, profile, coverage)
                if not context:
                    logger.debug(f"  Attempt {i+1}/{count}: No context available (quota exhausted)")
                    logger.debug(f"Attempt {i+1}/{count}: No context available (quota exhausted)")
                    continue
                
                logger.debug(f"  Attempt {i+1}/{count}: Selected context '{context}' (quota: {class_quota[context]})")
                
                # Sample pattern with weights (enhanced with semantic suggestions)
                logger.debug(f"  Attempt {i+1}/{count}: Sampling pattern from {len(patterns)} options...")
                pattern = self._weighted_sample(patterns, profile, context=context)
                logger.debug(f"  Attempt {i+1}/{count}: Selected pattern '{pattern.id}'")
                
                # Check if pattern is applicable to this context before trying
                if not self._is_pattern_applicable(pattern, context):
                    logger.debug(f"  Attempt {i+1}/{count}: Pattern '{pattern.id}' not applicable to '{context}' - skipping")
                    # Skip silently - pattern not compatible with this class structure
                    continue
                
                logger.debug(f"  Attempt {i+1}/{count}: Pattern '{pattern.id}' is applicable to '{context}'")
                
                # Generate
                try:
                    logger.debug(f"  Attempt {i+1}/{count}: Generating parameters...")
                    params = self._gen_params(pattern, context)
                    logger.debug(f"  Attempt {i+1}/{count}: Parameters: {params}")
                    
                    logger.debug(f"  Attempt {i+1}/{count}: Generating constraint...")
                    c = self.generator.generate(pattern.id, context, params)
                    logger.debug(f"  Attempt {i+1}/{count}: Generated: {c.ocl}")
                    
                    # A: Track successful generation
                    self.pattern_success_counts[(pattern.id, context)] += 1
                    logger.debug(f"  Attempt {i+1}/{count}: Pattern succeeded")
                    
                    # Check diversity (AST similarity)
                    logger.debug(f"  Attempt {i+1}/{count}: Checking diversity...")
                    if not self._is_diverse(c, constraints, profile):
                        logger.debug(f"  Attempt {i+1}/{count}: Failed diversity check - too similar to existing")
                        continue
                    
                    logger.debug(f"  Attempt {i+1}/{count}: Diversity check passed")
                    
                    # E: Context-aware semantic validation (avoid adding invalid constraints)
                    if self.semantic_validator:
                        logger.debug(f"  Attempt {i+1}/{count}: Running semantic validation...")
                        is_valid, reason = self.semantic_validator.validate_parameters(
                            pattern.id, context, params
                        )
                        if not is_valid:
                            logger.debug(f"  Attempt {i+1}/{count}: Semantic validation failed: {reason}")
                            # Semantically dubious constraint - skip silently
                            continue
                        logger.debug(f"  Attempt {i+1}/{count}: Semantic validation passed")
                    
                    # Pattern repetition check DISABLED for maximum generation
                    # This allows the same pattern to be used multiple times per class
                    # pattern_context_key = f"{pattern.id}_{context}"
                    # pattern_usage_count = sum(1 for existing in constraints 
                    #                          if f"{existing.pattern_id}_{existing.context}" == pattern_context_key)
                    # if pattern_usage_count >= 2:
                    #     continue
                    
                    constraints.append(c)
                    self._track_string_constraints(pattern, params, context)
                    coverage.add_constraint(c)
                    class_quota[context] += 1
                    logger.debug(f"  Attempt {i+1}/{count}: CONSTRAINT ADDED - Total: {len(constraints)}/{total}")
                    logger.debug(f"    Pattern: {pattern.id}")
                    logger.debug(f"    Context: {context}")
                    logger.debug(f"    Expression: {c.ocl}")
                    if progress_callback:
                        progress_callback(len(constraints), total, coverage.score())
                except Exception as e:
                    # A: Track pattern failure
                    self.pattern_fail_counts[(pattern.id, context)] += 1
                    logger.debug(f"  Attempt {i+1}/{count}: GENERATION FAILED")
                    logger.debug(f"    Pattern: {pattern.id}")
                    logger.debug(f"    Context: {context}")
                    logger.debug(f"    Error: {type(e).__name__}: {e}")
                    logger.debug(f"Attempt {i+1}/{count}: Failed to generate - {type(e).__name__}: {e}")
        
        # Phase 2: Coverage-driven backfill
        print(f"\n--- Phase 2: Coverage-driven backfill (target: {total - len(constraints)} more constraints) ---")
        phase2_attempts = 0
        phase2_generated = 0
        
        while len(constraints) < total:
            phase2_attempts += 1
            
            deficits = coverage.deficits()
            if not deficits:
                print(f"  Phase 2 stopped: No coverage deficits found (generated {phase2_generated} in {phase2_attempts} attempts)")
                break
            
            # Pick deficit and find pattern to address it
            target_name, _, _ = deficits[0]
            pattern = self._pattern_for_deficit(target_name, profile)
            if not pattern:
                print(f"  Phase 2 stopped: No pattern found for deficit '{target_name}' (generated {phase2_generated} in {phase2_attempts} attempts)")
                break
            
            context = self._select_context(class_quota, profile, coverage)
            if not context:
                print(f"  Phase 2 stopped: No context available - all classes at quota (generated {phase2_generated} in {phase2_attempts} attempts)")
                break
            
            try:
                params = self._gen_params(pattern, context)
                c = self.generator.generate(pattern.id, context, params)
                if self._is_diverse(c, constraints, profile):
                    constraints.append(c)
                    self._track_string_constraints(pattern, params, context)
                    coverage.add_constraint(c)
                    class_quota[context] += 1
                    phase2_generated += 1
                    
                    if phase2_generated % 10 == 0:
                        print(f"  Phase 2: Generated {phase2_generated} constraints (total: {len(constraints)}/{total})")
                    
                    if progress_callback:
                        progress_callback(len(constraints), total, coverage.score())
            except Exception as e:
                # A: Track pattern failure
                self.pattern_fail_counts[(pattern.id, context)] += 1
                print(f"  Phase 2 stopped: Generation error - {type(e).__name__}: {str(e)[:60]} (generated {phase2_generated} in {phase2_attempts} attempts)")
                break
        
        if phase2_generated > 0:
            print(f"  Phase 2 complete: Generated {phase2_generated} additional constraints")
        
        # Phase 2.5: Simple backfill to reach target count (no coverage requirements)
        if len(constraints) < total:
            print(f"\n--- Phase 2.5: Simple backfill (need {total - len(constraints)} more) ---")
            phase25_generated = 0
            max_attempts = (total - len(constraints)) * 10  # Try up to 10x the needed amount
            
            for attempt in range(max_attempts):
                if len(constraints) >= total:
                    break
                
                # C: Use centralized blacklist
                pattern = random.choice([p for p in self.all_patterns if p.id not in self.BLACKLIST])
                
                # Pick random context
                context = self._select_context(class_quota, profile, coverage)
                if not context:
                    # All classes at max, can't generate more
                    print(f"  Phase 2.5 stopped: All classes at quota (generated {phase25_generated} in {attempt+1} attempts)")
                    break
                
                try:
                    params = self._gen_params(pattern, context)
                    c = self.generator.generate(pattern.id, context, params)
                    
                    # Check diversity
                    if self._is_diverse(c, constraints, profile):
                        constraints.append(c)
                        self._track_string_constraints(pattern, params, context)
                        coverage.add_constraint(c)
                        class_quota[context] += 1
                        phase25_generated += 1
                        
                        if phase25_generated % 10 == 0:
                            print(f"  Phase 2.5: Generated {phase25_generated} constraints (total: {len(constraints)}/{total})")
                except Exception:
                    # Silently continue on errors
                    continue
            
            if phase25_generated > 0:
                print(f"  Phase 2.5 complete: Generated {phase25_generated} additional constraints")
        
        # Phase 3: Redundancy pruning (if enabled)
        if profile.redundancy.implication_mode != "off":
            before = len(constraints)
            constraints = self._prune_redundant(constraints, profile)
            print(f"\nPhase 3: Pruned {before - len(constraints)} redundant constraints")
        
        print(f"\n=== GENERATION COMPLETE ===")
        print(f"Total generated: {len(constraints)} / {total} requested")
        
        # A: Print failure statistics
        self._print_failure_stats()
        
        return constraints
    
    def _plan_families(self, profile: BenchmarkProfile) -> Dict[str, int]:
        """Convert family percentages to counts."""
        total = profile.quantities.invariants
        pct = profile.quantities.families_pct
        s = sum(max(0, int(v)) for v in pct.values()) or 1
        weights = {k: max(0, int(v)) / s for k, v in pct.items()}
        counts = {k: int(round(total * weights.get(k, 0))) for k in FAMILY_KEYS}
        
        # Adjust rounding
        diff = total - sum(counts.values())
        if diff != 0:
            top = max(FAMILY_KEYS, key=lambda k: weights.get(k, 0))
            counts[top] = max(0, counts[top] + diff)
        return counts
    
    def _get_enabled_patterns(self, family: str, profile: BenchmarkProfile):
        """Get enabled patterns for family."""
        all_in_family = self.patterns_by_family.get(family, [])
        if not profile.library.enabled:
            return all_in_family
        return [p for p in all_in_family if p.id in profile.library.enabled]
    
    def _is_pattern_applicable(self, pattern, context: str) -> bool:
        """Quick check if pattern is applicable to the given context class."""
        # Special applicability checks for known problematic patterns
        if pattern.id == 'acyclicity':
            # Require a self-association so closure stays in same type
            self_assocs = [
                a for a in self.metamodel.get_associations_for(context)
                if a.target_class == context
            ]
            if not self_assocs:
                return False
        if pattern.id in ['association_exists', 'type_check']:
            # Require at least one single-valued association
            if not self.metamodel.get_single_associations(context):
                return False
        if pattern.id == 'collection_collectNested':
            # Require a collection association whose target has a collection association
            for assoc in self.metamodel.get_collection_associations(context):
                if self.metamodel.get_collection_associations(assoc.target_class):
                    break
            else:
                return False

        for param in pattern.parameters:
            if not param.required:
                continue
            options = param.get_options_for_context(self.metamodel, context, {})
            if not options and param.default is None:
                return False
        return True

    def _weighted_sample(self, patterns, profile: BenchmarkProfile, context: str = None):
        """Sample pattern with weights using: base * universal * solver * relevance * history."""
        logger.debug(f"      _weighted_sample() - Input: {len(patterns)} patterns")
        # C: Use centralized blacklist
        logger.debug(f"      Applying blacklist ({len(self.BLACKLIST)} patterns)...")
        
        # Universal patterns that work with ANY model (71 total)
        UNIVERSAL_PATTERNS = {
            # Basic null checks
            'attribute_not_null_simple',
            # Comparisons
            'two_attributes_not_equal',
            # Numeric
            'numeric_positive', 'numeric_non_negative', 'numeric_bounded',
            'numeric_greater_than_value', 'numeric_less_than_value', 'numeric_even', 'numeric_odd',
            'numeric_multiple_of', 'numeric_sum_constraint', 'numeric_difference_constraint',
            'numeric_abs_bounded', 'numeric_product_constraint',
            # String
            'string_min_length', 'string_max_length', 'string_exact_length',
            'string_contains_substring', 'string_starts_with', 'string_to_upper_equals',
            'attribute_value_in_set',
            # Collection
            'collection_not_empty_simple', 'collection_has_size',
            'collection_size_range', 'collection_not_empty_check', 'collection_min_size',
            'collection_max_size', 'collection_empty_check',
            # Boolean
            'boolean_is_true', 'boolean_is_false',
            # Logic
            'implies_simple', 'implies_reverse',
            'all_attributes_defined', 'three_attributes_defined',
            # Advanced collections (10 new)
            'collection_including', 'collection_excluding', 'collection_sortedBy', 'collection_sum',
            'collection_isUnique_attr', 'collection_first', 'collection_last', 'collection_any_match',
            'collection_collectNested',
            # Conditionals (2 new)
            'conditional_if_then_else', 'conditional_value_selection',
            # Collection conversions & sequence ops (5 new)
            'collection_asSet', 'collection_asSequence', 'collection_at_index', 'collection_indexOf'
        }
        
        # Filter out blacklisted patterns
        filtered_patterns = [p for p in patterns if p.id not in self.BLACKLIST]
        blacklisted_count = len(patterns) - len(filtered_patterns)
        if blacklisted_count > 0:
            logger.debug(f"      Filtered out {blacklisted_count} blacklisted patterns")
            blacklisted_in_input = [p.id for p in patterns if p.id in self.BLACKLIST]
            logger.debug(f"      Blacklisted patterns in input: {blacklisted_in_input}")
        logger.debug(f"      After blacklist: {len(filtered_patterns)} patterns remaining")
        
        # Filter out patterns with invalid templates (if validator available)
        if self.pattern_template_validator:
            valid_patterns = []
            for p in filtered_patterns:
                template = getattr(p, 'template', '')
                if template:
                    is_valid, reason = self.pattern_template_validator(p.id, template)
                    if is_valid:
                        valid_patterns.append(p)
                else:
                    # No template, include it
                    valid_patterns.append(p)
            filtered_patterns = valid_patterns
        
        if not filtered_patterns:
            # Fallback: use original if all filtered out (shouldn't happen)
            filtered_patterns = patterns
        
        weights = []
        for p in filtered_patterns:
            # (1) User preference: explicit pattern weights (family mix is handled upstream)
            w_base = profile.library.weights.get(p.id, 1.0)

            # (2) Universal applicability boost
            w_gen = 5.0 if p.id in UNIVERSAL_PATTERNS else 1.0

            # (3) Solver-friendly boost (only when verification is enabled)
            w_solver = 2.0 if (self.verification_enabled and p.id in self.SMT_VERIFIED_PATTERNS) else 1.0

            # (4) Relevance to the class (soft applicability score)
            w_rel = self._compute_relevance_weight(p, context) if context else 1.0

            # (5) Run-time adaptation from success/failure history
            w_hist = 1.0
            if context:
                key = (p.id, context)
                success = self.pattern_success_counts.get(key, 0)
                failure = self.pattern_fail_counts.get(key, 0)
                if success + failure >= 3:
                    success_rate = success / (success + failure)
                    if success_rate > 0.7:
                        w_hist = 1.5
                    elif success_rate < 0.3:
                        w_hist = 0.3

            w = w_base * w_gen * w_solver * w_rel * w_hist

            # TIER 3: Boost semantically suggested patterns (3x boost)
            if context and hasattr(self, 'pattern_suggester'):
                try:
                    suggestions = self.pattern_suggester.suggest_for_class(context)
                    suggested_ids = {s.pattern_id for s in suggestions if s.priority in ['critical', 'high']}
                    if p.id in suggested_ids:
                        w *= 3.0
                except Exception:
                    pass

            weights.append(w)

        # If relevance weighting zeroed out candidates, fall back to uniform weights
        if context:
            filtered = [(p, w) for p, w in zip(filtered_patterns, weights) if w > 0]
            if filtered:
                filtered_patterns, weights = zip(*filtered)
                filtered_patterns = list(filtered_patterns)
                weights = list(weights)
            else:
                weights = [1.0 for _ in filtered_patterns]
        else:
            if all(w == 0 for w in weights):
                weights = [1.0 for _ in filtered_patterns]
        
        total_w = sum(weights) or 1
        probs = [w / total_w for w in weights]
        return random.choices(filtered_patterns, weights=probs, k=1)[0]

    def _compute_relevance_weight(self, pattern, context: str) -> float:
        """Soft relevance weight based on availability of required parameters."""
        if not context:
            return 1.0

        required = [p for p in pattern.parameters if getattr(p, 'required', False)]
        if not required:
            return 1.0

        scores = []
        for param in required:
            options = param.get_options_for_context(self.metamodel, context, {})
            if (not options) and param.default is None:
                return 0.0  # Not applicable to this class
            if options:
                cap = 5
                scores.append(min(len(options), cap) / cap)
            else:
                scores.append(1.0)

        return sum(scores) / max(1, len(scores))
    
    def _select_context(self, class_quota, profile, coverage: CoverageState) -> Optional[str]:
        """Select context class with room, preferring under-covered and high-complexity classes."""
        classes = self.metamodel.get_class_names()
        max_cap = profile.quantities.per_class_max
        
        # TIER 3: Prioritize high-complexity classes for better constraint coverage
        if profile.redundancy.novelty_boost and hasattr(self, 'structure_analyzer'):
            candidates = [c for c in classes if class_quota[c] < max_cap]
            if candidates:
                # Prefer classes not yet used
                unused = [c for c in candidates if c not in coverage.classes_used]
                if unused:
                    candidates = unused
                
                # Weight selection by complexity (higher complexity = higher chance)
                try:
                    candidates_with_complexity = []
                    for c in candidates:
                        metrics = self.structure_analyzer.analyze_class_complexity(c)
                        if metrics:
                            candidates_with_complexity.append((c, metrics['complexity_score']))
                        else:
                            candidates_with_complexity.append((c, 1.0))
                    
                    # Weight by complexity
                    if candidates_with_complexity:
                        total_complexity = sum(score for _, score in candidates_with_complexity) or 1
                        weights = [max(1.0, score) / total_complexity for _, score in candidates_with_complexity]
                        return random.choices([c for c, _ in candidates_with_complexity], weights=weights, k=1)[0]
                except Exception:
                    pass
                
                return random.choice(candidates)
        
        # Fallback: Random with cap
        for _ in range(10):
            c = random.choice(classes)
            if class_quota[c] < max_cap:
                return c
        return None
    
    def _gen_params(self, pattern, context: str) -> Dict:
        """Generate parameters for pattern.
        
        Raises:
            ValueError: If required parameters cannot be populated or semantic validation fails
        """
        def _number_default_for(name: str) -> int:
            """Generate a reasonable numeric default based on parameter name."""
            if name in ['max_length', 'min_length', 'length']:
                if name == 'max_length':
                    return random.randint(10, 100)
                if name == 'min_length':
                    return random.randint(1, 20)
                return random.randint(5, 50)
            if name in ['index', 'position']:
                return random.randint(1, 5)
            if name in ['start', 'substring_start']:
                return random.randint(1, 5)
            if name in ['end', 'substring_end']:
                # End must be > start if start exists
                start_val = params.get('start', params.get('substring_start', 1))
                return random.randint(start_val + 1, start_val + 10)
            if name in ['prefix_length', 'suffix_length']:
                return random.randint(2, 10)
            if name == 'size' or 'size' in name.lower():
                return random.randint(1, 3)
            if name in ['threshold', 'value', 'min_value', 'max_value']:
                return random.randint(1, 100)
            return random.randint(0, 10)

        logger.debug(f"    _gen_params() - Pattern: {pattern.id}, Context: {context}")
        logger.debug(f"    Pattern has {len(pattern.parameters)} parameters")
        params = {}
        for param in pattern.parameters:
            options = param.get_options_for_context(self.metamodel, context, params)

            # Override options for patterns requiring special association handling
            if pattern.id in ['association_exists', 'type_check'] and param.options == 'associations':
                options = [a.ref_name for a in self.metamodel.get_single_associations(context)]

            if pattern.id == 'type_check' and param.name == 'type':
                assoc_name = params.get('attribute')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [assoc.target_class]
                else:
                    options = []

            if pattern.id == 'acyclicity' and param.options == 'associations':
                self_assocs = [
                    a.ref_name for a in self.metamodel.get_associations_for(context)
                    if a.target_class == context
                ]
                options = self_assocs

            if pattern.id == 'collection_collectNested' and param.name == 'nested_collection':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [a.ref_name for a in self.metamodel.get_collection_associations(assoc.target_class)]
                else:
                    options = []

            if pattern.id == 'sum_operation' and param.name == 'attribute':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    # Prefer boolean attributes for select(...) predicate
                    options = [
                        a.name for a in self.metamodel.get_attributes_for(assoc.target_class)
                        if self._normalize_type(a.type) == 'boolean'
                    ]
                else:
                    options = []

            if pattern.id == 'collection_sum' and param.name == 'numeric_attribute':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.name for a in self.metamodel.get_attributes_for(assoc.target_class)
                        if self._normalize_type(a.type) == 'numeric'
                    ]
                else:
                    options = []

            if pattern.id == 'difference_operation' and param.name == 'collection2':
                assoc_name = params.get('collection1')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.ref_name for a in self.metamodel.get_collection_associations(context)
                        if a.target_class == assoc.target_class
                    ]
                else:
                    options = []

            if pattern.id in ['symmetricDifference', 'union_operation', 'intersection_operation', 'includesAll_excludesAll'] and param.name == 'collection2':
                assoc_name = params.get('collection1')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.ref_name for a in self.metamodel.get_collection_associations(context)
                        if a.target_class == assoc.target_class
                    ]
                else:
                    options = []

            if pattern.id == 'collection_including' and param.name == 'element':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.ref_name for a in self.metamodel.get_single_associations(context)
                        if a.target_class == assoc.target_class
                    ]
                else:
                    options = []

            if pattern.id == 'flatten_operation' and param.name == 'nested_collection':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.ref_name for a in self.metamodel.get_collection_associations(assoc.target_class)
                    ]
                else:
                    options = []

            if pattern.id == 'division_modulo' and param.name == 'attr1':
                options = [
                    a.name for a in self.metamodel.get_attributes_for(context)
                    if self._is_integer_type(a.type)
                ]

            if pattern.id in ['sortedBy', 'collection_sortedBy'] and param.name == 'attribute':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.name for a in self.metamodel.get_attributes_for(assoc.target_class)
                        if self._normalize_type(a.type) in ['numeric', 'string']
                    ]
                else:
                    options = []
            
            # FILTER: Date filter DISABLED to maximize constraint generation
            # Dates are typed as EString and have separate encoder support
            # if options and param.options == 'string_attributes':
            #     filtered_string_options = []
            #     for opt in options:
            #         opt_lower = opt.lower()
            #         if any(x in opt_lower for x in ['date', 'time', 'at', 'when', 'timestamp']):
            #             continue
            #         filtered_string_options.append(opt)
            #     if filtered_string_options:
            #         options = filtered_string_options
            
            if options:
                # Avoid null-checks on string attributes already constrained by string patterns
                if pattern.id in ['attribute_null_check', 'null_check'] and param.name == 'attribute':
                    blocked = self._string_constrained_attrs.get(context, set())
                    if blocked:
                        options = [opt for opt in options if opt not in blocked]
                        if not options:
                            raise ValueError(
                                "Parameter validation failed: no eligible attributes for null-check "
                                "(string attributes already constrained)"
                            )
                # Special handling for patterns that compare two things
                # Ensure they're different to avoid tautologies (e.g., self.x = self.x)
                if param.name in ['second_attribute', 'second_string', 'second_numeric', 'second_attr', 'attribute2', 'third_attribute', 'third_numeric']:
                    # Get the "first" parameter name
                    first_param_name = param.name.replace('second_', 'first_').replace('third_', 'first_').replace('attribute2', 'attribute1').replace('_attr', '')
                    if 'attribute' in first_param_name and first_param_name not in params:
                        first_param_name = 'attribute'  # Fallback
                    
                    if first_param_name in params:
                        # For third parameter, also check against second parameter
                        excluded_values = [params[first_param_name]]
                        if 'third' in param.name:
                            second_param_name = param.name.replace('third_', 'second_')
                            if second_param_name in params:
                                excluded_values.append(params[second_param_name])
                        
                        # Filter out excluded values
                        options = [opt for opt in options if opt not in excluded_values]
                        
                        # Additional validation: filter out attributes with incompatible types
                        first_attr = params[first_param_name]
                        first_type = self._get_attribute_type(context, first_attr)
                        filtered_options = []
                        for opt in options:
                            opt_type = self._get_attribute_type(context, opt)
                            if first_type and opt_type and not self._types_compatible(first_type, opt_type):
                                continue
                            if not self._attribute_names_compatible(first_attr, opt):
                                continue
                            
                            filtered_options.append(opt)
                        
                        # Apply filtered options (even if empty to force failure)
                        options = filtered_options
                        
                        # TIER 2: Apply semantic filtering for attribute pairs
                        if hasattr(self, 'semantic_validator') and self.semantic_validator:
                            options = self._filter_semantically_valid_options(
                                pattern.id, first_param_name, params[first_param_name], 
                                param.name, options, context
                            )
                        
                        if not options:
                            # If no valid option available, fail
                            raise ValueError(f"Parameter validation failed: {param.label} must differ from {first_param_name} and be type-compatible")
                
                params[param.name] = random.choice(options)
            elif param.required and param.default is not None:
                # Required parameter with no options but has default (e.g., iterator variable)
                params[param.name] = param.default
            elif param.required:
                # No options/default; try to synthesize required numeric values
                t = getattr(param.type, "value", str(param.type))
                if t == "number":
                    params[param.name] = _number_default_for(param.name)
                else:
                    # Cannot populate required parameter - pattern not applicable to this context
                    raise ValueError(f"Parameter validation failed: {param.label} is required")
            else:
                # Optional parameter - use defaults
                t = getattr(param.type, "value", str(param.type))
                if t == "number":
                    params[param.name] = _number_default_for(param.name)
                elif t == "boolean":
                    params[param.name] = random.choice(["true", "false"])
                elif t == "text":
                    params[param.name] = "value"
                else:
                    # Use default if available, otherwise fail for missing metamodel-dependent params
                    if param.default is not None:
                        params[param.name] = param.default
                    elif param.options in ['string_attributes', 'numeric_attributes', 'attributes', 
                                          'collection_associations', 'single_associations']:
                        # Metamodel-dependent parameter with no options - pattern not applicable
                        raise ValueError(f"Parameter '{param.label}' requires {param.options} but none found for {context}")
                    else:
                        # Generic parameter - use empty string as fallback
                        params[param.name] = ""
        
        # Universal self-comparison check (catches all patterns)
        logger.debug(f"    _gen_params() - Running self-comparison check...")
        self._check_self_comparison(pattern.id, params)
        logger.debug(f"    _gen_params() - Self-comparison check passed")

        # Type compatibility check for attribute pairs
        logger.debug(f"    _gen_params() - Running attribute type compatibility check...")
        self._validate_attribute_pairs(context, pattern.id, params)
        logger.debug(f"    _gen_params() - Attribute type compatibility check passed")
        
        # Pattern-specific parameter validation
        logger.debug(f"    _gen_params() - Running parameter validation...")
        self._validate_pattern_params(pattern.id, params)
        logger.debug(f"    _gen_params() - Parameter validation passed")
        
        # TIER 2: Final semantic validation of complete parameter set
        if hasattr(self, 'semantic_validator') and self.semantic_validator:
            is_valid, reason = self.semantic_validator.validate_parameters(pattern.id, context, params)
            if not is_valid:
                raise ValueError(f"Semantic validation failed: {reason}")
        
        return params

    def _track_string_constraints(self, pattern, params: Dict, context: str) -> None:
        """Track string attributes already constrained by string patterns."""
        category = getattr(pattern.category, "value", str(pattern.category))
        is_string_pattern = category == "string" or pattern.id.startswith("string_")
        if not is_string_pattern:
            return

        for param in pattern.parameters:
            if param.options == "string_attributes":
                attr = params.get(param.name)
                if attr:
                    self._string_constrained_attrs[context].add(attr)
            elif param.name in ("attribute", "string_attribute", "first_string", "second_string"):
                attr = params.get(param.name)
                if attr:
                    attr_type = self._get_attribute_type(context, attr)
                    if attr_type and self._normalize_type(attr_type) == "string":
                        self._string_constrained_attrs[context].add(attr)
    
    def _check_self_comparison(self, pattern_id: str, params: Dict):
        """Check for self-comparison tautologies across all attribute parameters.
        
        Raises:
            ValueError: If same attribute used in comparison
        """
        logger.debug(f"      _check_self_comparison() - Pattern: {pattern_id}")
        logger.debug(f"      Parameters: {params}")
        
        # Collect all attribute parameter values
        attr_params = {}
        for key, value in params.items():
            # Look for attribute parameters (not numeric values or operators)
            if 'attribute' in key.lower() or 'attr' in key:
                # Skip if it's a boolean/operator value
                if value not in ['true', 'false', None, ''] and not isinstance(value, (int, float)):
                    attr_params[key] = value
        
        logger.debug(f"      Found {len(attr_params)} attribute parameters: {attr_params}")
        
        # Check for duplicates
        if len(attr_params) >= 2:
            values = list(attr_params.values())
            unique_values = set(values)
            logger.debug(f"      Checking for duplicates: {len(values)} values, {len(unique_values)} unique")
            if len(unique_values) < len(values):
                # Found duplicate - identify which
                for i, val in enumerate(values):
                    if values.count(val) > 1:
                        logger.debug(f"      Self-comparison detected: attribute '{val}' used multiple times")
                        raise ValueError(f"Self-comparison detected: attribute '{val}' used multiple times in {pattern_id}")
        else:
            logger.debug(f"      No duplicate check needed (only {len(attr_params)} attribute params)")

    def _validate_attribute_pairs(self, context: str, pattern_id: str, params: Dict):
        """Ensure attribute pairs are type- and name-compatible.

        Raises:
            ValueError: If attribute pairs are incompatible
        """
        attr_names = {a.name for a in self.metamodel.get_attributes_for(context)}
        pairs = [
            ('first_attribute', 'second_attribute'),
            ('attribute1', 'attribute2'),
            ('first_attr', 'second_attr'),
            ('first_numeric', 'second_numeric'),
            ('first_string', 'second_string'),
            ('first_attribute', 'third_attribute'),
            ('second_attribute', 'third_attribute'),
            ('first_numeric', 'third_numeric'),
            ('second_numeric', 'third_numeric'),
        ]

        for left_key, right_key in pairs:
            if left_key not in params or right_key not in params:
                continue
            left = params[left_key]
            right = params[right_key]
            if left not in attr_names or right not in attr_names:
                continue
            left_type = self._get_attribute_type(context, left)
            right_type = self._get_attribute_type(context, right)
            if left_type and right_type and not self._types_compatible(left_type, right_type):
                raise ValueError(
                    f"Parameter validation failed: {left_key} and {right_key} must be type-compatible"
                )
            if not self._attribute_names_compatible(left, right):
                raise ValueError(
                    f"Parameter validation failed: {left_key} and {right_key} must be semantically compatible"
                )
    
    def _validate_pattern_params(self, pattern_id: str, params: Dict):
        """Validate pattern-specific parameter constraints.
        
        Raises:
            ValueError: If parameters don't meet pattern-specific constraints
        """
        logger.debug(f"      _validate_pattern_params() - Pattern: {pattern_id}")
        logger.debug(f"      Parameters to validate: {params}")
        
        # String length patterns
        if pattern_id in ['string_max_length', 'string_min_length', 'string_exact_length',
                         'string_not_empty', 'string_operation']:
            logger.debug(f"      Checking string length constraints...")
            if 'max_length' in params and params['max_length'] < 1:
                logger.debug(f"      Invalid max_length: {params['max_length']}")
                raise ValueError(f"max_length must be at least 1, got {params['max_length']}")
            if 'min_length' in params and params['min_length'] < 1:
                logger.debug(f"      Invalid min_length: {params['min_length']}")
                raise ValueError(f"min_length must be at least 1, got {params['min_length']}")
            if 'length' in params and params['length'] < 1:
                logger.debug(f"      Invalid length: {params['length']}")
                raise ValueError(f"length must be at least 1, got {params['length']}")
            logger.debug(f"      String length constraints OK")
        
        # Substring patterns - ensure valid ranges
        if pattern_id == 'string_starts_with' and 'prefix_length' in params:
            logger.debug(f"      Checking prefix_length for string_starts_with...")
            # For substring(1, prefix_length), prefix_length must be > 1 to extract characters
            if params['prefix_length'] <= 1:
                logger.debug(f"      Invalid prefix_length: {params['prefix_length']} (must be > 1)")
                raise ValueError(f"prefix_length ({params['prefix_length']}) must be > 1 to extract characters")
            logger.debug(f"      prefix_length OK: {params['prefix_length']}")
        
        if pattern_id in ['string_operation'] and 'start' in params and 'end' in params:
            logger.debug(f"      Checking substring start/end...")
            if params['start'] >= params['end']:
                logger.debug(f"      Invalid substring range: start={params['start']}, end={params['end']}")
                raise ValueError(f"substring start ({params['start']}) must be less than end ({params['end']})")
            logger.debug(f"      Substring range OK: start={params['start']}, end={params['end']}")
        
        # Numeric range patterns - ensure min < max
        if pattern_id in ['numeric_bounded', 'range_constraint']:
            if 'min_value' in params and 'max_value' in params:
                if params['min_value'] >= params['max_value']:
                    raise ValueError(f"min_value ({params['min_value']}) must be less than max_value ({params['max_value']})")
        
        # Collection size patterns
        if pattern_id in ['size_constraint', 'collection_min_size', 'collection_max_size', 
                         'collection_has_size', 'collection_size_range']:
            if 'size' in params and params['size'] < 0:
                raise ValueError(f"size must be non-negative, got {params['size']}")
            if 'min_size' in params and params['min_size'] < 0:
                raise ValueError(f"min_size must be non-negative, got {params['min_size']}")
            if 'max_size' in params and params['max_size'] < 0:
                raise ValueError(f"max_size must be non-negative, got {params['max_size']}")
            
            # Ensure min_size <= max_size for range patterns
            if 'min_size' in params and 'max_size' in params:
                if params['min_size'] > params['max_size']:
                    raise ValueError(f"min_size ({params['min_size']}) must not exceed max_size ({params['max_size']})")
        
        # Modulo/divisor patterns - ensure divisor > 0
        if pattern_id in ['numeric_even', 'numeric_odd', 'numeric_multiple_of', 'division_modulo']:
            if 'divisor' in params and params['divisor'] <= 0:
                raise ValueError(f"divisor must be positive, got {params['divisor']}")
    
    def _is_diverse(self, candidate: OCLConstraint, existing: List[OCLConstraint], profile: BenchmarkProfile) -> bool:
        """Check if candidate is diverse enough."""
        threshold = profile.redundancy.similarity_threshold
        for e in existing[-min(20, len(existing)):]:  # check last 20
            if similarity(candidate, e) > threshold:
                return False
        return True
    
    def _filter_semantically_valid_options(self, pattern_id: str, first_param: str, 
                                            first_value: str, second_param: str, 
                                            options: List[str], context: str) -> List[str]:
        """Filter options to keep only semantically valid combinations.
        
        Args:
            pattern_id: Pattern identifier
            first_param: Name of first parameter
            first_value: Value of first parameter
            second_param: Name of second parameter
            options: Available options for second parameter
            context: Context class name
            
        Returns:
            Filtered list of valid options
        """
        if not self.semantic_validator:
            return options
        
        # For two-attribute patterns, filter based on semantic compatibility
        if pattern_id in ['two_attributes_equal', 'two_attributes_not_equal', 'numeric_comparison']:
            from semantic_rules import is_valid_equality_pair
            
            valid_options = []
            for option in options:
                # Check if this pair makes semantic sense
                if is_valid_equality_pair(first_value, option, context):
                    valid_options.append(option)
            
            # If all options filtered out, return at least one to avoid complete failure
            # (Better to generate one weird constraint than fail completely)
            if not valid_options and options:
                return [options[0]]  # Fallback: return first option
            
            return valid_options if valid_options else options
        
        return options
    
    def _pattern_for_deficit(self, target_name: str, profile: BenchmarkProfile):
        """Find pattern that addresses a coverage deficit."""
        # C: Use centralized blacklist
        available_patterns = [p for p in self.all_patterns if p.id not in self.BLACKLIST]
        if not available_patterns:
            available_patterns = self.all_patterns  # Fallback
        
        # Parse target
        if target_name.startswith("op:"):
            op = target_name.split(":")[1]
            # Find patterns that use this operator
            for p in available_patterns:
                if op.lower() in p.template.lower():
                    return p
        elif target_name.startswith("hops:"):
            # Prefer navigation patterns
            nav_patterns = [p for p in self.patterns_by_family.get("navigation", []) if p.id not in self.BLACKLIST]
            if nav_patterns:
                return random.choice(nav_patterns)
        elif target_name.startswith("depth:"):
            # Prefer quantified patterns
            quant_patterns = [p for p in self.patterns_by_family.get("quantified", []) if p.id not in self.BLACKLIST]
            if quant_patterns:
                return random.choice(quant_patterns)
        
        return random.choice(available_patterns) if available_patterns else None
    
    def _prune_redundant(self, constraints: List[OCLConstraint], profile: BenchmarkProfile) -> List[OCLConstraint]:
        """Remove redundant constraints (greedy mode)."""
        if profile.redundancy.implication_mode == "greedy":
            # Simple greedy: remove high-similarity pairs
            kept = []
            for c in constraints:
                if all(similarity(c, k) < 0.95 for k in kept):
                    kept.append(c)
            return kept
        return constraints
    
    def _print_failure_stats(self):
        """A: Print pattern failure statistics for debugging and optimization."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if not self.pattern_fail_counts:
            return
        
        print(f"\nPattern Success/Failure Statistics:")
        
        # Aggregate by pattern across all contexts
        pattern_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        for (pattern_id, context), count in self.pattern_fail_counts.items():
            pattern_stats[pattern_id]['failure'] += count
        
        for (pattern_id, context), count in self.pattern_success_counts.items():
            pattern_stats[pattern_id]['success'] += count
        
        # Sort by failure count
        sorted_patterns = sorted(pattern_stats.items(), 
                                key=lambda x: x[1]['failure'], 
                                reverse=True)
        
        print("\nTop 10 patterns by failure count:")
        for pattern_id, stats in sorted_patterns[:10]:
            success = stats['success']
            failure = stats['failure']
            total = success + failure
            if total > 0:
                success_rate = success / total * 100
                print(f"  {pattern_id}: {failure} failures, {success} successes ({success_rate:.1f}% success rate)")
    
    def generate_sat_unsat_pairs(self, profile: BenchmarkProfile, num_pairs: int = 10) -> List[Tuple[OCLConstraint, OCLConstraint]]:
        """B: Generate paired SAT/UNSAT constraints for evaluation.
        
        For each pair:
        - First constraint is likely SAT (lenient, single condition)
        - Second constraint adds conflicting condition to make it UNSAT
        
        Args:
            profile: Benchmark configuration
            num_pairs: Number of SAT/UNSAT pairs to generate
            
        Returns:
            List of (sat_constraint, unsat_constraint) tuples
        """
        pairs = []
        contexts = self.metamodel.get_class_names()
        
        for _ in range(num_pairs):
            context = random.choice(contexts)
            
            # Generate base SAT constraint using SAT-likely pattern
            sat_patterns = [p for p in self.all_patterns 
                          if p.id in self.SAT_LIKELY_PATTERNS and p.id not in self.BLACKLIST]
            
            if not sat_patterns:
                continue
            
            sat_pattern = random.choice(sat_patterns)
            
            try:
                params = self._gen_params(sat_pattern, context)
                sat_constraint = self.generator.generate(sat_pattern.id, context, params)
                
                # Try to generate conflicting UNSAT constraint
                # Use UNSAT-likely patterns or add conflicting bounds
                unsat_patterns = [p for p in self.all_patterns 
                                if p.id in self.UNSAT_LIKELY_PATTERNS and p.id not in self.BLACKLIST]
                
                if unsat_patterns:
                    unsat_pattern = random.choice(unsat_patterns)
                    unsat_params = self._gen_params(unsat_pattern, context)
                    unsat_constraint = self.generator.generate(unsat_pattern.id, context, unsat_params)
                    
                    pairs.append((sat_constraint, unsat_constraint))
            
            except Exception:
                continue
        
        return pairs
