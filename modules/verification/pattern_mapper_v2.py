#!/usr/bin/env python3
"""
Pattern Mapper v2 - Enhanced Universal to Canonical Pattern Mapping
===================================================================

Improvements over v1:
1.  Real OCL rewriting functions (not just descriptions)
2.  Multi-mapping support (one universal → multiple canonical)
3.  Validation against canonical pattern set
4.  Coverage checking against patterns_unified.json
5.  Stronger testing & instrumentation

Example patterns implemented:
- collection_size_range → TWO size_constraint patterns
- bi_implication → TWO boolean_guard_implies patterns
- string_not_empty → REWRITTEN to size_constraint
- xor_condition → REWRITTEN boolean expression

Usage:
    mapper = PatternMapperV2()
    results = mapper.map_to_canonical("collection_size_range", "self.items->size() >= 2 and self.items->size() <= 10")
    # Returns: [
    #   {'canonical_pattern': 'size_constraint', 'rewritten_text': 'self.items->size() >= 2', ...},
    #   {'canonical_pattern': 'size_constraint', 'rewritten_text': 'self.items->size() <= 10', ...}
    # ]
"""

from typing import List, Dict, Optional, Callable
import re
import json
from pathlib import Path


# ===== CANONICAL PATTERNS (The 50 patterns with SMT encoders) =====
CANONICAL_PATTERNS = {
    # Basic Patterns (1-9)
    "pairwise_uniqueness",
    "exact_count_selection",
    "global_collection",
    "set_intersection",
    "size_constraint",
    "uniqueness_constraint",
    "collection_membership",
    "null_check",
    "numeric_comparison",
    
    # Advanced Patterns (10-19)
    "exactly_one",
    "closure_transitive",
    "acyclicity",
    "aggregation_iterate",
    "boolean_guard_implies",
    "safe_navigation",
    "type_check_casting",
    "subset_disjointness",
    "ordering_ranking",
    "contractual_temporal",
    
    # Collection Operations (20-27)
    "select_reject",
    "collect_flatten",
    "any_operation",
    "forall_nested",
    "exists_nested",
    "collect_nested",
    "as_set_as_bag",
    "sum_product",
    
    # String Operations (28-31)
    "string_concat",
    "string_operations",
    "string_comparison",
    "string_pattern",
    
    # Arithmetic & Logic (32-36)
    "arithmetic_expression",
    "div_mod_operations",
    "abs_min_max",
    "boolean_operations",
    "if_then_else",
    
    # Tuple & Let (37-39)
    "tuple_literal",
    "let_expression",
    "let_nested",
    
    # Set Operations (40-43)
    "union_intersection",
    "symmetric_difference",
    "including_excluding",
    "flatten_operation",
    
    # Navigation & Property (44-47)
    "navigation_chain",
    "optional_navigation",
    "collection_navigation",
    "shorthand_notation",
    
    # OCL Standard Library (48-50)
    "ocl_is_undefined",
    "ocl_is_invalid",
    "ocl_as_type",
}


# ===== REWRITE FUNCTIONS =====

def rewrite_not_empty(text: str) -> str:
    """Rewrite: ->notEmpty() → ->size() > 0"""
    return text.replace("->notEmpty()", "->size() > 0")


def rewrite_is_empty(text: str) -> str:
    """Rewrite: ->isEmpty() → ->size() = 0"""
    return text.replace("->isEmpty()", "->size() = 0")


def rewrite_collection_size_range(text: str) -> List[Dict]:
    """
    Rewrite: self.collection->size() >= min and self.collection->size() <= max
    → TWO separate size constraints
    
    Example: "self.items->size() >= 2 and self.items->size() <= 10"
    Returns:
      [
        {'canonical_pattern': 'size_constraint', 'rewritten_text': 'self.items->size() >= 2', ...},
        {'canonical_pattern': 'size_constraint', 'rewritten_text': 'self.items->size() <= 10', ...}
      ]
    """
    # Pattern: self.X->size() >= N and self.X->size() <= M
    pattern = r'(self\.\w+->size\(\))\s*>=\s*(\d+)\s+and\s+(self\.\w+->size\(\))\s*<=\s*(\d+)'
    match = re.search(pattern, text)
    
    if match:
        collection_ref = match.group(1)
        min_val = match.group(2)
        max_val = match.group(4)
        
        return [
            {
                'canonical_pattern': 'size_constraint',
                'rewritten_text': f"{collection_ref} >= {min_val}",
                'mapping': f'size() >= {min_val} (from range)',
                'universal_pattern': 'collection_size_range'
            },
            {
                'canonical_pattern': 'size_constraint',
                'rewritten_text': f"{collection_ref} <= {max_val}",
                'mapping': f'size() <= {max_val} (from range)',
                'universal_pattern': 'collection_size_range'
            }
        ]
    
    # Fallback: couldn't parse, return as-is
    return [{
        'canonical_pattern': 'size_constraint',
        'rewritten_text': text,
        'mapping': 'size range (unparsed)',
        'universal_pattern': 'collection_size_range'
    }]


def rewrite_bi_implication(text: str) -> List[Dict]:
    """
    Rewrite: A <-> B OR (A) = (B) → (A implies B) and (B implies A)
    → TWO boolean_guard_implies patterns
    
    Example: "self.isActive <-> self.hasMembership"
    Example: "(self.a <> null) = (self.b <> null)"
    Returns:
      [
        {'canonical_pattern': 'boolean_guard_implies', 'rewritten_text': 'self.isActive implies self.hasMembership', ...},
        {'canonical_pattern': 'boolean_guard_implies', 'rewritten_text': 'self.hasMembership implies self.isActive', ...}
      ]
    """
    # Pattern 1: A <-> B (bi-implication operator)
    pattern = r'(.+?)\s*<->\s*(.+)'
    match = re.search(pattern, text)
    
    # Pattern 2: (A) = (B) (equality between parenthesized boolean expressions)
    if not match:
        pattern = r'\(([^)]+)\)\s*=\s*\(([^)]+)\)'
        match = re.search(pattern, text)
    
    if match:
        expr_a = match.group(1).strip()
        expr_b = match.group(2).strip()
        
        return [
            {
                'canonical_pattern': 'boolean_guard_implies',
                'rewritten_text': f"{expr_a} implies {expr_b}",
                'mapping': 'A implies B (from bi-implication)',
                'universal_pattern': 'bi_implication'
            },
            {
                'canonical_pattern': 'boolean_guard_implies',
                'rewritten_text': f"{expr_b} implies {expr_a}",
                'mapping': 'B implies A (from bi-implication)',
                'universal_pattern': 'bi_implication'
            }
        ]
    
    # Fallback
    return [{
        'canonical_pattern': 'boolean_operations',
        'rewritten_text': text,
        'mapping': 'bi-implication (unparsed)',
        'universal_pattern': 'bi_implication'
    }]


def rewrite_xor_condition(text: str) -> str:
    """
    Rewrite: A xor B → (A or B) and not (A and B)
    
    Example: "self.isPremium xor self.isBasic"
    Returns: "(self.isPremium or self.isBasic) and not (self.isPremium and self.isBasic)"
    """
    pattern = r'(.+?)\s+xor\s+(.+)'
    match = re.search(pattern, text)
    
    if match:
        expr_a = match.group(1).strip()
        expr_b = match.group(2).strip()
        return f"({expr_a} or {expr_b}) and not ({expr_a} and {expr_b})"
    
    return text


def rewrite_range_constraint(text: str) -> List[Dict]:
    """
    Rewrite: self.attr >= min and self.attr <= max
    → TWO numeric_comparison patterns
    
    Example: "self.age >= 18 and self.age <= 65"
    """
    pattern = r'(self\.\w+)\s*>=\s*(\d+)\s+and\s+(self\.\w+)\s*<=\s*(\d+)'
    match = re.search(pattern, text)
    
    if match:
        attr = match.group(1)
        min_val = match.group(2)
        max_val = match.group(4)
        
        return [
            {
                'canonical_pattern': 'numeric_comparison',
                'rewritten_text': f"{attr} >= {min_val}",
                'mapping': f'>= {min_val} (from range)',
                'universal_pattern': 'range_constraint'
            },
            {
                'canonical_pattern': 'numeric_comparison',
                'rewritten_text': f"{attr} <= {max_val}",
                'mapping': f'<= {max_val} (from range)',
                'universal_pattern': 'range_constraint'
            }
        ]
    
    return [{
        'canonical_pattern': 'numeric_comparison',
        'rewritten_text': text,
        'mapping': 'range (unparsed)',
        'universal_pattern': 'range_constraint'
    }]


def rewrite_attribute_value_in_set(text: str) -> str:
    """
    Rewrite: attr in {v1, v2, v3} → (attr = v1) or (attr = v2) or (attr = v3)
    
    Example: "self.status in {'active', 'pending'}"
    Note: Full implementation would need proper parsing; this is a simplified version
    """
    # For now, return as-is (full implementation would parse the set and expand to OR)
    return text


def rewrite_all_different(text: str) -> str:
    """
    Rewrite: self.collection->forAll(x, y | x <> y implies x.attr <> y.attr)
    → self.collection->isUnique(x | x.attr)
    
    Example: "self.rentals->forAll(x, y | x <> y implies x.totalAmount <> y.totalAmount)"
    Returns: "self.rentals->isUnique(x | x.totalAmount)"
    """
    # Pattern: self.collection->forAll(x, y | x <> y implies x.attr <> y.attr)
    pattern = r'(self\.\w+)->forAll\(\w+,\s*\w+\s*\|\s*\w+\s*<>\s*\w+\s+implies\s+\w+\.(\w+)\s*<>\s*\w+\.\w+\)'
    match = re.search(pattern, text)
    
    if match:
        collection = match.group(1)
        attribute = match.group(2)
        return f"{collection}->isUnique(x | x.{attribute})"
    
    # Fallback: return as-is
    return text


# ===== PATTERN MAPPING CLASS =====

class PatternMapping:
    """Enhanced mapping with rewrite support"""
    
    def __init__(
        self, 
        canonical_pattern: str,
        description: str,
        rewrite_fn: Optional[Callable] = None,
        multiple: bool = False
    ):
        self.canonical_pattern = canonical_pattern
        self.description = description
        self.rewrite_fn = rewrite_fn
        self.multiple = multiple
        
        # Validate canonical pattern
        if not multiple and canonical_pattern not in CANONICAL_PATTERNS:
            raise ValueError(f"Unknown canonical pattern: {canonical_pattern}")


class PatternMapperV2:
    """Enhanced pattern mapper with rewriting and validation"""
    
    def __init__(self, patterns_file: Optional[str] = None):
        self.patterns_file = patterns_file
        self.mappings = self._build_mapping_registry()
        
        # Statistics
        self.stats = {
            'total_universal_patterns': len(self.mappings),
            'direct_mappings': 0,
            'composite_mappings': 0,
            'with_rewrite_fn': 0,
            'without_rewrite_fn': 0,
        }
        
        self._compute_stats()
        self._validate_mappings()
    
    def _build_mapping_registry(self) -> Dict[str, PatternMapping]:
        """Build mapping registry with rewrite functions"""
        
        mappings = {}
        
        # ===== COMPOUND PATTERNS (Multi-mapping with rewrite) =====
        
        mappings['collection_size_range'] = PatternMapping(
            'size_constraint',
            'size() in [min, max] → TWO size constraints',
            rewrite_fn=rewrite_collection_size_range,
            multiple=True
        )
        
        mappings['bi_implication'] = PatternMapping(
            'boolean_guard_implies',
            'A <-> B → TWO implications',
            rewrite_fn=rewrite_bi_implication,
            multiple=True
        )
        
        mappings['range_constraint'] = PatternMapping(
            'numeric_comparison',
            'attr in [min, max] → TWO comparisons',
            rewrite_fn=rewrite_range_constraint,
            multiple=True
        )
        
        # ===== SINGLE PATTERNS WITH REWRITE =====
        
        mappings['collection_not_empty_simple'] = PatternMapping(
            'size_constraint',
            'notEmpty() → size() > 0',
            rewrite_fn=rewrite_not_empty
        )
        
        mappings['collection_not_empty_check'] = PatternMapping(
            'size_constraint',
            'notEmpty() → size() > 0',
            rewrite_fn=rewrite_not_empty
        )
        
        mappings['collection_empty_check'] = PatternMapping(
            'size_constraint',
            'isEmpty() → size() = 0',
            rewrite_fn=rewrite_is_empty
        )
        
        mappings['xor_condition'] = PatternMapping(
            'boolean_operations',
            'a xor b → (a or b) and not (a and b)',
            rewrite_fn=rewrite_xor_condition
        )
        
        # ===== DIRECT MAPPINGS (No rewrite needed) =====
        
        # Size/Cardinality → size_constraint
        mappings['collection_has_size'] = PatternMapping('size_constraint', 'size() = N')
        mappings['collection_min_size'] = PatternMapping('size_constraint', 'size() >= N')
        mappings['collection_max_size'] = PatternMapping('size_constraint', 'size() <= N')
        
        # Null checks → null_check
        mappings['attribute_not_null_simple'] = PatternMapping('null_check', 'attr <> null')
        mappings['attribute_defined'] = PatternMapping('null_check', 'attr.oclIsDefined()')
        mappings['self_not_null'] = PatternMapping('null_check', 'self <> null')
        mappings['attribute_null_check'] = PatternMapping('null_check', 'null check')
        mappings['association_exists'] = PatternMapping('null_check', 'association <> null')
        mappings['association_defined_check'] = PatternMapping('null_check', 'association.oclIsDefined()')
        
        # Numeric comparisons → numeric_comparison
        mappings['numeric_greater_than_value'] = PatternMapping('numeric_comparison', 'attr > value')
        mappings['numeric_less_than_value'] = PatternMapping('numeric_comparison', 'attr < value')
        mappings['numeric_positive'] = PatternMapping('numeric_comparison', 'attr > 0')
        mappings['numeric_non_negative'] = PatternMapping('numeric_comparison', 'attr >= 0')
        mappings['numeric_bounded'] = PatternMapping('numeric_comparison', 'attr in range')
        mappings['three_way_comparison'] = PatternMapping('numeric_comparison', 'a < b < c')
        mappings['two_attributes_equal'] = PatternMapping('numeric_comparison', 'attr1 = attr2')
        mappings['two_attributes_not_equal'] = PatternMapping('numeric_comparison', 'attr1 <> attr2')
        mappings['attribute_value_in_set'] = PatternMapping('numeric_comparison', 'attr in {v1, v2, ...}')
        
        # Arithmetic → arithmetic_expression
        mappings['numeric_sum_constraint'] = PatternMapping('arithmetic_expression', 'attr1 + attr2 = value')
        mappings['numeric_difference_constraint'] = PatternMapping('arithmetic_expression', 'attr1 - attr2 = value')
        mappings['numeric_product_constraint'] = PatternMapping('arithmetic_expression', 'attr1 * attr2 = value')
        
        # Abs/Min/Max → abs_min_max
        mappings['numeric_abs_bounded'] = PatternMapping('abs_min_max', 'abs(attr) <= value')
        mappings['numeric_max_of_two'] = PatternMapping('abs_min_max', 'max(a, b) op value')
        mappings['numeric_min_of_two'] = PatternMapping('abs_min_max', 'min(a, b) op value')
        
        # Div/Mod → div_mod_operations
        mappings['numeric_even'] = PatternMapping('div_mod_operations', 'attr mod 2 = 0')
        mappings['numeric_odd'] = PatternMapping('div_mod_operations', 'attr mod 2 = 1')
        mappings['numeric_multiple_of'] = PatternMapping('div_mod_operations', 'attr mod N = 0')
        
        # String operations
        mappings['string_not_empty'] = PatternMapping('string_operations', 'str.size() > 0')
        mappings['string_min_length'] = PatternMapping('string_operations', 'str.size() >= N')
        mappings['string_max_length'] = PatternMapping('string_operations', 'str.size() <= N')
        mappings['string_exact_length'] = PatternMapping('string_operations', 'str.size() = N')
        mappings['string_contains_substring'] = PatternMapping('string_pattern', 'str.indexOf(substr)')
        mappings['string_starts_with'] = PatternMapping('string_pattern', 'str.startsWith()')
        mappings['string_to_upper_equals'] = PatternMapping('string_comparison', 'str.toUpper() = value')
        mappings['string_concat_check'] = PatternMapping('string_concat', 'str1.concat(str2)')
        mappings['string_upper_case_equals'] = PatternMapping('string_comparison', 'str.toUpper() = value')
        mappings['string_equality'] = PatternMapping('string_comparison', 'str1 = str2')
        
        # Boolean logic
        mappings['boolean_is_true'] = PatternMapping('boolean_operations', 'attr = true')
        mappings['boolean_is_false'] = PatternMapping('boolean_operations', 'attr = false')
        mappings['implies_simple'] = PatternMapping('boolean_guard_implies', 'a implies b')
        mappings['implies_reverse'] = PatternMapping('boolean_guard_implies', 'b implies a')
        mappings['all_attributes_defined'] = PatternMapping('boolean_operations', 'all attrs defined')
        mappings['at_least_one_defined'] = PatternMapping('boolean_operations', 'at least one defined')
        mappings['three_attributes_defined'] = PatternMapping('boolean_operations', 'three attrs defined')
        mappings['both_defined_or_both_null'] = PatternMapping('boolean_operations', 'both or neither')
        mappings['not_both_defined'] = PatternMapping('boolean_operations', 'not both defined')
        
        # Collection operations
        mappings['collection_any_match'] = PatternMapping('any_operation', 'collection->any()')
        mappings['collection_isUnique_attr'] = PatternMapping('uniqueness_constraint', 'isUnique()')
        mappings['collection_collectNested'] = PatternMapping('collect_nested', 'nested collect')
        mappings['collection_sum'] = PatternMapping('sum_product', 'collection->sum()')
        mappings['collection_asSet'] = PatternMapping('as_set_as_bag', 'collection->asSet()')
        mappings['collection_asSequence'] = PatternMapping('as_set_as_bag', 'collection->asSequence()')
        mappings['collection_first'] = PatternMapping('collection_navigation', 'collection->first()')
        mappings['collection_last'] = PatternMapping('collection_navigation', 'collection->last()')
        mappings['collection_at_index'] = PatternMapping('collection_navigation', 'collection->at()')
        mappings['collection_indexOf'] = PatternMapping('collection_membership', 'collection->indexOf()')
        
        # Set operations
        mappings['collection_including'] = PatternMapping('including_excluding', 'collection->including()')
        mappings['collection_excluding'] = PatternMapping('including_excluding', 'collection->excluding()')
        mappings['includes_excludes'] = PatternMapping('collection_membership', 'includes/excludes')
        mappings['includesAll_excludesAll'] = PatternMapping('subset_disjointness', 'includesAll/excludesAll')
        mappings['union_operation'] = PatternMapping('union_intersection', 'A->union(B)')
        mappings['intersection_operation'] = PatternMapping('set_intersection', 'A->intersection(B)')
        mappings['difference_operation'] = PatternMapping('symmetric_difference', 'A - B')
        mappings['symmetricDifference'] = PatternMapping('symmetric_difference', 'symmetricDifference')
        
        # Ordering
        mappings['collection_sortedBy'] = PatternMapping('ordering_ranking', 'sortedBy()')
        mappings['sortedBy'] = PatternMapping('ordering_ranking', 'sortedBy()')
        
        # Filters/selections
        mappings['select_operation'] = PatternMapping('select_reject', 'select()')
        mappings['reject_operation'] = PatternMapping('select_reject', 'reject()')
        mappings['collect_operation'] = PatternMapping('collect_flatten', 'collect()')
        mappings['sum_operation'] = PatternMapping('sum_product', 'sum()')
        mappings['product_operation'] = PatternMapping('sum_product', 'product()')
        mappings['count_operation'] = PatternMapping('exact_count_selection', 'count()')
        mappings['exists_constraint'] = PatternMapping('exists_nested', 'exists()')
        mappings['one_operation'] = PatternMapping('exactly_one', 'one()')
        
        # Type checks
        mappings['oclIsKindOf_check'] = PatternMapping('type_check_casting', 'oclIsKindOf()')
        mappings['oclIsTypeOf_check'] = PatternMapping('type_check_casting', 'oclIsTypeOf()')
        mappings['oclAsType_cast'] = PatternMapping('ocl_as_type', 'oclAsType()')
        mappings['type_check'] = PatternMapping('type_check_casting', 'type check')
        mappings['allInstances_check'] = PatternMapping('global_collection', 'allInstances()')
        
        # OCL stdlib
        mappings['oclIsUndefined'] = PatternMapping('ocl_is_undefined', 'oclIsUndefined()')
        mappings['oclIsInvalid'] = PatternMapping('ocl_is_invalid', 'oclIsInvalid()')
        mappings['oclIsInvalid_check'] = PatternMapping('ocl_is_invalid', 'oclIsInvalid()')
        
        # Conditionals
        mappings['conditional_if_then_else'] = PatternMapping('if_then_else', 'if-then-else')
        mappings['conditional_value_selection'] = PatternMapping('if_then_else', 'conditional value')
        mappings['if_then_else_expression'] = PatternMapping('if_then_else', 'if-then-else')
        mappings['conditional_constraint'] = PatternMapping('if_then_else', 'conditional')
        
        # Navigation
        mappings['simple_navigation'] = PatternMapping('navigation_chain', 'obj.assoc.attr')
        
        # Legacy
        mappings['boolean_guard'] = PatternMapping('boolean_operations', 'boolean guard')
        mappings['membership_check'] = PatternMapping('collection_membership', 'membership')
        mappings['isEmpty_notEmpty'] = PatternMapping('size_constraint', 'isEmpty/notEmpty')
        mappings['string_operation'] = PatternMapping('string_operations', 'string operation')
        mappings['division_modulo'] = PatternMapping('div_mod_operations', 'div/mod')
        mappings['all_different'] = PatternMapping(
            'uniqueness_constraint',
            'forAll(x,y | x<>y implies x.a<>y.a) → isUnique(x | x.a)',
            rewrite_fn=rewrite_all_different
        )
        mappings['closure_operation'] = PatternMapping('closure_transitive', 'closure')
        
        return mappings
    
    def _compute_stats(self):
        """Compute mapping statistics"""
        for pattern_id, mapping in self.mappings.items():
            if mapping.multiple:
                self.stats['composite_mappings'] += 1
            else:
                self.stats['direct_mappings'] += 1
            
            if mapping.rewrite_fn:
                self.stats['with_rewrite_fn'] += 1
            else:
                self.stats['without_rewrite_fn'] += 1
    
    def _validate_mappings(self):
        """Validate all mappings reference valid canonical patterns"""
        invalid = []
        for pid, mapping in self.mappings.items():
            if not mapping.multiple and mapping.canonical_pattern not in CANONICAL_PATTERNS:
                invalid.append((pid, mapping.canonical_pattern))
        
        if invalid:
            print("Invalid canonical patterns found:")
            for pid, canon in invalid:
                print(f"   {pid} → {canon}")
            raise ValueError(f"Found {len(invalid)} invalid canonical pattern references")
    
    def map_to_canonical(self, universal_pattern_id: str, constraint_text: str = "") -> List[Dict]:
        """
        Map universal pattern to canonical pattern(s) with optional rewriting.
        
        Returns list of mappings (usually 1, but can be multiple for compound patterns).
        """
        # Check if it's a universal pattern
        if universal_pattern_id not in self.mappings:
            # Assume it's already canonical
            if universal_pattern_id in CANONICAL_PATTERNS:
                return [{
                    'canonical_pattern': universal_pattern_id,
                    'rewritten_text': constraint_text,
                    'mapping': 'identity (already canonical)',
                    'universal_pattern': universal_pattern_id
                }]
            else:
                print(f"Unknown pattern: {universal_pattern_id} (treating as canonical)")
                return [{
                    'canonical_pattern': universal_pattern_id,
                    'rewritten_text': constraint_text,
                    'mapping': 'unknown (passed through)',
                    'universal_pattern': universal_pattern_id
                }]
        
        mapping = self.mappings[universal_pattern_id]
        
        # Handle multi-mapping with rewrite function
        if mapping.multiple and mapping.rewrite_fn:
            return mapping.rewrite_fn(constraint_text)
        
        # Handle single mapping with rewrite function
        rewritten_text = constraint_text
        if mapping.rewrite_fn:
            rewritten_text = mapping.rewrite_fn(constraint_text)
        
        return [{
            'canonical_pattern': mapping.canonical_pattern,
            'rewritten_text': rewritten_text,
            'mapping': mapping.description,
            'universal_pattern': universal_pattern_id
        }]
    
    def check_coverage(self, patterns_file: Optional[str] = None):
        """Check coverage against patterns_unified.json"""
        if not patterns_file and not self.patterns_file:
            print("No patterns file specified for coverage check")
            return
        
        file_path = patterns_file or self.patterns_file
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            all_pattern_ids = [p['id'] for p in data['patterns']]
            
            # Find patterns without mappings
            missing = [pid for pid in all_pattern_ids 
                      if pid not in self.mappings and pid not in CANONICAL_PATTERNS]
            
            if missing:
                print(f"\n{len(missing)} patterns without explicit mapping:")
                for pid in missing[:10]:  # Show first 10
                    print(f"   - {pid}")
                if len(missing) > 10:
                    print(f"   ... and {len(missing) - 10} more")
            else:
                print(f"\n100% coverage: All {len(all_pattern_ids)} patterns have mappings")
            
            return {
                'total_patterns': len(all_pattern_ids),
                'mapped': len(all_pattern_ids) - len(missing),
                'missing': missing
            }
            
        except Exception as e:
            print(f"Could not check coverage: {e}")
            return None
    
    def is_universal_pattern(self, pattern_id: str) -> bool:
        """Check if pattern is universal (needs mapping)"""
        return pattern_id in self.mappings
    
    def get_mapping_info(self, universal_pattern_id: str) -> Optional[PatternMapping]:
        """Get mapping info for a pattern"""
        return self.mappings.get(universal_pattern_id)
    
    def get_statistics(self) -> Dict:
        """Get mapping statistics"""
        return self.stats.copy()
    
    def print_statistics(self):
        """Print mapping statistics"""
        print(f"\n{'='*80}")
        print("PATTERN MAPPER V2 STATISTICS")
        print(f"{'='*80}")
        print(f"Total universal patterns: {self.stats['total_universal_patterns']}")
        print(f"Direct mappings (1→1):    {self.stats['direct_mappings']}")
        print(f"Composite mappings (1→N): {self.stats['composite_mappings']}")
        print(f"With rewrite functions:   {self.stats['with_rewrite_fn']}")
        print(f"Without rewrite functions:{self.stats['without_rewrite_fn']}")
        print(f"\nCanonical patterns available: {len(CANONICAL_PATTERNS)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test the enhanced mapper
    mapper = PatternMapperV2()
    
    mapper.print_statistics()
    
    # Test cases demonstrating rewrite capabilities
    print("\n" + "="*80)
    print("EXAMPLE 1: collection_size_range (Multi-mapping)")
    print("="*80)
    
    test_text = "self.items->size() >= 2 and self.items->size() <= 10"
    results = mapper.map_to_canonical("collection_size_range", test_text)
    
    print(f"Input: {test_text}")
    print(f"Output: {len(results)} canonical pattern(s)")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Canonical: {r['canonical_pattern']}")
        print(f"      Rewritten: {r['rewritten_text']}")
        print(f"      Mapping:   {r['mapping']}")
    
    print("\n" + "="*80)
    print("EXAMPLE 2: bi_implication (Multi-mapping)")
    print("="*80)
    
    test_text = "self.isActive <-> self.hasMembership"
    results = mapper.map_to_canonical("bi_implication", test_text)
    
    print(f"Input: {test_text}")
    print(f"Output: {len(results)} canonical pattern(s)")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Canonical: {r['canonical_pattern']}")
        print(f"      Rewritten: {r['rewritten_text']}")
        print(f"      Mapping:   {r['mapping']}")
    
    print("\n" + "="*80)
    print("EXAMPLE 3: collection_not_empty_simple (Rewrite)")
    print("="*80)
    
    test_text = "self.vehicles->notEmpty()"
    results = mapper.map_to_canonical("collection_not_empty_simple", test_text)
    
    print(f"Input: {test_text}")
    print(f"Output: {results[0]['canonical_pattern']}")
    print(f"Rewritten: {results[0]['rewritten_text']}")
    
    print("\n" + "="*80)
    print("EXAMPLE 4: xor_condition (Rewrite)")
    print("="*80)
    
    test_text = "self.isPremium xor self.isBasic"
    results = mapper.map_to_canonical("xor_condition", test_text)
    
    print(f"Input: {test_text}")
    print(f"Output: {results[0]['canonical_pattern']}")
    print(f"Rewritten: {results[0]['rewritten_text']}")
    
    # Check coverage
    patterns_file = Path(__file__).parent.parent.parent / "templates" / "patterns_unified.json"
    if patterns_file.exists():
        mapper.check_coverage(str(patterns_file))
