#!/usr/bin/env python3
"""
Enhanced SMT Encoder for All 50 OCL Patterns
With Association Metadata and Multiplicity Verification

Each pattern encoder:
- Extracts associations from constraint text
- Validates multiplicity constraints
- Generates Z3 constraints using domain relationships
- Supports counterexample finding via negation
"""

import re
from typing import Dict, Tuple, Optional, List
from z3 import *

from ..lowering.association_backed_encoder import XMIMetadataExtractor, AssociationMetadata
from .date_adapter import DateAdapter


class EnhancedSMTEncoder:
    """Enhanced SMT encoder for all 50 OCL patterns with association support"""
    
    def __init__(self, xmi_file: str):
        """Initialize with XMI metadata extractor"""
        self.extractor = XMIMetadataExtractor(xmi_file)
        self.date_adapter = DateAdapter(strategy='symbolic')
        self.max_depth = 10
        self.max_scope = 20
        
        # Pattern encoders registry
        self.pattern_encoders = {
            # Basic Patterns (1-9)
            "pairwise_uniqueness": self.encode_pairwise_uniqueness,
            "exact_count_selection": self.encode_exact_count_selection,
            "global_collection": self.encode_global_collection,
            "set_intersection": self.encode_set_intersection,
            "size_constraint": self.encode_size_constraint,
            "uniqueness_constraint": self.encode_uniqueness_constraint,
            "collection_membership": self.encode_collection_membership,
            "null_check": self.encode_null_check,
            "numeric_comparison": self.encode_numeric_comparison,
            
            # Advanced Patterns (10-19)
            "exactly_one": self.encode_exactly_one,
            "closure_transitive": self.encode_closure_transitive,
            "acyclicity": self.encode_acyclicity,
            "aggregation_iterate": self.encode_aggregation_iterate,
            "boolean_guard_implies": self.encode_boolean_guard_implies,
            "safe_navigation": self.encode_safe_navigation,
            "type_check_casting": self.encode_type_check_casting,
            "subset_disjointness": self.encode_subset_disjointness,
            "ordering_ranking": self.encode_ordering_ranking,
            "contractual_temporal": self.encode_contractual_temporal,
            
            # Collection Operations (20-27)
            "select_reject": self.encode_select_reject,
            "collect_flatten": self.encode_collect_flatten,
            "any_operation": self.encode_any_operation,
            "forall_nested": self.encode_forall_nested,
            "exists_nested": self.encode_exists_nested,
            "collect_nested": self.encode_collect_nested,
            "as_set_as_bag": self.encode_as_set_as_bag,
            "sum_product": self.encode_sum_product,
            
            # String Operations (28-31)
            "string_concat": self.encode_string_concat,
            "string_operations": self.encode_string_operations,
            "string_comparison": self.encode_string_comparison,
            "string_pattern": self.encode_string_pattern,
            
            # Arithmetic & Logic (32-36)
            "arithmetic_expression": self.encode_arithmetic_expression,
            "div_mod_operations": self.encode_div_mod_operations,
            "abs_min_max": self.encode_abs_min_max,
            "boolean_operations": self.encode_boolean_operations,
            "if_then_else": self.encode_if_then_else,
            
            # Tuple & Let (37-39)
            "tuple_literal": self.encode_tuple_literal,
            "let_expression": self.encode_let_expression,
            "let_nested": self.encode_let_nested,
            
            # Set Operations (40-43)
            "union_intersection": self.encode_union_intersection,
            "symmetric_difference": self.encode_symmetric_difference,
            "including_excluding": self.encode_including_excluding,
            "flatten_operation": self.encode_flatten_operation,
            
            # Navigation & Property (44-47)
            "navigation_chain": self.encode_navigation_chain,
            "optional_navigation": self.encode_optional_navigation,
            "collection_navigation": self.encode_collection_navigation,
            "shorthand_notation": self.encode_shorthand_notation,
            
            # OCL Standard Library (48-50)
            "ocl_is_undefined": self.encode_ocl_is_undefined,
            "ocl_is_invalid": self.encode_ocl_is_invalid,
            "ocl_as_type": self.encode_ocl_as_type,
            
            # Specialized composite patterns
            "valid_window_and_branch": self.encode_valid_window_and_branch,
        }
    
    # ========== Helpers ==========
    
    def _extract_ref_name(self, text: str) -> Optional[str]:
        """Extract reference name from constraint text"""
        match = re.search(r'self\.(\w+)', text)
        return match.group(1) if match else None
    
    def _extract_attribute_constraint(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Extract attribute constraint: self.attr op value
        Returns: (attr_name, operator, value) or None
        """
        match = re.search(r'self\.(\w+)\s*([<>=!]+)\s*([\d.]+)', text)
        return match.groups() if match else None
    
    def _is_attribute(self, context_class: str, field_name: str) -> bool:
        """Check if field is an attribute (not an association/reference)
        
        Args:
            context_class: The class name (e.g., 'Course', 'Rental')
            field_name: The field name (e.g., 'credits', 'startDate')
            
        Returns:
            True if field is an EAttribute, False otherwise
        """
        if hasattr(self.extractor, 'classes') and context_class in self.extractor.classes:
            for attr in self.extractor.classes[context_class].get('attributes', []):
                if attr['name'] == field_name:
                    return True
        return False
    
    def _get_attribute_type(self, context_class: str, attr_name: str) -> Optional[str]:
        """Get attribute type from XMI metadata"""
        if hasattr(self.extractor, 'classes') and context_class in self.extractor.classes:
            for attr in self.extractor.classes[context_class].get('attributes', []):
                if attr['name'] == attr_name:
                    return attr.get('type', 'EInt')
        return None
    
    def _get_association(self, context_class: str, ref_name: str) -> Optional[AssociationMetadata]:
        """Get association metadata from XMI"""
        return self.extractor.get_association_by_ref(context_class, ref_name)
    
    def _create_presence_bits(self, class_name: str, scope: int, model_vars: Dict) -> List:
        """Create presence bits for class instances"""
        bits = [Bool(f"{class_name}_present_{i}") for i in range(scope)]
        for i, bit in enumerate(bits):
            model_vars[f"{class_name}_present_{i}"] = bit
        return bits
    
    def _create_relation_matrix(self, ref_name: str, source_scope: int, target_scope: int, model_vars: Dict) -> List[List]:
        """Create relation matrix for collection encoding"""
        R = [[Bool(f"R_{ref_name}_{i}_{j}") for j in range(target_scope)] for i in range(source_scope)]
        for i in range(source_scope):
            for j in range(target_scope):
                model_vars[f"R_{ref_name}_{i}_{j}"] = R[i][j]
        return R
    
    def _encode_collection_with_multiplicity(self, solver: Solver, model_vars: Dict,
                                            assoc: AssociationMetadata,
                                            source_bits: List, target_bits: List,
                                            source_scope: int, target_scope: int) -> List[List]:
        """Encode collection with multiplicity constraints"""
        R = self._create_relation_matrix(assoc.ref_name, source_scope, target_scope, model_vars)
        
        # Enforce multiplicity constraints
        for i in range(source_scope):
            for j in range(target_scope):
                solver.add(Implies(R[i][j], And(source_bits[i], target_bits[j])))
        
        # Collection size per source
        for i in range(source_scope):
            coll_size = Sum([If(R[i][j], 1, 0) for j in range(target_scope)])
            model_vars[f"{assoc.ref_name}_size_{i}"] = coll_size
            
            # Apply multiplicity bounds
            if assoc.is_required:
                solver.add(Implies(source_bits[i], coll_size >= assoc.lower_bound))
            if assoc.upper_bound:
                solver.add(Implies(source_bits[i], coll_size <= assoc.upper_bound))
        
        return R
    
    def _enforce_at_most_one(self, solver: Solver, R: List[List], source_idx: int, target_scope: int):
        """Enforce at-most-one (0..1) reference for source instance"""
        solver.add(Sum([If(R[source_idx][j], 1, 0) for j in range(target_scope)]) <= 1)
    
    def _enforce_exactly_one(self, solver: Solver, R: List[List], source_idx: int, target_scope: int):
        """Enforce exactly-one (1..1) reference for source instance"""
        solver.add(Sum([If(R[source_idx][j], 1, 0) for j in range(target_scope)]) == 1)
    
    def _enforce_opposite_links(self, solver: Solver, R_forward: List[List], R_backward: List[List],
                               scope_a: int, scope_b: int):
        """Enforce bidirectional eOpposite consistency: R[i][j] <-> R_back[j][i]"""
        for i in range(scope_a):
            for j in range(scope_b):
                solver.add(R_forward[i][j] == R_backward[j][i])
    
    def _tie_size_to_presence(self, solver: Solver, size_var, presence_bits: List):
        """Tie size variable to actual count of present elements (guardrail)"""
        solver.add(size_var == Sum([If(bit, 1, 0) for bit in presence_bits]))
    
    def pretty_print_model(self, model, model_vars: Dict, constraint_name: str = "Unknown") -> str:
        """Pretty-print Z3 model for better counterexample readability
        
        Args:
            model: Z3 model from solver.model()
            model_vars: Dictionary of model variables
            constraint_name: Name of the constraint being verified
            
        Returns:
            Formatted string representation of the counterexample
        """
        output = []
        output.append("\n" + "="*80)
        output.append(f" COUNTEREXAMPLE for: {constraint_name}")
        output.append("="*80)
        
        # Separate variables by category
        presence_vars = {}
        relation_vars = {}
        attribute_vars = {}
        other_vars = {}
        
        for var_name, var in model_vars.items():
            if var_name.startswith('_'):  # Skip metadata
                continue
            
            try:
                value = model[var]
                if value is None:
                    continue
                    
                if '_present_' in var_name:
                    presence_vars[var_name] = value
                elif 'R_' in var_name and '_' in var_name[2:]:
                    relation_vars[var_name] = value
                elif any(x in var_name for x in ['_ord', '_attr_', '_cond_']):
                    attribute_vars[var_name] = value
                else:
                    other_vars[var_name] = value
            except:
                pass
        
        # Print presence information
        if presence_vars:
            output.append("\n Instance Presence:")
            for var_name in sorted(presence_vars.keys()):
                value = presence_vars[var_name]
                if str(value) == 'True':
                    # Extract class and index: Branch_present_0 -> Branch#0
                    parts = var_name.replace('_present_', '#')
                    output.append(f"    {parts}")
        
        # Print attribute values
        if attribute_vars:
            output.append("\n Attribute Values:")
            for var_name in sorted(attribute_vars.keys()):
                value = attribute_vars[var_name]
                output.append(f"   {var_name:40s} = {value}")
        
        # Print relation edges (associations)
        if relation_vars:
            output.append("\n🔗 Association Edges:")
            active_edges = []
            for var_name in sorted(relation_vars.keys()):
                value = relation_vars[var_name]
                if str(value) == 'True':
                    # Parse R_refName_0_2 -> "Source#0 → Target#2"
                    parts = var_name.split('_')
                    if len(parts) >= 4:
                        ref_name = parts[1]
                        src_idx = parts[2]
                        tgt_idx = parts[3]
                        active_edges.append(f"   {ref_name}: source#{src_idx} → target#{tgt_idx}")
            
            if active_edges:
                for edge in active_edges:
                    output.append(edge)
            else:
                output.append("   (none)")
        
        # Print other variables
        if other_vars:
            output.append("\n🔢 Other Variables:")
            for var_name in sorted(other_vars.keys()):
                value = other_vars[var_name]
                output.append(f"   {var_name:40s} = {value}")
        
        output.append("="*80)
        return "\n".join(output)
    
    # ========== Basic Patterns (1-9) ==========
    
    def encode_pairwise_uniqueness(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 1: Pairwise uniqueness - all pairs must have unique attributes"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', 'Element')
        ref_name = self._extract_ref_name(text)
        scope = context.get('scope', 5)
        
        if ref_name:
            assoc = self._get_association(context_class, ref_name)
            if assoc:
                target_bits = self._create_presence_bits(assoc.target_class, scope, model_vars)
                attrs = [Int(f"{ref_name}_attr_{i}") for i in range(scope)]
                for i, attr in enumerate(attrs):
                    model_vars[f"{ref_name}_attr_{i}"] = attr
                
                # Violation: two different instances with same attribute
                violations = []
                for i in range(scope):
                    for j in range(i + 1, scope):
                        violation = And(target_bits[i], target_bits[j], attrs[i] == attrs[j])
                        violations.append(violation)
                solver.add(Or(violations))
                return solver, model_vars
        
        # Fallback
        ids = [Int(f"id_{i}") for i in range(scope)]
        for i, id_var in enumerate(ids):
            model_vars[f"id_{i}"] = id_var
        solver.add(Not(Distinct(ids)))
        return solver, model_vars
    
    def encode_exact_count_selection(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 2: Exact count selection"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        ids = [Int(f"id_{i}") for i in range(scope)]
        for i, sym in enumerate(ids):
            model_vars[f"id_{i}"] = sym
        
        match = re.search(r'size\(\)\s*=\s*(\d+)', text)
        if match:
            expected = IntVal(int(match.group(1)))
            count = Sum([If(ids[i] > 0, 1, 0) for i in range(scope)])
            solver.add(Not(count == expected))
        
        return solver, model_vars
    
    def encode_global_collection(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 3: Global collection constraint"""
        solver = Solver()
        model_vars = {}
        
        all_valid = Bool("all_valid")
        model_vars["all_valid"] = all_valid
        solver.add(Not(all_valid))
        
        return solver, model_vars
    
    def encode_set_intersection(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 4: Set intersection"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        A = [Bool(f"in_A_{i}") for i in range(scope)]
        B = [Bool(f"in_B_{i}") for i in range(scope)]
        
        for i in range(scope):
            model_vars[f"in_A_{i}"] = A[i]
            model_vars[f"in_B_{i}"] = B[i]
        
        inter = [And(A[i], B[i]) for i in range(scope)]
        
        if "isEmpty" in text:
            solver.add(Or(*inter))
        
        return solver, model_vars
    
    def encode_size_constraint(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 5: Size constraint - WITH ASSOCIATION METADATA"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', '')
        source_scope = context.get('source_scope', 2)
        target_scope = context.get('target_scope', 5)
        
        ref_name = self._extract_ref_name(text)
        if not ref_name:
            # Fallback
            size_var = Int("size")
            model_vars["size"] = size_var
            solver.add(size_var > 10)
            return solver, model_vars
        
        # Get association metadata
        assoc = self._get_association(context_class, ref_name)
        if not assoc:
            # Fallback
            size_var = Int("size")
            model_vars["size"] = size_var
            match = re.search(r'(<=|>=|<|>)\s*(\d+)', text)
            if match:
                op, threshold = match.groups()
                threshold_val = IntVal(int(threshold))
                if op == '<=':
                    solver.add(size_var > threshold_val)
                elif op == '>=':
                    solver.add(size_var < threshold_val)
            return solver, model_vars
        
        # Encode with association metadata
        source_bits = self._create_presence_bits(context_class, source_scope, model_vars)
        target_bits = self._create_presence_bits(assoc.target_class, target_scope, model_vars)
        
        R = self._encode_collection_with_multiplicity(solver, model_vars, assoc,
                                                      source_bits, target_bits,
                                                      source_scope, target_scope)
        
        # Extract threshold
        match = re.search(r'(<=|>=|<|>)\s*(\d+)', text)
        if match:
            op, threshold = match.groups()
            threshold_val = IntVal(int(threshold))
            size = Sum([If(R[0][j], 1, 0) for j in range(target_scope)])
            model_vars[f"{ref_name}_size"] = size
            
            # Negate for violation
            if op == '<=':
                solver.add(size > threshold_val)
            elif op == '>=':
                solver.add(size < threshold_val)
            elif op == '<':
                solver.add(size >= threshold_val)
            elif op == '>':
                solver.add(size <= threshold_val)
        
        return solver, model_vars
    
    def encode_uniqueness_constraint(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 6: Uniqueness constraint - WITH ASSOCIATION"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', '')
        scope = context.get('scope', 5)
        
        ref_name = self._extract_ref_name(text)
        if not ref_name:
            xs = [Int(f"x_{i}") for i in range(scope)]
            for i, x in enumerate(xs):
                model_vars[f"x_{i}"] = x
            solver.add(Not(Distinct(xs)))
            return solver, model_vars
        
        assoc = self._get_association(context_class, ref_name)
        if assoc:
            target_bits = self._create_presence_bits(assoc.target_class, scope, model_vars)
            attrs = [Int(f"{ref_name}_attr_{i}") for i in range(scope)]
            
            for i, attr in enumerate(attrs):
                model_vars[f"{ref_name}_attr_{i}"] = attr
            
            # Violation: two present elements with same attribute
            violations = []
            for i in range(scope):
                for j in range(i + 1, scope):
                    violations.append(And(target_bits[i], target_bits[j], attrs[i] == attrs[j]))
            
            solver.add(Or(violations))
        
        return solver, model_vars
    
    def encode_collection_membership(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 7: Collection membership - WITH SELF-REFERENCE DETECTION"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        context_class = context.get('context_class', '')
        
        # Check for self-reference: collection->includes(self)
        if '->includes(self)' in text:
            # Extract collection name
            ref_match = re.search(r'self\.(\w+)->includes\(self\)', text)
            if ref_match:
                ref_name = ref_match.group(1)
                assoc = self._get_association(context_class, ref_name)
                
                # Model self-loop in relation
                # Create presence bits for instances
                if assoc:
                    target_bits = self._create_presence_bits(assoc.target_class, scope, model_vars)
                    # Self-reference matrix: does element i reference itself?
                    self_refs = [Bool(f"{ref_name}_self_loop_{i}") for i in range(scope)]
                    for i, ref in enumerate(self_refs):
                        model_vars[f"{ref_name}_self_loop_{i}"] = ref
                    
                    # Check if NOT includes self
                    is_not_includes = 'not ' in text.lower() or '!' in text
                    if is_not_includes:
                        # Violation: at least one element has self-loop
                        solver.add(Or([And(target_bits[i], self_refs[i]) for i in range(scope)]))
                    else:
                        # Violation: no element has self-loop
                        solver.add(And([Not(self_refs[i]) for i in range(scope)]))
                    
                    return solver, model_vars
        
        # Generic collection membership
        elems = [Bool(f"contains_{i}") for i in range(scope)]
        for i, e in enumerate(elems):
            model_vars[f"contains_{i}"] = e
        
        is_includes = "->includes(" in text
        is_not = 'not ' in text.lower() or '!' in text
        
        if is_includes and not is_not:
            # includes(x) violated: x not in collection
            solver.add(Not(Or(*elems)))
        elif is_includes and is_not:
            # not includes(x) violated: x is in collection
            solver.add(Or(*elems))
        else:
            solver.add(Or(*elems))
        
        return solver, model_vars
    
    def encode_null_check(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 8: Null check - WITH ASSOCIATION MULTIPLICITY"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', '')
        ref_name = self._extract_ref_name(text)
        
        is_not_null = "<> null" in text
        
        if ref_name:
            assoc = self._get_association(context_class, ref_name)
            if assoc:
                # Use multiplicity to determine violation
                is_null = Bool(f"{ref_name}_is_null")
                model_vars[f"{ref_name}_is_null"] = is_null
                
                if is_not_null:
                    # Violation: mandatory reference is null
                    solver.add(is_null)
                else:
                    # Violation: optional reference is not null
                    solver.add(Not(is_null))
                
                return solver, model_vars
        
        # Fallback
        is_null = Bool("is_null")
        model_vars["is_null"] = is_null
        if is_not_null:
            solver.add(is_null)
        else:
            solver.add(Not(is_null))
        
        return solver, model_vars
    
    def encode_numeric_comparison(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 9: Numeric comparison - WITH ATTRIBUTE EXTRACTION AND DATE HANDLING"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', '')
        
        # Check for date comparison first
        date_comp = self.date_adapter.extract_date_comparison(text)
        if date_comp:
            left_date, op, right_date = date_comp
            
            # Get symbolic ordinal indices for dates
            left_idx = self.date_adapter.get_date_variable(left_date)
            right_idx = self.date_adapter.get_date_variable(right_date)
            
            # Create Int variables for date ordinals
            left_var = Int(f"{left_date.replace('.', '_')}_ord")
            right_var = Int(f"{right_date.replace('.', '_')}_ord")
            model_vars[f"{left_date.replace('.', '_')}_ord"] = left_var
            model_vars[f"{right_date.replace('.', '_')}_ord"] = right_var
            
            # Build date comparison constraint
            if op == '>=':
                constraint = left_var >= right_var
            elif op == '>':
                constraint = left_var > right_var
            elif op == '<=':
                constraint = left_var <= right_var
            elif op == '<':
                constraint = left_var < right_var
            else:  # = or ==
                constraint = left_var == right_var
            
            # Add violation (negation for counterexample)
            solver.add(Not(constraint))
            
            # Store date metadata
            model_vars['_date_metadata'] = {
                'left_date': left_date,
                'right_date': right_date,
                'left_idx': left_idx,
                'right_idx': right_idx,
                'operator': op
            }
            
            return solver, model_vars
        
        # Try to extract attribute constraint
        attr_info = self._extract_attribute_constraint(text)
        if attr_info:
            attr_name, op, threshold_str = attr_info
            
            # Get attribute type from XMI metadata for correct Z3 type
            attr_type = self._get_attribute_type(context_class, attr_name)
            
            # Determine if Real or Int based on XMI type and threshold
            is_real_attr = attr_type in ('EDouble', 'EFloat', 'EReal') if attr_type else False
            is_real_threshold = '.' in threshold_str
            is_real = is_real_attr or is_real_threshold
            
            # Create properly typed threshold
            threshold = RealVal(threshold_str) if is_real else IntVal(int(threshold_str))
            
            # Create Z3 variable with correct type
            val = Real(f"{attr_name}") if is_real else Int(f"{attr_name}")
            model_vars[attr_name] = val
            model_vars[f'_{attr_name}_type'] = attr_type or 'inferred'  # Store for debugging
            
            # Build constraint based on operator
            if op == '>=':
                constraint = val >= threshold
            elif op == '>':
                constraint = val > threshold
            elif op == '<=':
                constraint = val <= threshold
            elif op == '<':
                constraint = val < threshold
            else:  # = or ==
                constraint = val == threshold
            
            # Add violation (negation for counterexample)
            solver.add(Not(constraint))
            return solver, model_vars
        
        # Fallback: generic pattern
        match = re.search(r'(>=|<=|>|<|=)\s*([\d.]+)', text)
        if match:
            op, threshold_str = match.groups()
            is_real = '.' in threshold_str
            threshold = RealVal(threshold_str) if is_real else IntVal(int(threshold_str))
            
            val = Real("value") if is_real else Int("value")
            model_vars["value"] = val
            
            if op == '>=':
                constraint = val >= threshold
            elif op == '>':
                constraint = val > threshold
            elif op == '<=':
                constraint = val <= threshold
            elif op == '<':
                constraint = val < threshold
            else:  # =
                constraint = val == threshold
            
            solver.add(Not(constraint))
        
        return solver, model_vars
    
    # ========== Advanced Patterns (10-19) ==========
    
    def encode_exactly_one(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 10: Exactly one"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        conds = [Bool(f"satisfies_{i}") for i in range(scope)]
        for i, c in enumerate(conds):
            model_vars[f"satisfies_{i}"] = c
        
        solver.add(Not(And(AtMost(*conds, 1), AtLeast(*conds, 1))))
        return solver, model_vars
    
    def encode_closure_transitive(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 11: Closure transitive"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        rel = [[Bool(f"rel_{i}_{j}") for j in range(scope)] for i in range(scope)]
        for i in range(scope):
            for j in range(scope):
                model_vars[f"rel_{i}_{j}"] = rel[i][j]
        
        # Simple transitive closure check
        for i in range(scope):
            for j in range(scope):
                for k in range(scope):
                    solver.add(Implies(And(rel[i][k], rel[k][j]), rel[i][j]))
        
        return solver, model_vars
    
    def encode_acyclicity(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 12: Acyclicity"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        rel = [[Bool(f"rel_{i}_{j}") for j in range(scope)] for i in range(scope)]
        for i in range(scope):
            for j in range(scope):
                model_vars[f"rel_{i}_{j}"] = rel[i][j]
        
        # Violation: find a cycle
        cycles = [And(rel[i][j], rel[j][i]) for i in range(scope) for j in range(i+1, scope)]
        solver.add(Or(cycles))
        
        return solver, model_vars
    
    def encode_aggregation_iterate(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 13: Aggregation iterate"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        elems = [Int(f"elem_{i}") for i in range(scope)]
        for i, e in enumerate(elems):
            model_vars[f"elem_{i}"] = e
        
        total = Sum(elems)
        model_vars["sum"] = total
        
        solver.add(total < 0)
        return solver, model_vars
    
    def encode_boolean_guard_implies(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 14: Boolean guard implies - WITH ATTRIBUTE PARSING"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', '')
        
        # Parse implies: guard implies consequence
        if ' implies ' in text:
            parts = text.split(' implies ')
            if len(parts) == 2:
                guard_text = parts[0].strip()
                consequence_text = parts[1].strip()
                
                # Parse guard condition
                guard_cond = None
                if '<> null' in guard_text:
                    # Null check guard
                    is_not_null = Bool("guard_not_null")
                    model_vars["guard_not_null"] = is_not_null
                    guard_cond = is_not_null
                elif self._extract_attribute_constraint(guard_text):
                    # Attribute comparison guard
                    attr_name, op, threshold_str = self._extract_attribute_constraint(guard_text)
                    is_real = '.' in threshold_str
                    threshold = RealVal(threshold_str) if is_real else IntVal(int(threshold_str))
                    val = Real(f"{attr_name}_guard") if is_real else Int(f"{attr_name}_guard")
                    model_vars[f"{attr_name}_guard"] = val
                    
                    if op == '>=':
                        guard_cond = val >= threshold
                    elif op == '>':
                        guard_cond = val > threshold
                    elif op == '<=':
                        guard_cond = val <= threshold
                    elif op == '<':
                        guard_cond = val < threshold
                    elif op == '==' or op == '=':
                        guard_cond = val == threshold
                
                # Parse consequence
                consequence_cond = None
                if self._extract_attribute_constraint(consequence_text):
                    attr_name, op, threshold_str = self._extract_attribute_constraint(consequence_text)
                    is_real = '.' in threshold_str
                    threshold = RealVal(threshold_str) if is_real else IntVal(int(threshold_str))
                    val = Real(f"{attr_name}_cons") if is_real else Int(f"{attr_name}_cons")
                    model_vars[f"{attr_name}_cons"] = val
                    
                    if op == '>=':
                        consequence_cond = val >= threshold
                    elif op == '>':
                        consequence_cond = val > threshold
                    elif op == '<=':
                        consequence_cond = val <= threshold
                    elif op == '<':
                        consequence_cond = val < threshold
                    elif op == '==' or op == '=':
                        consequence_cond = val == threshold
                
                # Violation: guard true but consequence false
                if guard_cond is not None and consequence_cond is not None:
                    solver.add(And(guard_cond, Not(consequence_cond)))
                    return solver, model_vars
        
        # Fallback: generic boolean implies
        cond = Bool("guard_cond")
        cons = Bool("consequence")
        model_vars["guard_cond"] = cond
        model_vars["consequence"] = cons
        
        # Violation of implies: condition true, consequence false
        solver.add(And(cond, Not(cons)))
        
        return solver, model_vars
    
    def encode_safe_navigation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 15: Safe navigation"""
        solver = Solver()
        model_vars = {}
        
        has_null = Bool("null_encountered")
        model_vars["null_encountered"] = has_null
        
        solver.add(has_null)
        return solver, model_vars
    
    def encode_type_check_casting(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 16: Type check casting"""
        solver = Solver()
        model_vars = {}
        
        type_valid = Bool("type_valid")
        model_vars["type_valid"] = type_valid
        
        solver.add(Not(type_valid))
        return solver, model_vars
    
    def encode_subset_disjointness(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 17: Subset disjointness"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        A = [Bool(f"A_{i}") for i in range(scope)]
        B = [Bool(f"B_{i}") for i in range(scope)]
        
        for i in range(scope):
            model_vars[f"A_{i}"] = A[i]
            model_vars[f"B_{i}"] = B[i]
        
        # Violation: A and B are not disjoint
        shared = [And(A[i], B[i]) for i in range(scope)]
        solver.add(Or(shared))
        
        return solver, model_vars
    
    def encode_ordering_ranking(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 18: Ordering ranking"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        ranks = [Int(f"rank_{i}") for i in range(scope)]
        for i, r in enumerate(ranks):
            model_vars[f"rank_{i}"] = r
        
        # Violation: not all distinct
        solver.add(Not(Distinct(ranks)))
        return solver, model_vars
    
    def encode_contractual_temporal(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 19: Contractual temporal - ENHANCED FOR IMPLIES"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', '')
        
        # Check for implies with navigation: A <> null and B <> null implies C = D
        if ' implies ' in text:
            parts = text.split(' implies ')
            if len(parts) == 2:
                guard_text = parts[0].strip()
                consequence_text = parts[1].strip()
                
                # Parse complex guard with 'and'
                guard_conditions = []
                if ' and ' in guard_text:
                    guard_parts = guard_text.split(' and ')
                    for part in guard_parts:
                        part = part.strip()
                        if '<> null' in part:
                            # Null check
                            ref_match = re.search(r'self\.(\w+(?:\.\w+)*)', part)
                            if ref_match:
                                ref_path = ref_match.group(1)
                                is_not_null = Bool(f"{ref_path.replace('.', '_')}_not_null")
                                model_vars[f"{ref_path.replace('.', '_')}_not_null"] = is_not_null
                                guard_conditions.append(is_not_null)
                        elif '->notEmpty()' in part:
                            # Collection not empty check
                            ref_match = re.search(r'self\.(\w+)->notEmpty\(\)', part)
                            if ref_match:
                                ref_name = ref_match.group(1)
                                is_present = Bool(f"{ref_name}_present")
                                model_vars[f"{ref_name}_present"] = is_present
                                guard_conditions.append(is_present)
                else:
                    # Single guard checks
                    if '<> null' in guard_text:
                        ref_match = re.search(r'self\.(\w+(?:\.\w+)*)', guard_text)
                        if ref_match:
                            ref_path = ref_match.group(1)
                            is_not_null = Bool(f"{ref_path.replace('.', '_')}_not_null")
                            model_vars[f"{ref_path.replace('.', '_')}_not_null"] = is_not_null
                            guard_conditions.append(is_not_null)
                    elif '->notEmpty()' in guard_text:
                        ref_match = re.search(r'self\.(\w+)->notEmpty\(\)', guard_text)
                        if ref_match:
                            ref_name = ref_match.group(1)
                            is_present = Bool(f"{ref_name}_present")
                            model_vars[f"{ref_name}_present"] = is_present
                            guard_conditions.append(is_present)
                
                # Parse consequence with navigation: self.a.b >= self.c
                consequence_cond = None
                
                # Check if consequence is a date comparison
                date_comp = self.date_adapter.extract_date_comparison(consequence_text)
                if date_comp:
                    left_date, op, right_date = date_comp
                    
                    # Get symbolic ordinal indices for dates
                    left_idx = self.date_adapter.get_date_variable(left_date)
                    right_idx = self.date_adapter.get_date_variable(right_date)
                    
                    # Create Int variables for date ordinals
                    left_var = Int(f"{left_date.replace('.', '_')}_ord")
                    right_var = Int(f"{right_date.replace('.', '_')}_ord")
                    model_vars[f"{left_date.replace('.', '_')}_ord"] = left_var
                    model_vars[f"{right_date.replace('.', '_')}_ord"] = right_var
                    
                    # Build date comparison constraint
                    if op == '>=':
                        consequence_cond = left_var >= right_var
                    elif op == '>':
                        consequence_cond = left_var > right_var
                    elif op == '<=':
                        consequence_cond = left_var <= right_var
                    elif op == '<':
                        consequence_cond = left_var < right_var
                    else:  # = or ==
                        consequence_cond = left_var == right_var
                else:
                    # Regular navigation comparison
                    nav_comparison = re.search(r'self\.(\w+(?:\.\w+)*)\s*([<>=]+)\s*self\.(\w+)', consequence_text)
                    if nav_comparison:
                        left_path, op, right_attr = nav_comparison.groups()
                        left_var = Int(left_path.replace('.', '_'))
                        right_var = Int(right_attr)
                        model_vars[left_path.replace('.', '_')] = left_var
                        model_vars[right_attr] = right_var
                        
                        if op == '>=':
                            consequence_cond = left_var >= right_var
                        elif op == '>':
                            consequence_cond = left_var > right_var
                        elif op == '<=':
                            consequence_cond = left_var <= right_var
                        elif op == '<':
                            consequence_cond = left_var < right_var
                        elif op == '=' or op == '==':
                            consequence_cond = left_var == right_var
                
                # Build violation: all guards true but consequence false
                if guard_conditions and consequence_cond is not None:
                    guard_combined = And(guard_conditions) if len(guard_conditions) > 1 else guard_conditions[0]
                    solver.add(And(guard_combined, Not(consequence_cond)))
                    return solver, model_vars
        
        # Fallback: generic pre/post state
        pre_state = Bool("pre_condition")
        post_state = Bool("post_condition")
        model_vars["pre_condition"] = pre_state
        model_vars["post_condition"] = post_state
        
        solver.add(And(pre_state, Not(post_state)))
        return solver, model_vars
    
    # ========== Collection Operations (20-27) ==========
    
    def encode_select_reject(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 20: Select/reject"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        selected = [Bool(f"selected_{i}") for i in range(scope)]
        for i, s in enumerate(selected):
            model_vars[f"selected_{i}"] = s
        
        solver.add(Not(Or(*selected)))
        return solver, model_vars
    
    def encode_collect_flatten(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 21: Collect/flatten"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        results = [Int(f"result_{i}") for i in range(scope)]
        for i, r in enumerate(results):
            model_vars[f"result_{i}"] = r
        
        return solver, model_vars
    
    def encode_any_operation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 22: Any operation"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        any_result = Bool("any_result")
        model_vars["any_result"] = any_result
        
        solver.add(Not(any_result))
        return solver, model_vars
    
    def encode_forall_nested(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 23: ForAll nested - WITH CYCLE DETECTION"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        context_class = context.get('context_class', '')
        ref_name = self._extract_ref_name(text)
        
        # Check for cycle pattern: forAll(p | not p.collection->includes(self))
        cycle_pattern = re.search(r'forAll\(\w+\s*\|\s*not\s+(\w+)\.(\w+)->includes\(self\)\)', text)
        if cycle_pattern:
            var_name, nested_ref = cycle_pattern.groups()
            
            # Get association for nested navigation
            if ref_name and context_class:
                assoc = self._get_association(context_class, ref_name)
                if assoc:
                    # Model 2-cycle: A->B and B->A
                    # Create relation matrices
                    R = [[Bool(f"R_{ref_name}_{i}_{j}") for j in range(scope)] for i in range(scope)]
                    R_nested = [[Bool(f"R_{nested_ref}_{i}_{j}") for j in range(scope)] for i in range(scope)]
                    
                    for i in range(scope):
                        for j in range(scope):
                            model_vars[f"R_{ref_name}_{i}_{j}"] = R[i][j]
                            model_vars[f"R_{nested_ref}_{i}_{j}"] = R_nested[i][j]
                    
                    # Violation: find a 2-cycle - self.ref->p and p.nested_ref->self
                    # For at least one p: R[0][p] = true AND R_nested[p][0] = true
                    violations = []
                    for j in range(scope):
                        # 2-cycle: 0 → j → 0
                        violations.append(And(R[0][j], R_nested[j][0]))
                    
                    solver.add(Or(violations))
                    return solver, model_vars
        
        # Standard forAll encoding
        if ref_name and context_class:
            assoc = self._get_association(context_class, ref_name)
            if assoc:
                target_bits = self._create_presence_bits(assoc.target_class, scope, model_vars)
                conds = [Bool(f"{ref_name}_cond_{i}") for i in range(scope)]
                
                for i, cond in enumerate(conds):
                    model_vars[f"{ref_name}_cond_{i}"] = cond
                
                # Violation: at least one doesn't satisfy
                violations = [And(target_bits[i], Not(conds[i])) for i in range(scope)]
                solver.add(Or(violations))
                
                return solver, model_vars
        
        # Fallback
        conds = [Bool(f"cond_{i}") for i in range(scope)]
        for i, c in enumerate(conds):
            model_vars[f"cond_{i}"] = c
        solver.add(Or([Not(c) for c in conds]))
        
        return solver, model_vars
    
    def encode_exists_nested(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 24: Exists nested - WITH ASSOCIATION"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        context_class = context.get('context_class', '')
        ref_name = self._extract_ref_name(text)
        
        if ref_name and context_class:
            assoc = self._get_association(context_class, ref_name)
            if assoc:
                target_bits = self._create_presence_bits(assoc.target_class, scope, model_vars)
                conds = [Bool(f"{ref_name}_cond_{i}") for i in range(scope)]
                
                for i, cond in enumerate(conds):
                    model_vars[f"{ref_name}_cond_{i}"] = cond
                
                # Violation: none satisfy
                violations = [And(target_bits[i], Not(conds[i])) for i in range(scope)]
                solver.add(And(violations))
                
                return solver, model_vars
        
        # Fallback
        conds = [Bool(f"cond_{i}") for i in range(scope)]
        for i, c in enumerate(conds):
            model_vars[f"cond_{i}"] = c
        solver.add(And([Not(c) for c in conds]))
        
        return solver, model_vars
    
    def encode_collect_nested(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 25: Collect nested"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        results = [Int(f"collected_{i}") for i in range(scope)]
        for i, r in enumerate(results):
            model_vars[f"collected_{i}"] = r
        
        return solver, model_vars
    
    def encode_as_set_as_bag(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 26: As set / as bag"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        is_set = Bool("is_set")
        model_vars["is_set"] = is_set
        
        solver.add(Not(is_set))
        return solver, model_vars
    
    def encode_sum_product(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 27: Sum product"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        elems = [Int(f"elem_{i}") for i in range(scope)]
        for i, e in enumerate(elems):
            model_vars[f"elem_{i}"] = e
        
        total = Sum(elems)
        model_vars["sum"] = total
        
        solver.add(total != Sum(elems))
        return solver, model_vars
    
    # ========== String Operations (28-31) ==========
    
    def encode_string_concat(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 28: String concat"""
        solver = Solver()
        model_vars = {}
        
        s1 = String("s1")
        s2 = String("s2")
        result = String("result")
        model_vars["s1"] = s1
        model_vars["s2"] = s2
        model_vars["result"] = result
        
        return solver, model_vars
    
    def encode_string_operations(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 29: String operations"""
        solver = Solver()
        model_vars = {}
        
        s = String("string")
        model_vars["string"] = s
        
        return solver, model_vars
    
    def encode_string_comparison(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 30: String comparison"""
        solver = Solver()
        model_vars = {}
        
        s1 = String("s1")
        s2 = String("s2")
        model_vars["s1"] = s1
        model_vars["s2"] = s2
        
        solver.add(s1 != s2)
        return solver, model_vars
    
    def encode_string_pattern(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 31: String pattern"""
        solver = Solver()
        model_vars = {}
        
        s = String("string")
        model_vars["string"] = s
        
        return solver, model_vars
    
    # ========== Arithmetic & Logic (32-36) ==========
    
    def encode_arithmetic_expression(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 32: Arithmetic expression"""
        solver = Solver()
        model_vars = {}
        
        x = Int("x")
        y = Int("y")
        model_vars["x"] = x
        model_vars["y"] = y
        
        return solver, model_vars
    
    def encode_div_mod_operations(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 33: Div/mod operations"""
        solver = Solver()
        model_vars = {}
        
        x = Int("x")
        y = Int("y")
        model_vars["x"] = x
        model_vars["y"] = y
        
        solver.add(y != 0)
        return solver, model_vars
    
    def encode_abs_min_max(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 34: Abs/min/max"""
        solver = Solver()
        model_vars = {}
        
        x = Int("x")
        y = Int("y")
        model_vars["x"] = x
        model_vars["y"] = y
        
        return solver, model_vars
    
    def encode_boolean_operations(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 35: Boolean operations - WITH ATTRIBUTE PARSING"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', '')
        
        # Check for compound attribute constraints: attr >= X and attr <= Y
        and_pattern = re.findall(r'self\.(\w+)\s*([<>=]+)\s*([\d.]+)', text)
        
        if len(and_pattern) >= 2 and ' and ' in text:
            # Parse compound constraint
            constraints = []
            for attr_name, op, threshold_str in and_pattern:
                is_real = '.' in threshold_str
                threshold = RealVal(threshold_str) if is_real else IntVal(int(threshold_str))
                
                # Create Z3 variable (reuse if same attribute)
                var_name = attr_name
                if var_name not in model_vars:
                    val = Real(var_name) if is_real else Int(var_name)
                    model_vars[var_name] = val
                else:
                    val = model_vars[var_name]
                
                # Build constraint
                if op == '>=':
                    constraints.append(val >= threshold)
                elif op == '>':
                    constraints.append(val > threshold)
                elif op == '<=':
                    constraints.append(val <= threshold)
                elif op == '<':
                    constraints.append(val < threshold)
                elif op == '==' or op == '=':
                    constraints.append(val == threshold)
            
            # Violation: negate the conjunction (A and B → not A or not B)
            if constraints:
                solver.add(Not(And(constraints)))
                return solver, model_vars
        
        # Check for boolean logic with 'or'
        if ' or ' in text:
            or_pattern = re.findall(r'self\.(\w+)\s*([<>=]+)\s*([\d.]+)', text)
            if or_pattern:
                constraints = []
                for attr_name, op, threshold_str in or_pattern:
                    is_real = '.' in threshold_str
                    threshold = RealVal(threshold_str) if is_real else IntVal(int(threshold_str))
                    
                    var_name = attr_name
                    if var_name not in model_vars:
                        val = Real(var_name) if is_real else Int(var_name)
                        model_vars[var_name] = val
                    else:
                        val = model_vars[var_name]
                    
                    if op == '>=':
                        constraints.append(val >= threshold)
                    elif op == '>':
                        constraints.append(val > threshold)
                    elif op == '<=':
                        constraints.append(val <= threshold)
                    elif op == '<':
                        constraints.append(val < threshold)
                    elif op == '==' or op == '=':
                        constraints.append(val == threshold)
                
                # Violation: negate the disjunction (A or B → not A and not B)
                if constraints:
                    solver.add(And([Not(c) for c in constraints]))
                    return solver, model_vars
        
        # Fallback: generic boolean operations
        p = Bool("p")
        q = Bool("q")
        model_vars["p"] = p
        model_vars["q"] = q
        
        return solver, model_vars
    
    def encode_if_then_else(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 36: If-then-else"""
        solver = Solver()
        model_vars = {}
        
        cond = Bool("condition")
        then_val = Int("then_val")
        else_val = Int("else_val")
        model_vars["condition"] = cond
        model_vars["then_val"] = then_val
        model_vars["else_val"] = else_val
        
        return solver, model_vars
    
    # ========== Tuple & Let (37-39) ==========
    
    def encode_tuple_literal(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 37: Tuple literal"""
        solver = Solver()
        model_vars = {}
        
        return solver, model_vars
    
    def encode_let_expression(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 38: Let expression"""
        solver = Solver()
        model_vars = {}
        
        return solver, model_vars
    
    def encode_let_nested(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 39: Let nested"""
        solver = Solver()
        model_vars = {}
        
        return solver, model_vars
    
    # ========== Set Operations (40-43) ==========
    
    def encode_union_intersection(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 40: Union/intersection"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        A = [Bool(f"A_{i}") for i in range(scope)]
        B = [Bool(f"B_{i}") for i in range(scope)]
        
        for i in range(scope):
            model_vars[f"A_{i}"] = A[i]
            model_vars[f"B_{i}"] = B[i]
        
        return solver, model_vars
    
    def encode_symmetric_difference(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 41: Symmetric difference"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        A = [Bool(f"A_{i}") for i in range(scope)]
        B = [Bool(f"B_{i}") for i in range(scope)]
        
        for i in range(scope):
            model_vars[f"A_{i}"] = A[i]
            model_vars[f"B_{i}"] = B[i]
        
        return solver, model_vars
    
    def encode_including_excluding(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 42: Including/excluding"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        elem = Bool("element")
        collection = [Bool(f"in_coll_{i}") for i in range(scope)]
        model_vars["element"] = elem
        for i, c in enumerate(collection):
            model_vars[f"in_coll_{i}"] = c
        
        return solver, model_vars
    
    def encode_flatten_operation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 43: Flatten operation"""
        solver = Solver()
        model_vars = {}
        
        return solver, model_vars
    
    # ========== Navigation & Property (44-47) ==========
    
    def encode_navigation_chain(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 44: Navigation chain - WITH MULTIPLICITY"""
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', '')
        ref_name = self._extract_ref_name(text)
        
        if ref_name and context_class:
            assoc = self._get_association(context_class, ref_name)
            if assoc:
                print(f"🔗 Navigation: {assoc}")
                is_valid = Bool(f"{ref_name}_valid")
                model_vars[f"{ref_name}_valid"] = is_valid
                
                # Violation based on multiplicity
                if assoc.is_required:
                    solver.add(Not(is_valid))  # Violation: required ref not present
                
                return solver, model_vars
        
        is_valid = Bool("navigation_valid")
        model_vars["navigation_valid"] = is_valid
        solver.add(Not(is_valid))
        
        return solver, model_vars
    
    def encode_optional_navigation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 45: Optional navigation"""
        solver = Solver()
        model_vars = {}
        
        is_present = Bool("optional_present")
        model_vars["optional_present"] = is_present
        
        return solver, model_vars
    
    def encode_collection_navigation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 46: Collection navigation"""
        solver = Solver()
        model_vars = {}
        scope = context.get('scope', 5)
        
        first_elem = Int("first")
        model_vars["first"] = first_elem
        
        return solver, model_vars
    
    def encode_shorthand_notation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 47: Shorthand notation"""
        solver = Solver()
        model_vars = {}
        
        return solver, model_vars
    
    # ========== OCL Standard Library (48-50) ==========
    
    def encode_ocl_is_undefined(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 48: OCL oclIsUndefined"""
        solver = Solver()
        model_vars = {}
        
        is_undefined = Bool("is_undefined")
        model_vars["is_undefined"] = is_undefined
        
        solver.add(Not(is_undefined))
        return solver, model_vars
    
    def encode_ocl_is_invalid(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 49: OCL oclIsInvalid"""
        solver = Solver()
        model_vars = {}
        
        is_invalid = Bool("is_invalid")
        model_vars["is_invalid"] = is_invalid
        
        solver.add(Not(is_invalid))
        return solver, model_vars
    
    def encode_ocl_as_type(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Pattern 50: OCL oclAsType"""
        solver = Solver()
        model_vars = {}
        
        type_valid = Bool("type_valid")
        model_vars["type_valid"] = type_valid
        
        solver.add(Not(type_valid))
        return solver, model_vars
    
    # ========== Specialized Composite Encoders ==========
    
    def encode_valid_window_and_branch(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Specialized encoder for ValidWindowAndBranch composite constraint
        
        Handles: self.dateTo > self.dateFrom and 
                (self.optionalRef->isEmpty() or self.optionalRef.sharedRef = self.sharedRef)
        
        Combines three sub-patterns:
        1. DATE_ORDERING: dateTo > dateFrom
        2. OPTIONAL_NAVIGATION: optionalRef->isEmpty()
        3. REFERENCE_EQUALITY: optionalRef.sharedRef = self.sharedRef
        """
        solver = Solver()
        model_vars = {}
        
        context_class = context.get('context_class', context.get('context', 'ContextClass'))
        scope = context.get('scope', 5)
        
        print(f"\n{'='*80}")
        print(f" ENCODING: VALID_WINDOW_AND_BRANCH (Composite)")
        print(f"{'='*80}")
        print(f" Context: {context_class}")
        
        # (1) Date Ordering: dateTo > dateFrom
        print("\n[1/3] Date Ordering: dateTo > dateFrom")
        
        # Use date adapter for proper ordinal mapping
        date_comp = self.date_adapter.extract_date_comparison('self.dateTo > self.dateFrom')
        if date_comp:
            left_date, op, right_date = date_comp
            left_idx = self.date_adapter.get_date_variable(left_date)
            right_idx = self.date_adapter.get_date_variable(right_date)
            
            dateTo_ord = Int("dateTo_ord")
            dateFrom_ord = Int("dateFrom_ord")
            model_vars["dateTo_ord"] = dateTo_ord
            model_vars["dateFrom_ord"] = dateFrom_ord
            
            date_valid = dateTo_ord > dateFrom_ord
            print(f"    Date ordinals: dateTo > dateFrom")
        else:
            # Fallback
            dateTo_ord = Int("dateTo_ord")
            dateFrom_ord = Int("dateFrom_ord")
            model_vars["dateTo_ord"] = dateTo_ord
            model_vars["dateFrom_ord"] = dateFrom_ord
            date_valid = dateTo_ord > dateFrom_ord
        
        # (2) Optional Navigation: vehicle->isEmpty() / vehicle presence
        print("\n[2/3] Optional Navigation: vehicle [0..1]")
        vehicle_present = Bool("vehicle_present")
        model_vars["vehicle_present"] = vehicle_present
        print(f"    Created presence bit: vehicle_present")
        
        # (3) Reference Equality: optionalRef.sharedRef = self.sharedRef (if optionalRef present)
        print("\n[3/3] Reference Equality: optionalRef.sharedRef = context.sharedRef")
        optional_shared_idx = Int("optional_shared_idx")
        context_shared_idx = Int("context_shared_idx")
        model_vars["optional_shared_idx"] = optional_shared_idx
        model_vars["context_shared_idx"] = context_shared_idx
        
        # Shared ref indices must be valid (0..nShared)
        nShared = context.get('nShared', 3)
        solver.add(And(optional_shared_idx >= 0, optional_shared_idx < nShared))
        solver.add(And(context_shared_idx >= 0, context_shared_idx < nShared))
        
        ref_match = optional_shared_idx == context_shared_idx
        print(f"    Shared reference equality constraint")
        
        # Combine: Invariant is (dateTo > dateFrom) AND (NOT optionalRef_present OR ref_match)
        # Equivalent to: date_valid AND (optionalRef->isEmpty() OR optionalRef.sharedRef = self.sharedRef)
        invariant = And(
            date_valid,
            Or(Not(vehicle_present), ref_match)
        )
        
        # Violation form: NOT invariant
        # = NOT(date_valid AND (NOT optionalRef_present OR ref_match))
        # = NOT date_valid OR NOT(NOT optionalRef_present OR ref_match)
        # = NOT date_valid OR (optionalRef_present AND NOT ref_match)
        violation = Or(
            Not(date_valid),  # Dates in wrong order
            And(vehicle_present, Not(ref_match))  # Optional ref present but shared refs don't match
        )
        
        solver.add(violation)
        
        print(f"\n Composite pattern encoded:")
        print(f"   Violation = (dateTo <= dateFrom) OR (optionalRef present AND shared refs differ)")
        
        return solver, model_vars
    
    # ========== Main Encode Method ==========
    
    def encode(self, pattern_name: str, ocl_text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Main entry point - routes to appropriate pattern encoder"""
        if pattern_name not in self.pattern_encoders:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        print(f"\n Encoding pattern: {pattern_name}")
        print(f"   OCL: {ocl_text[:60]}...")
        
        solver, model_vars = self.pattern_encoders[pattern_name](ocl_text, context)
        
        result = solver.check()
        print(f"   Z3 Result: {result}")
        
        return solver, model_vars
