"""
Solver-based Implication Checking for OCL Constraints

Uses Z3 solver to detect logical implications between constraints:
- C1 implies C2: If C2 is always true when C1 is true
- C1 equivalent to C2: If C1 ⟺ C2
- C1 contradicts C2: If C1 ∧ C2 is unsatisfiable

This provides the strongest form of redundancy detection.
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from modules.core.models import OCLConstraint


class ImplicationRelation(Enum):
    """Relationship between two constraints"""
    INDEPENDENT = "independent"          # No logical relationship
    C1_IMPLIES_C2 = "c1_implies_c2"     # C1 → C2 (C2 is redundant if C1 is present)
    C2_IMPLIES_C1 = "c2_implies_c1"     # C2 → C1 (C1 is redundant if C2 is present)
    EQUIVALENT = "equivalent"            # C1 ⟺ C2 (one is redundant)
    CONTRADICTORY = "contradictory"      # C1 ∧ C2 is unsatisfiable


def _split_ocl(ocl: str, default_context: str) -> Tuple[str, str]:
    if 'inv:' in ocl:
        header, body = ocl.split('inv:', 1)
        header = header.strip()
        body = body.strip()
        if not header.lower().startswith('context'):
            header = f"context {default_context}"
        return header, body
    return f"context {default_context}", ocl.strip()


def _negated_constraint(constraint: OCLConstraint) -> OCLConstraint:
    header, body = _split_ocl(constraint.ocl, constraint.context)
    negated_body = _build_negated_body(constraint, body)
    if negated_body is None:
        return None
    neg_ocl = f"{header} inv: {negated_body}"
    base_pattern = constraint.pattern_id.replace('_unsat', '')
    return OCLConstraint(
        ocl=neg_ocl,
        pattern_id=base_pattern,
        pattern_name=f"Negation({constraint.pattern_name})",
        context=constraint.context,
        parameters=dict(constraint.parameters),
        confidence=1.0,
        metadata={**constraint.metadata, 'is_negated': True}
    )


def _invert_operator(op: str) -> Optional[str]:
    mapping = {
        '>': '<=',
        '>=': '<',
        '<': '>=',
        '<=': '>',
        '<>': '=',
        '!=': '=',
        '=': None,
        '==': None,
    }
    return mapping.get(op)


def _build_negated_body(constraint: OCLConstraint, body: str) -> Optional[str]:
    """
    Build a negated body without introducing unsupported patterns.
    Returns None if negation cannot be expressed safely.
    """
    pid = constraint.pattern_id.replace('_unsat', '')

    # 1) Null checks / association exists
    if pid in {
        'attribute_not_null_simple',
        'association_exists',
        'attribute_null_check',
        'attribute_defined',
        'self_not_null'
    }:
        match = re.search(r'(self\.\w+)\s*(<>|=)\s*null', body)
        if match:
            left, op = match.group(1), match.group(2)
            if op == '<>':
                return f"{left} = null"
            if op == '=':
                return f"{left} <> null"
        # If we can't parse, fall back to unsupported
        return None

    # 2) Boolean checks
    if pid in {'boolean_is_true', 'boolean_is_false'}:
        match = re.search(r'(self\.\w+)\s*=\s*(true|false)', body)
        if match:
            left, val = match.group(1), match.group(2)
            neg_val = 'false' if val == 'true' else 'true'
            return f"{left} = {neg_val}"
        return None

    # 3) Collection size / emptiness
    if pid in {
        'collection_min_size', 'collection_max_size', 'collection_has_size',
        'collection_not_empty_simple', 'collection_not_empty_check', 'collection_empty_check',
        'size_constraint', 'collection_size_range'
    }:
        # notEmpty()/isEmpty()
        match = re.search(r'(self\.\w+)->(notEmpty|isEmpty)\(\)', body)
        if match:
            coll, op = match.group(1), match.group(2)
            if op == 'notEmpty':
                return f"{coll}->isEmpty()"
            return f"{coll}->notEmpty()"
        # size() OP N
        match = re.search(r'(self\.\w+)->size\(\)\s*(>=|<=|=|>|<)\s*(-?\d+(?:\.\d+)?)', body)
        if match:
            coll, op, value = match.group(1), match.group(2), match.group(3)
            inv_op = _invert_operator(op)
            if inv_op is None:
                return None
            return f"{coll}->size() {inv_op} {value}"
        return None

    # 4) Numeric comparisons (attr OP value or attr1 OP attr2)
    if pid in {
        'numeric_greater_than_value', 'numeric_less_than_value',
        'numeric_positive', 'numeric_non_negative', 'numeric_bounded',
        'two_attributes_not_equal', 'two_attributes_equal',
        'numeric_sum_constraint', 'numeric_difference_constraint',
        'numeric_product_constraint', 'range_constraint'
    }:
        # attr OP value
        match = re.search(r'(self\.\w+)\s*(>=|<=|=|==|<>|!=|>|<)\s*(-?\d+(?:\.\d+)?)', body)
        if match:
            left, op, value = match.group(1), match.group(2), match.group(3)
            inv_op = _invert_operator(op)
            if inv_op is None:
                return None
            return f"{left} {inv_op} {value}"

        # attr OP attr
        match = re.search(r'(self\.\w+)\s*(>=|<=|=|==|<>|!=|>|<)\s*(self\.\w+)', body)
        if match:
            left, op, right = match.group(1), match.group(2), match.group(3)
            inv_op = _invert_operator(op)
            if inv_op is None:
                return None
            return f"{left} {inv_op} {right}"
        return None

    # Fallback: unsupported for solver negation
    return None


def _check_consistency_with_verifier(
    constraints: List[OCLConstraint],
    verifier,
    silent: bool = False
) -> Optional[str]:
    """
    Return solver_result string ('sat'/'unsat'/'unknown') or None on error.
    """
    try:
        if silent:
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                result = verifier.check_consistency(constraints)
            finally:
                sys.stdout = old_stdout
        else:
            result = verifier.check_consistency(constraints)
        if not result.get('verified', False):
            return None
        return result.get('solver_result')
    except Exception:
        return None


def check_implication_z3(
    c1: OCLConstraint,
    c2: OCLConstraint,
    timeout_ms: int = 5000,
    verifier=None
) -> ImplicationRelation:
    """
    Check logical implication between two constraints using Z3.
    
    Strategy:
    1. Convert OCL to Z3-compatible logic (simplified)
    2. Check if C1 → C2 (by checking if ¬(C1 ∧ ¬C2) is valid)
    3. Check if C2 → C1
    4. Check if C1 ⟺ C2
    5. Check if C1 ∧ C2 is satisfiable
    
    Args:
        c1: First constraint
        c2: Second constraint
        timeout_ms: Z3 solver timeout in milliseconds
        verifier: FrameworkConstraintVerifier instance
        
    Returns:
        ImplicationRelation indicating relationship
    """
    # Real solver-based implication check using verification framework.
    # Requires a FrameworkConstraintVerifier instance.
    if verifier is None or not getattr(verifier, 'framework_available', False):
        return ImplicationRelation.INDEPENDENT

    # 1) Check contradiction: C1 ∧ C2 is UNSAT
    solver_result = _check_consistency_with_verifier([c1, c2], verifier, silent=True)
    if solver_result == 'unsat':
        return ImplicationRelation.CONTRADICTORY
    if solver_result not in ['sat', 'unsat']:
        return ImplicationRelation.INDEPENDENT

    # 2) Check C1 → C2 by testing C1 ∧ ¬C2 UNSAT
    not_c2 = _negated_constraint(c2)
    if not_c2 is None:
        return ImplicationRelation.INDEPENDENT
    solver_result = _check_consistency_with_verifier([c1, not_c2], verifier, silent=True)
    c1_implies_c2 = (solver_result == 'unsat')
    if solver_result not in ['sat', 'unsat']:
        c1_implies_c2 = False

    # 3) Check C2 → C1 by testing C2 ∧ ¬C1 UNSAT
    not_c1 = _negated_constraint(c1)
    if not_c1 is None:
        return ImplicationRelation.INDEPENDENT
    solver_result = _check_consistency_with_verifier([c2, not_c1], verifier, silent=True)
    c2_implies_c1 = (solver_result == 'unsat')
    if solver_result not in ['sat', 'unsat']:
        c2_implies_c1 = False

    if c1_implies_c2 and c2_implies_c1:
        return ImplicationRelation.EQUIVALENT
    if c1_implies_c2:
        return ImplicationRelation.C1_IMPLIES_C2
    if c2_implies_c1:
        return ImplicationRelation.C2_IMPLIES_C1
    return ImplicationRelation.INDEPENDENT


def check_syntactic_implication(c1: OCLConstraint, c2: OCLConstraint) -> Optional[ImplicationRelation]:
    """
    Check for syntactic implications (heuristic-based, fast).
    
    Detects simple cases like:
    - x > 10 implies x > 5
    - x = 5 implies x > 0
    - collection.size() > 10 implies collection.notEmpty()
    
    Args:
        c1: First constraint
        c2: Second constraint
        
    Returns:
        ImplicationRelation if detected, None otherwise
    """
    ocl1 = normalize_constraint_body(c1.ocl)
    ocl2 = normalize_constraint_body(c2.ocl)
    
    # Check if constraints are syntactically identical
    if ocl1 == ocl2:
        return ImplicationRelation.EQUIVALENT
    
    # Check for simple numeric implications
    # Pattern: self.attr <op> N
    numeric1 = _parse_numeric_constraint(ocl1)
    numeric2 = _parse_numeric_constraint(ocl2)
    if numeric1 and numeric2 and numeric1[0] == numeric2[0]:
        relation = _compare_numeric_constraints(numeric1, numeric2)
        if relation:
            return relation

    # Check for collection size/emptiness implications
    size1 = _parse_collection_size_constraint(ocl1)
    size2 = _parse_collection_size_constraint(ocl2)
    if size1 and size2 and size1[0] == size2[0]:
        relation = _compare_numeric_constraints(size1, size2)
        if relation:
            return relation
    
    # Check for subsumption in forAll patterns
    # collection->forAll(x | P and Q) implies collection->forAll(x | P)
    if '->' in ocl1 and '->' in ocl2:
        if 'forAll' in ocl1 and 'forAll' in ocl2:
            # Simplified check: if ocl1 contains ocl2 as substring
            if ocl2 in ocl1 and len(ocl1) > len(ocl2):
                return ImplicationRelation.C1_IMPLIES_C2
    
    return None


def _parse_numeric_constraint(ocl: str) -> Optional[Tuple[str, str, float]]:
    """
    Parse a simple numeric constraint: <var> <op> <number>
    Returns (var, op, value)
    """
    num_pattern = r'(\w+(?:\.\w+)*)\s*(>=|<=|=|>|<)\s*(-?\d+(?:\.\d+)?)'
    match = re.search(num_pattern, ocl)
    if not match:
        return None
    var, op, val = match.groups()
    try:
        return var, op, float(val)
    except ValueError:
        return None


def _parse_collection_size_constraint(ocl: str) -> Optional[Tuple[str, str, float]]:
    """
    Parse collection size/emptiness constraints:
    - collection->size() <op> N
    - collection->notEmpty()
    - collection->isEmpty()
    Returns (collection, op, value) where op is one of <, <=, =, >, >=
    """
    size_pattern = r'([\w.]+)->size\(\)\s*(>=|<=|=|>|<)\s*(-?\d+(?:\.\d+)?)'
    match = re.search(size_pattern, ocl)
    if match:
        coll, op, val = match.groups()
        try:
            return coll, op, float(val)
        except ValueError:
            return None

    if 'notEmpty()' in ocl:
        coll = extract_collection_name(ocl, 'notEmpty')
        if coll:
            return coll, '>=', 1.0

    if 'isEmpty()' in ocl:
        coll = extract_collection_name(ocl, 'isEmpty')
        if coll:
            return coll, '=', 0.0

    return None


def _compare_numeric_constraints(
    c1: Tuple[str, str, float],
    c2: Tuple[str, str, float]
) -> Optional[ImplicationRelation]:
    """
    Compare two numeric constraints on the same variable/collection.
    Returns implication/equivalence/contradiction if detected.
    """
    _, op1, v1 = c1
    _, op2, v2 = c2

    i1 = _interval_from(op1, v1)
    i2 = _interval_from(op2, v2)

    if _intervals_disjoint(i1, i2):
        return ImplicationRelation.CONTRADICTORY
    if _interval_implies(i1, i2) and _interval_implies(i2, i1):
        return ImplicationRelation.EQUIVALENT
    if _interval_implies(i1, i2):
        return ImplicationRelation.C1_IMPLIES_C2
    if _interval_implies(i2, i1):
        return ImplicationRelation.C2_IMPLIES_C1
    return None


def _interval_from(op: str, val: float) -> Dict[str, Optional[Tuple[float, bool]]]:
    """
    Represent interval as bounds:
    - lower: (value, strict)
    - upper: (value, strict)
    """
    lower = None
    upper = None
    if op == '>':
        lower = (val, True)
    elif op == '>=':
        lower = (val, False)
    elif op == '<':
        upper = (val, True)
    elif op == '<=':
        upper = (val, False)
    elif op == '=':
        lower = (val, False)
        upper = (val, False)
    return {'lower': lower, 'upper': upper}


def _interval_implies(i1: Dict[str, Optional[Tuple[float, bool]]],
                      i2: Dict[str, Optional[Tuple[float, bool]]]) -> bool:
    """Return True if interval i1 is a subset of i2."""
    l1, u1 = i1['lower'], i1['upper']
    l2, u2 = i2['lower'], i2['upper']

    # Lower bound check
    if l2 is not None:
        if l1 is None:
            return False
        v1, s1 = l1
        v2, s2 = l2
        if v1 < v2:
            return False
        if v1 == v2 and s2 and not s1:
            return False

    # Upper bound check
    if u2 is not None:
        if u1 is None:
            return False
        v1, s1 = u1
        v2, s2 = u2
        if v1 > v2:
            return False
        if v1 == v2 and s2 and not s1:
            return False

    return True


def _intervals_disjoint(i1: Dict[str, Optional[Tuple[float, bool]]],
                        i2: Dict[str, Optional[Tuple[float, bool]]]) -> bool:
    """Return True if intervals i1 and i2 are disjoint."""
    l = _max_lower(i1['lower'], i2['lower'])
    u = _min_upper(i1['upper'], i2['upper'])
    if l is None or u is None:
        return False
    lv, ls = l
    uv, us = u
    if lv > uv:
        return True
    if lv == uv and (ls or us):
        return True
    return False


def _max_lower(a: Optional[Tuple[float, bool]], b: Optional[Tuple[float, bool]]) -> Optional[Tuple[float, bool]]:
    if a is None:
        return b
    if b is None:
        return a
    if a[0] > b[0]:
        return a
    if b[0] > a[0]:
        return b
    return (a[0], a[1] or b[1])


def _min_upper(a: Optional[Tuple[float, bool]], b: Optional[Tuple[float, bool]]) -> Optional[Tuple[float, bool]]:
    if a is None:
        return b
    if b is None:
        return a
    if a[0] < b[0]:
        return a
    if b[0] < a[0]:
        return b
    return (a[0], a[1] or b[1])


def normalize_constraint_body(ocl: str) -> str:
    """
    Normalize OCL constraint for comparison.
    
    Removes:
    - Context declaration
    - Extra whitespace
    - Comments
    
    Args:
        ocl: OCL constraint text
        
    Returns:
        Normalized constraint body
    """
    # Remove context
    if 'inv:' in ocl:
        ocl = ocl.split('inv:', 1)[1]
    
    # Remove comments
    ocl = re.sub(r'--.*$', '', ocl, flags=re.MULTILINE)
    
    # Normalize whitespace
    ocl = ' '.join(ocl.split())
    
    return ocl.strip()


def extract_collection_name(ocl: str, operation: str) -> Optional[str]:
    """
    Extract collection name from OCL expression.
    
    Example: 'self.books->notEmpty()' -> 'self.books'
    """
    pattern = rf'([\w.]+)->(?:\w+\(.*?\))*{operation}'
    match = re.search(pattern, ocl)
    if match:
        return match.group(1)
    return None


def find_implications(
    constraints: List[OCLConstraint],
    use_z3: bool = False,
    timeout_ms: int = 5000,
    same_context_only: bool = True,
    verifier=None
) -> List[Tuple[int, int, ImplicationRelation]]:
    """
    Find all implication relationships in constraint set.
    
    Args:
        constraints: List of OCLConstraints
        use_z3: Whether to use Z3 solver (slow but precise)
        timeout_ms: Z3 timeout per query
        
    Returns:
        List of (index1, index2, relation) tuples
    """
    implications = []
    n = len(constraints)
    
    for i in range(n):
        for j in range(i + 1, n):
            if same_context_only:
                c1_ctx = getattr(constraints[i], 'context', None)
                c2_ctx = getattr(constraints[j], 'context', None)
                if c1_ctx and c2_ctx and c1_ctx != c2_ctx:
                    continue
            # Try syntactic check first (fast)
            relation = check_syntactic_implication(constraints[i], constraints[j])
            
            if relation is None and use_z3:
                # Skip solver implication checks for UNSAT constraints
                if constraints[i].metadata.get('is_unsat') or constraints[j].metadata.get('is_unsat'):
                    continue
                # Fall back to Z3 (slow but thorough)
                relation = check_implication_z3(constraints[i], constraints[j], timeout_ms, verifier=verifier)
            
            if relation and relation != ImplicationRelation.INDEPENDENT:
                implications.append((i, j, relation))
    
    return implications


def prune_implied_constraints(
    constraints: List[OCLConstraint],
    use_z3: bool = False,
    same_context_only: bool = True
) -> List[OCLConstraint]:
    """
    Remove logically redundant constraints based on implications.
    
    Strategy:
    - If C1 → C2, keep C1 (stronger constraint)
    - If C1 ⟺ C2, keep C1 (arbitrary choice)
    - If C1 contradicts C2, keep both (different requirements)
    
    Args:
        constraints: List of OCLConstraints
        use_z3: Whether to use Z3 solver
        
    Returns:
        Pruned list of constraints
    """
    implications = find_implications(constraints, use_z3=use_z3, same_context_only=same_context_only)
    
    # Track indices to remove
    to_remove = set()
    
    for i, j, relation in implications:
        if relation == ImplicationRelation.C1_IMPLIES_C2:
            # C1 is stronger, remove C2
            to_remove.add(j)
        elif relation == ImplicationRelation.C2_IMPLIES_C1:
            # C2 is stronger, remove C1
            to_remove.add(i)
        elif relation == ImplicationRelation.EQUIVALENT:
            # Keep first, remove second
            to_remove.add(j)
        # CONTRADICTORY and INDEPENDENT: keep both
    
    # Return constraints not in removal set
    return [c for idx, c in enumerate(constraints) if idx not in to_remove]


def compute_implication_graph(
    constraints: List[OCLConstraint],
    use_z3: bool = False,
    same_context_only: bool = True
) -> Dict[str, Any]:
    """
    Compute implication graph showing logical relationships.
    
    Returns dictionary with:
    - nodes: List of constraint indices
    - edges: List of (source, target, relation) tuples
    - strongly_connected_components: Groups of equivalent constraints
    
    Args:
        constraints: List of OCLConstraints
        use_z3: Whether to use Z3 solver
        
    Returns:
        Implication graph structure
    """
    implications = find_implications(constraints, use_z3=use_z3, same_context_only=same_context_only)
    
    # Build adjacency list
    graph = {i: [] for i in range(len(constraints))}
    equivalence_groups = []
    
    for i, j, relation in implications:
        if relation == ImplicationRelation.C1_IMPLIES_C2:
            graph[i].append((j, 'implies'))
        elif relation == ImplicationRelation.C2_IMPLIES_C1:
            graph[j].append((i, 'implies'))
        elif relation == ImplicationRelation.EQUIVALENT:
            # Track equivalence groups
            added = False
            for group in equivalence_groups:
                if i in group or j in group:
                    group.add(i)
                    group.add(j)
                    added = True
                    break
            if not added:
                equivalence_groups.append({i, j})
    
    return {
        'nodes': list(range(len(constraints))),
        'edges': [(i, j, rel.value) for i, j, rel in implications],
        'implication_count': len([r for _, _, r in implications 
                                   if r in [ImplicationRelation.C1_IMPLIES_C2, 
                                           ImplicationRelation.C2_IMPLIES_C1]]),
        'equivalence_groups': [list(g) for g in equivalence_groups],
        'contradictions': [(i, j) for i, j, r in implications 
                          if r == ImplicationRelation.CONTRADICTORY]
    }


def get_implication_statistics(
    constraints: List[OCLConstraint],
    use_z3: bool = False,
    same_context_only: bool = True
) -> Dict[str, Any]:
    """
    Compute statistics about logical relationships in constraint set.
    
    Args:
        constraints: List of OCLConstraints
        use_z3: Whether to use Z3 solver
        
    Returns:
        Dictionary with statistics
    """
    implications = find_implications(constraints, use_z3=use_z3, same_context_only=same_context_only)
    
    stats = {
        'total_constraints': len(constraints),
        'total_pairs_checked': len(constraints) * (len(constraints) - 1) // 2,
        'relationships_found': len(implications),
        'implications': 0,
        'equivalences': 0,
        'contradictions': 0,
        'independent_pairs': 0
    }
    
    for _, _, relation in implications:
        if relation == ImplicationRelation.C1_IMPLIES_C2 or relation == ImplicationRelation.C2_IMPLIES_C1:
            stats['implications'] += 1
        elif relation == ImplicationRelation.EQUIVALENT:
            stats['equivalences'] += 1
        elif relation == ImplicationRelation.CONTRADICTORY:
            stats['contradictions'] += 1
    
    stats['independent_pairs'] = stats['total_pairs_checked'] - stats['relationships_found']
    
    # Estimate redundancy
    if stats['total_constraints'] > 0:
        redundant = stats['implications'] + stats['equivalences']
        stats['redundancy_ratio'] = redundant / stats['total_constraints']
    else:
        stats['redundancy_ratio'] = 0.0
    
    return stats
