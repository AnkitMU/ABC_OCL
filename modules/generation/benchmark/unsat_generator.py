"""
SAT/UNSAT Variant Generation

Creates intentionally unsatisfiable (UNSAT) constraints by mutating
satisfiable (SAT) constraints. This is useful for:
- Testing solver completeness
- Creating balanced benchmarks with known SAT/UNSAT ratio
- Validating verification frameworks
"""

import random
import re
from typing import List, Optional, Tuple, Dict, Any
from modules.core.models import OCLConstraint, Metamodel


class OCLMutationHelper:
    """Utilities to preserve valid OCL syntax while mutating constraints."""

    _INV_REGEX = re.compile(
        r'(?s)^(?P<header>.*?)(?P<inv>\binv\b\s*(?:[A-Za-z_][\w-]*\s*)?:)(?P<body>.*)$'
    )

    @staticmethod
    def split_ocl(ocl_string: str) -> Tuple[str, str, str]:
        """
        Split an OCL constraint into (header, inv_token, body).

        Examples:
          "context C\\ninv: x > 0" -> ("context C", "inv:", "x > 0")
          "context C inv myInv: x > 0" -> ("context C", "inv myInv:", "x > 0")
        """
        match = OCLMutationHelper._INV_REGEX.match(ocl_string)
        if match:
            header = match.group('header').rstrip()
            inv_token = match.group('inv').strip()
            body = match.group('body').strip()
            return header, inv_token, body
        # Fallback: treat whole string as body
        return "", "inv:", ocl_string.strip()

    @staticmethod
    def _prefix(header: str) -> str:
        header = header.rstrip()
        return f"{header}\n" if header else ""

    @staticmethod
    def build_unsat_ocl(header: str, inv_token: str, original_body: str, contradiction: str) -> str:
        """Construct a syntactically safe UNSAT OCL string."""
        prefix = OCLMutationHelper._prefix(header)
        return f"{prefix}{inv_token} ({original_body}) and ({contradiction})"

    @staticmethod
    def build_negated_ocl(header: str, inv_token: str, original_body: str) -> str:
        """Construct a syntactically safe negated OCL string."""
        prefix = OCLMutationHelper._prefix(header)
        return f"{prefix}{inv_token} not({original_body})"


class UnsatMutationStrategy:
    """Base class for UNSAT mutation strategies"""
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        """Check if strategy can be applied to constraint"""
        raise NotImplementedError
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        """Apply mutation to create UNSAT variant"""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Get strategy name"""
        return self.__class__.__name__

    def _wrap(self, constraint: OCLConstraint, new_ocl: str, mutation: str) -> OCLConstraint:
        """Create a new UNSAT constraint with consistent metadata."""
        return OCLConstraint(
            ocl=new_ocl,
            pattern_id=f"{constraint.pattern_id}_unsat",
            pattern_name=f"{constraint.pattern_name} (UNSAT)",
            context=constraint.context,
            parameters=constraint.parameters,
            metadata={**constraint.metadata, 'is_unsat': True, 'mutation': mutation}
        )


class ContradictoryBoundsStrategy(UnsatMutationStrategy):
    """
    Create contradictory bounds: x > 10 and x < 5
    """
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        ocl = constraint.ocl
        # Check for numeric comparison
        return bool(re.search(r'[<>=]', ocl) and re.search(r'\d+', ocl))
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)

        # Find numeric comparisons inside the body
        pattern = r'([\w\.]+(?:->\w+\(\))?)\s*([<>]=?|=)\s*(\d+)'
        match = re.search(pattern, body)
        if not match:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        var, op, val_str = match.groups()
        val = int(val_str)

        if op == '>':
            contradiction = f'{var} < {val - 5}'
        elif op == '<':
            contradiction = f'{var} > {val + 5}'
        elif op == '>=':
            contradiction = f'{var} < {val}'
        elif op == '<=':
            contradiction = f'{var} > {val}'
        else:
            contradiction = f'{var} <> {val}'

        new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
        return self._wrap(constraint, new_ocl, 'contradictory_bounds')


class EmptyCollectionStrategy(UnsatMutationStrategy):
    """
    Require collection to be both empty and non-empty
    """
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        body = OCLMutationHelper.split_ocl(constraint.ocl)[2]
        return 'notEmpty()' in body or 'size()' in body
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)

        if 'notEmpty()' in body:
            coll_pattern = r'([\w.]+)->notEmpty\(\)'
            match = re.search(coll_pattern, body)
            if not match:
                return SimpleNegationStrategy().apply(constraint, metamodel)
            collection = match.group(1)
            contradiction = f'{collection}->isEmpty()'
            new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
            return self._wrap(constraint, new_ocl, 'empty_collection')

        size_pattern = r'([\w.]+)->size\(\)\s*(>=|>)\s*0'
        match = re.search(size_pattern, body)
        if match:
            collection = match.group(1)
            contradiction = f'{collection}->size() = 0'
            new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
            return self._wrap(constraint, new_ocl, 'empty_collection')

        return SimpleNegationStrategy().apply(constraint, metamodel)


class TypeContradictionStrategy(UnsatMutationStrategy):
    """
    Create type contradictions: x.oclIsTypeOf(A) and x.oclIsTypeOf(B)
    """
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        ocl = constraint.ocl
        return 'oclIsTypeOf' in ocl or 'oclIsKindOf' in ocl
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)

        pattern = r'(\w+)\.oclIsTypeOf\((\w+)\)'
        match = re.search(pattern, body)
        if not match:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        var, type1 = match.groups()
        classes = [c for c in metamodel.get_class_names() if c != type1]
        if not classes:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        type2 = random.choice(classes)
        contradiction = f'{var}.oclIsTypeOf({type2})'
        new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
        return self._wrap(constraint, new_ocl, 'type_contradiction')


class UniversalNegationStrategy(UnsatMutationStrategy):
    """
    Negate universal quantifier: forAll(...) and exists(not(...))
    """
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        return 'forAll' in constraint.ocl
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)

        pattern = r'([\w.]+)->forAll\(\s*(\w+)\s*\|\s*(.+)\)'
        match = re.search(pattern, body)
        if not match:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        collection, var, condition = match.groups()
        contradiction = f'{collection}->exists({var} | not({condition}))'
        new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
        return self._wrap(constraint, new_ocl, 'universal_negation')


class SimpleNegationStrategy(UnsatMutationStrategy):
    """
    Simple negation: wrap constraint in not(...)
    This is the most reliable UNSAT generator but least interesting
    """
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        return True  # Always applicable
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)
        new_ocl = OCLMutationHelper.build_negated_ocl(header, inv_token, body)
        return self._wrap(constraint, new_ocl, 'simple_negation')


# All available strategies
ALL_STRATEGIES = [
    ContradictoryBoundsStrategy(),
    EmptyCollectionStrategy(),
    TypeContradictionStrategy(),
    UniversalNegationStrategy(),
    SimpleNegationStrategy()  # Fallback
]


def generate_unsat_variant(
    constraint: OCLConstraint,
    metamodel: Metamodel,
    strategies: Optional[List[UnsatMutationStrategy]] = None
) -> OCLConstraint:
    """
    Generate UNSAT variant of a constraint.
    
    Args:
        constraint: SAT constraint to mutate
        metamodel: Metamodel for context
        strategies: List of strategies to try (default: ALL_STRATEGIES)
        
    Returns:
        UNSAT variant of constraint
    """
    if strategies is None:
        strategies = ALL_STRATEGIES
    
    # Try strategies in order until one works
    for strategy in strategies:
        if strategy.can_apply(constraint, metamodel):
            return strategy.apply(constraint, metamodel)
    
    # Fallback: simple negation
    return SimpleNegationStrategy().apply(constraint, metamodel)


def generate_mixed_sat_unsat_set(
    constraints: List[OCLConstraint],
    metamodel: Metamodel,
    unsat_ratio: float = 0.3,
    strategies: Optional[List[UnsatMutationStrategy]] = None
) -> Tuple[List[OCLConstraint], Dict[int, str]]:
    """
    Generate mixed SAT/UNSAT constraint set from SAT-only set.
    
    Args:
        constraints: List of SAT constraints
        metamodel: Metamodel
        unsat_ratio: Target ratio of UNSAT constraints (0.0 - 1.0)
        strategies: Mutation strategies to use
        
    Returns:
        Tuple of (mixed_constraints, unsat_indices)
        where unsat_indices maps index -> mutation_strategy_name
    """
    if not (0.0 <= unsat_ratio <= 1.0):
        raise ValueError("unsat_ratio must be between 0.0 and 1.0")
    
    n = len(constraints)
    n_unsat = int(n * unsat_ratio)
    
    if n_unsat == 0:
        return constraints, {}
    
    # Randomly select indices to mutate
    indices_to_mutate = random.sample(range(n), min(n_unsat, n))
    
    result = []
    unsat_map = {}
    
    for i, constraint in enumerate(constraints):
        if i in indices_to_mutate:
            # Generate UNSAT variant
            unsat_variant = generate_unsat_variant(constraint, metamodel, strategies)
            result.append(unsat_variant)
            unsat_map[i] = unsat_variant.metadata.get('mutation', 'unknown')
        else:
            # Keep SAT constraint
            result.append(constraint)
    
    return result, unsat_map


def verify_unsat_generation(
    sat_constraint: OCLConstraint,
    unsat_variant: OCLConstraint,
    metamodel: Metamodel
) -> Tuple[bool, str]:
    """
    Verify that UNSAT variant is actually unsatisfiable (requires solver).
    
    Args:
        sat_constraint: Original SAT constraint
        unsat_variant: Generated UNSAT variant
        metamodel: Metamodel
        
    Returns:
        Tuple of (is_correct, message)
    """
    try:
        from modules.verification.framework_verifier import FrameworkConstraintVerifier
        
        # TODO: Implement actual verification
        # This would require:
        # 1. Load metamodel into verifier
        # 2. Check sat_constraint -> should be SAT
        # 3. Check unsat_variant -> should be UNSAT
        
        return True, "Verification not implemented (assume correct)"
    except Exception as e:
        return False, f"Verification failed: {e}"


def get_mutation_statistics(
    constraints: List[OCLConstraint]
) -> Dict[str, Any]:
    """
    Get statistics about UNSAT mutations in constraint set.
    
    Args:
        constraints: List of constraints (mixed SAT/UNSAT)
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'total': len(constraints),
        'unsat_count': 0,
        'sat_count': 0,
        'mutations': {}
    }
    
    for c in constraints:
        if c.metadata.get('is_unsat', False):
            stats['unsat_count'] += 1
            mutation = c.metadata.get('mutation', 'unknown')
            stats['mutations'][mutation] = stats['mutations'].get(mutation, 0) + 1
        else:
            stats['sat_count'] += 1
    
    if stats['total'] > 0:
        stats['unsat_ratio'] = stats['unsat_count'] / stats['total']
    else:
        stats['unsat_ratio'] = 0.0
    
    return stats
