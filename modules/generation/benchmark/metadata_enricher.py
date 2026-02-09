"""
Rich Metadata Extraction for OCL Constraints

Extracts operators, navigation depth, quantifier depth, and difficulty labels
to create research-grade benchmark metadata.
"""

import re
from typing import List, Set, Dict, Any
from modules.core.models import OCLConstraint
from .coverage_tracker import nav_hops, quantifier_depth as coverage_quantifier_depth


# OCL Operators (grouped by category)
COMPARISON_OPS = ['=', '<>', '<', '>', '<=', '>=']
LOGICAL_OPS = ['and', 'or', 'not', 'implies', 'xor']
COLLECTION_OPS = [
    'forAll', 'exists', 'select', 'reject', 'collect', 
    'size', 'isEmpty', 'notEmpty', 'includes', 'excludes',
    'includesAll', 'excludesAll', 'sum', 'count', 'any',
    'one', 'isUnique', 'sortedBy', 'closure', 'intersection',
    'union', 'including', 'excluding', 'symmetricDifference',
    'flatten', 'asSet', 'asBag', 'asSequence', 'asOrderedSet',
    'first', 'last', 'at', 'indexOf', 'append', 'prepend'
]
STRING_OPS = [
    'concat', 'substring', 'toUpper', 'toLower', 'size',
    'indexOf', 'equalsIgnoreCase', 'startsWith', 'endsWith',
    'characters', 'toInteger', 'toReal', 'matches'
]
ARITHMETIC_OPS = ['+', '-', '*', '/', 'abs', 'div', 'mod', 'max', 'min', 'round', 'floor']
TYPE_OPS = ['oclIsTypeOf', 'oclIsKindOf', 'oclAsType', 'allInstances']
OTHER_OPS = ['if', 'then', 'else', 'endif', 'let', 'in']

ALL_OPERATORS = (
    COMPARISON_OPS + LOGICAL_OPS + COLLECTION_OPS + 
    STRING_OPS + ARITHMETIC_OPS + TYPE_OPS + OTHER_OPS
)


def extract_operators(ocl: str) -> List[str]:
    """
    Extract all OCL operators used in a constraint.
    
    Args:
        ocl: OCL constraint text
        
    Returns:
        List of operator names (deduplicated and sorted)
    """
    found_ops = set()
    ocl_lower = ocl.lower()
    
    # Extract word-based operators (forAll, exists, etc.)
    for op in ALL_OPERATORS:
        if isinstance(op, str) and op.isalpha():
            # Use word boundaries to avoid false positives
            pattern = r'\b' + re.escape(op.lower()) + r'\b'
            if re.search(pattern, ocl_lower):
                found_ops.add(op)
    
    # Extract symbol operators (=, <>, +, -, etc.)
    for op in COMPARISON_OPS + ARITHMETIC_OPS:
        if not op.isalpha():
            if op in ocl:
                found_ops.add(op)
    
    return sorted(list(found_ops))


def count_navigation_depth(ocl: str) -> int:
    """
    Count maximum navigation depth (number of consecutive dot navigations).
    
    Example:
        self.library.books.author.name  -> depth = 4
        self.name -> depth = 1
    
    Args:
        ocl: OCL constraint text
        
    Returns:
        Maximum navigation depth
    """
    # Extract the constraint body (remove context declaration)
    body = ocl
    if 'inv:' in ocl:
        body = ocl.split('inv:', 1)[1]
    
    # Find all navigation chains (sequences of identifiers separated by dots)
    # Pattern: word followed by one or more (.word)
    navigation_pattern = r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+'
    navigations = re.findall(navigation_pattern, body)
    
    if not navigations:
        return 0
    
    # Count dots in each navigation chain to get depth
    max_depth = 0
    for nav in navigations:
        # Count dots + 1 to get number of identifiers
        depth = nav.count('.') + 1
        max_depth = max(max_depth, depth)
    
    return max_depth


def count_quantifier_depth(ocl: str) -> int:
    """
    Count maximum nesting depth of quantifiers (forAll, exists, select, etc.).
    
    Example:
        self.books->forAll(b | b.authors->exists(a | a.age > 30))  -> depth = 2
    
    Args:
        ocl: OCL constraint text
        
    Returns:
        Maximum quantifier nesting depth
    """
    # Collection operations that introduce quantifier scope
    quantifiers = ['forAll', 'exists', 'select', 'reject', 'collect', 
                   'one', 'any', 'sortedBy', 'closure']
    
    max_depth = 0
    current_depth = 0
    
    # Tokenize OCL to track quantifier nesting
    tokens = re.findall(r'\b\w+\b|[(){}|]', ocl)
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Check if this is a quantifier
        if token in quantifiers:
            # Look ahead for opening parenthesis and pipe (quantifier pattern)
            if i + 2 < len(tokens):
                # Pattern: quantifier ( ... | ... )
                current_depth += 1
                max_depth = max(max_depth, current_depth)
        
        # Track scope ending with closing parenthesis
        elif token == ')':
            # Check if this closes a quantifier scope
            # Simple heuristic: decrease depth if we've entered quantifiers
            if current_depth > 0:
                current_depth -= 1
        
        i += 1
    
    return max_depth


def classify_difficulty(constraint: OCLConstraint) -> str:
    """
    Classify constraint difficulty based on operators, depth, and complexity.
    
    Difficulty levels:
        - trivial: Basic attribute access, simple comparisons
        - easy: Single collection operation, 1-2 operators
        - medium: Multiple operators, moderate navigation/quantifier depth
        - hard: Deep nesting, complex operators, multiple quantifiers
        - expert: Very deep nesting, advanced OCL features
    
    Args:
        constraint: OCLConstraint with metadata
        
    Returns:
        Difficulty label: "trivial", "easy", "medium", "hard", or "expert"
    """
    ocl = constraint.ocl
    
    # Extract features
    operators = extract_operators(ocl)
    nav_depth = count_navigation_depth(ocl)
    quant_depth = count_quantifier_depth(ocl)
    
    # Count operator types
    has_collection_ops = any(op in operators for op in COLLECTION_OPS)
    has_quantifiers = any(op in operators for op in ['forAll', 'exists', 'select', 'reject'])
    has_advanced_ops = any(op in operators for op in ['closure', 'iterate', 'oclIsTypeOf', 'allInstances'])
    has_let = 'let' in operators
    
    # Scoring system
    score = 0
    
    # Base complexity from operator count
    score += len(operators)
    
    # Navigation depth contribution
    if nav_depth >= 4:
        score += 3
    elif nav_depth >= 3:
        score += 2
    elif nav_depth >= 2:
        score += 1
    
    # Quantifier depth contribution (heavily weighted)
    if quant_depth >= 3:
        score += 5
    elif quant_depth >= 2:
        score += 3
    elif quant_depth >= 1:
        score += 2
    
    # Advanced features
    if has_advanced_ops:
        score += 3
    if has_let:
        score += 2
    
    # Classify based on score
    if score <= 2:
        return "trivial"
    elif score <= 5:
        return "easy"
    elif score <= 10:
        return "medium"
    elif score <= 15:
        return "hard"
    else:
        return "expert"


def enrich_constraint_metadata(constraint: OCLConstraint) -> OCLConstraint:
    """
    Enrich OCLConstraint with operators_used, navigation_depth, 
    quantifier_depth, and difficulty metadata.
    
    Args:
        constraint: OCLConstraint to enrich
        
    Returns:
        Same constraint with updated metadata dict
    """
    # Extract metadata
    operators = extract_operators(constraint.ocl)
    nav_depth = count_navigation_depth(constraint.ocl)
    quant_depth = count_quantifier_depth(constraint.ocl)
    difficulty = classify_difficulty(constraint)
    
    # Update metadata dict (preserve existing metadata)
    constraint.metadata.update({
        'operators_used': operators,
        'navigation_depth': nav_depth,
        'quantifier_depth': quant_depth,
        'difficulty': difficulty,
        'operator_count': len(operators)
    })
    
    return constraint


def extract_families(constraint: OCLConstraint) -> List[str]:
    """
    Extract constraint families based on operators used.
    
    Families:
        - basic: Simple attribute/association constraints
        - collection: Collection operations
        - navigation: Multi-hop navigation
        - quantifier: Universal/existential quantifiers
        - arithmetic: Numeric operations
        - string: String operations
        - advanced: Advanced OCL features
    
    Args:
        constraint: OCLConstraint
        
    Returns:
        List of family names
    """
    ocl = constraint.ocl
    operators = constraint.metadata.get('operators_used', extract_operators(ocl))
    nav_depth = constraint.metadata.get('navigation_depth', count_navigation_depth(ocl))
    quant_depth = constraint.metadata.get('quantifier_depth', count_quantifier_depth(ocl))
    
    families = set()
    
    # Determine families
    if any(op in operators for op in COLLECTION_OPS):
        families.add('collection')
    
    if any(op in operators for op in ['forAll', 'exists']):
        families.add('quantifier')
    
    if nav_depth >= 2:
        families.add('navigation')
    
    if any(op in operators for op in ARITHMETIC_OPS):
        families.add('arithmetic')
    
    if any(op in operators for op in STRING_OPS):
        families.add('string')
    
    if any(op in operators for op in ['closure', 'iterate', 'oclIsTypeOf', 'allInstances', 'let']):
        families.add('advanced')
    
    # Default to basic if no specific family
    if not families:
        families.add('basic')
    
    return sorted(list(families))


def get_enrichment_summary(constraints: List[OCLConstraint]) -> Dict[str, Any]:
    """
    Get summary statistics of enriched constraints.
    
    Args:
        constraints: List of enriched OCLConstraints
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not constraints:
        return {}
    
    difficulties = {}
    all_operators = set()
    nav_depths = []
    quant_depths = []
    families_count = {}
    
    for c in constraints:
        # Difficulty distribution
        diff = c.metadata.get('difficulty', 'unknown')
        difficulties[diff] = difficulties.get(diff, 0) + 1
        
        # All operators used
        ops = c.metadata.get('operators_used', [])
        all_operators.update(ops)
        
        # Depth distributions
        nav_depths.append(c.metadata.get('navigation_depth', 0))
        quant_depths.append(c.metadata.get('quantifier_depth', 0))
        
        # Family distribution
        fams = extract_families(c)
        for fam in fams:
            families_count[fam] = families_count.get(fam, 0) + 1
    
    return {
        'total_constraints': len(constraints),
        'difficulty_distribution': difficulties,
        'unique_operators': len(all_operators),
        'operators_used': sorted(list(all_operators)),
        'navigation_depth': {
            'max': max(nav_depths) if nav_depths else 0,
            'avg': sum(nav_depths) / len(nav_depths) if nav_depths else 0,
            'distribution': {i: nav_depths.count(i) for i in range(max(nav_depths) + 1)} if nav_depths else {}
        },
        'quantifier_depth': {
            'max': max(quant_depths) if quant_depths else 0,
            'avg': sum(quant_depths) / len(quant_depths) if quant_depths else 0,
            'distribution': {i: quant_depths.count(i) for i in range(max(quant_depths) + 1)} if quant_depths else {}
        },
        'family_distribution': families_count
    }


def normalize_ocl(ocl: str) -> str:
    """Normalize OCL string for lightweight similarity comparisons."""
    s = re.sub(r"\s+", " ", ocl)
    s = s.replace("self.", "$")
    return s.strip().lower()


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity over token sets."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def similarity(c1: OCLConstraint, c2: OCLConstraint) -> float:
    """Lightweight similarity between two constraints."""
    return jaccard(normalize_ocl(c1.ocl), normalize_ocl(c2.ocl))


def difficulty_score(ocl: str) -> int:
    """
    Coarse difficulty bucket based on navigation hops and quantifier depth.
    Returns: 0 (easy), 1 (medium), 2 (hard)
    """
    hops = nav_hops(ocl)
    depth = coverage_quantifier_depth(ocl)
    score = hops + depth
    if score <= 1:
        return 0  # easy
    if score <= 3:
        return 1  # medium
    return 2      # hard
