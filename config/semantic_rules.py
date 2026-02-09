"""
Semantic Attribute Rules for Business Logic Generation

Provides domain knowledge about attribute semantics to prevent
nonsensical constraint generation (e.g., dateFrom = dateTo).
"""

from typing import Dict, List, Set, Tuple

# ===== ATTRIBUTE SEMANTIC GROUPS =====

TEMPORAL_ATTRIBUTES = {
    'start', 'end', 'from', 'to', 'date', 'timestamp', 'time',
    'created', 'updated', 'deleted', 'modified', 'expired',
    'dateFrom', 'dateTo', 'startDate', 'endDate', 'createdAt', 'updatedAt'
}

IDENTIFIER_ATTRIBUTES = {
    'id', 'code', 'key', 'number', 'plate', 'license', 'ssn',
    'email', 'username', 'phone', 'vin', 'serial'
}

MEASUREMENT_ATTRIBUTES = {
    'amount', 'price', 'cost', 'total', 'subtotal', 'discount',
    'quantity', 'count', 'size', 'length', 'width', 'height',
    'weight', 'volume', 'distance', 'mileage', 'tankLevel'
}

STATUS_ATTRIBUTES = {
    'status', 'state', 'type', 'category', 'level', 'priority',
    'available', 'active', 'enabled', 'isActive', 'isPremium'
}

DESCRIPTIVE_ATTRIBUTES = {
    'name', 'title', 'description', 'comment', 'notes', 'message',
    'firstName', 'lastName', 'address', 'city', 'country'
}


# ===== SEMANTIC COMPATIBILITY RULES =====

def get_attribute_semantic_group(attr_name: str) -> str:
    """Get semantic group for an attribute."""
    attr_lower = attr_name.lower()
    
    if any(keyword in attr_lower for keyword in TEMPORAL_ATTRIBUTES):
        return 'temporal'
    elif any(keyword in attr_lower for keyword in IDENTIFIER_ATTRIBUTES):
        return 'identifier'
    elif any(keyword in attr_lower for keyword in MEASUREMENT_ATTRIBUTES):
        return 'measurement'
    elif any(keyword in attr_lower for keyword in STATUS_ATTRIBUTES):
        return 'status'
    elif any(keyword in attr_lower for keyword in DESCRIPTIVE_ATTRIBUTES):
        return 'descriptive'
    else:
        return 'unknown'


# ===== FORBIDDEN EQUALITY COMBINATIONS =====

FORBIDDEN_EQUALITY_PAIRS = {
    # Temporal bounds should use comparison, not equality
    ('temporal', 'temporal'): ['dateFrom/dateTo', 'startDate/endDate', 'createdAt/updatedAt'],
    
    # Different measurement types shouldn't be equal
    ('measurement', 'measurement'): ['mileage/tankLevel', 'price/quantity', 'weight/volume'],
    
    # Identifiers vs non-identifiers rarely equal
    ('identifier', 'measurement'): ['*/*'],  # * = all combinations
    ('identifier', 'temporal'): ['*/*'],
    ('identifier', 'descriptive'): ['*/*'],
    
    # Status vs measurements
    ('status', 'measurement'): ['*/*'],
}


def is_valid_equality_pair(attr1: str, attr2: str, class_name: str = None) -> bool:
    """
    Check if two attributes make semantic sense for equality comparison.
    
    Args:
        attr1: First attribute name
        attr2: Second attribute name
        class_name: Optional context class name
        
    Returns:
        True if the pair makes semantic sense, False otherwise
    """
    # Same attribute always invalid (tautology)
    if attr1 == attr2:
        return False
    
    # Get semantic groups
    group1 = get_attribute_semantic_group(attr1)
    group2 = get_attribute_semantic_group(attr2)
    
    # Check forbidden pairs
    pair_key = tuple(sorted([group1, group2]))
    
    if pair_key in FORBIDDEN_EQUALITY_PAIRS:
        forbidden_list = FORBIDDEN_EQUALITY_PAIRS[pair_key]
        
        # Check wildcard rules (*/* means all combinations forbidden)
        if '*/*' in forbidden_list:
            return False
        
        # Check specific attribute name patterns
        for forbidden_pattern in forbidden_list:
            parts = forbidden_pattern.split('/')
            if len(parts) == 2:
                pattern1, pattern2 = parts
                if (pattern1 in attr1.lower() and pattern2 in attr2.lower()) or \
                   (pattern1 in attr2.lower() and pattern2 in attr1.lower()):
                    return False
    
    # Special case: temporal start/end pairs
    if group1 == 'temporal' and group2 == 'temporal':
        # Check if it's a start/end pair
        if any(start_word in attr1.lower() for start_word in ['start', 'from', 'begin']) and \
           any(end_word in attr2.lower() for end_word in ['end', 'to', 'finish']):
            return False  # Use comparison instead
    
    return True


def get_preferred_operator_for_pair(attr1: str, attr2: str) -> str:
    """
    Get preferred comparison operator for attribute pair.
    
    Returns:
        Operator: '<=', '>=', '<', '>', '=' or None
    """
    group1 = get_attribute_semantic_group(attr1)
    group2 = get_attribute_semantic_group(attr2)
    
    # Temporal: start <= end
    if group1 == 'temporal' and group2 == 'temporal':
        if any(word in attr1.lower() for word in ['start', 'from', 'begin']):
            return '<='  # startDate <= endDate
        elif any(word in attr2.lower() for word in ['start', 'from', 'begin']):
            return '>='  # endDate >= startDate
    
    # Measurements: use any comparison
    if group1 == 'measurement' and group2 == 'measurement':
        # Price comparisons, quantity ranges, etc.
        return '<='  # Prefer <= for ranges
    
    # Same group, same type: equality might make sense
    if group1 == group2 and group1 in ['identifier', 'status', 'descriptive']:
        return '='
    
    return None  # No strong preference


# ===== PATTERN TEMPLATE VALIDATION =====

def is_valid_pattern_template(pattern_id: str, template: str) -> Tuple[bool, str]:
    """
    Validate if a pattern template is logically sound.
    
    Args:
        pattern_id: Pattern identifier  
        template: OCL template string
        
    Returns:
        (is_valid, reason)
    """
    template_lower = template.lower()
    
    # Tautologies: Always true patterns
    TAUTOLOGY_PATTERNS = [
        ('isempty()', 'notempty()'),  # isEmpty() or notEmpty() is always true
    ]
    
    for part1, part2 in TAUTOLOGY_PATTERNS:
        if part1 in template_lower and part2 in template_lower and ' or ' in template_lower:
            return False, f"Tautology: Pattern always evaluates to true"
    
    # Contradictions: Always false patterns  
    if '<> null and' in template_lower and '= null' in template_lower:
        # Check if same attribute: x <> null AND x = null
        import re
        match = re.search(r'(\{\w+\}|self\.\w+)\s*<>\s*null\s+and\s+\1\s*=\s*null', template_lower)
        if match:
            return False, f"Contradiction: Pattern always evaluates to false"
    
    # Self-comparison tautologies: x = x, x - x, etc.
    SELF_COMPARE_PATTERNS = [
        r'(\{\w+\})\s*-\s*\1',  # {attr} - {attr}
        r'(\{\w+\})\s*=\s*\1',   # {attr} = {attr} (without different param names)
    ]
    
    for pattern in SELF_COMPARE_PATTERNS:
        import re
        if re.search(pattern, template):
            # Check if it's the SAME parameter (not first_attr vs second_attr)
            if not ('first_' in template and 'second_' in template):
                if not ('attribute1' in template and 'attribute2' in template):
                    return False, f"Self-comparison: Pattern compares same value to itself"
    
    # Non-constraints: Expressions without assertions
    if '->closure(' in template_lower and not any(op in template_lower for op in ['=', '>', '<', 'includes', 'excludes']):
        return False, f"Not a constraint: Expression without comparison/assertion"
    
    if '->product(' in template_lower and not any(op in template_lower for op in ['=', '>', '<', '->size()', 'includes']):
        return False, f"Not a constraint: Product expression without comparison"
    
    # String operations without assertions
    if '.concat(' in template_lower and not any(op in template_lower for op in ['=', '<>', '>', '<', 'includes']):
        return False, f"Not a constraint: String concat without comparison"
    
    if '.toupper()' in template_lower and not any(op in template_lower for op in ['=', '<>', 'includes']):
        return False, f"Not a constraint: String operation without comparison"
    
    if '.tolower()' in template_lower and not any(op in template_lower for op in ['=', '<>', 'includes']):
        return False, f"Not a constraint: String operation without comparison"
    
    # Invalid operations on wrong types
    if '.notempty()' in template_lower or '.isempty()' in template_lower:
        # isEmpty/notEmpty only work on collections, not primitives
        # Check if being called on {attribute} rather than {collection}
        import re
        if re.search(r'\{(?!.*collection)\w*attribute\w*\}\.(notempty|isempty)\(\)', template, re.IGNORECASE):
            return False, f"Invalid operation: isEmpty/notEmpty only work on collections"
    
    # Low-value OCL operations
    if 'oclisundefined()' in template_lower or 'oclisinvalid()' in template_lower:
        return False, f"Low-value: oclIsUndefined/oclIsInvalid checks are not useful constraints"
    
    # Invalid type operations
    if 'oclastype(' in template_lower:
        # oclAsType needs to be in a meaningful expression
        if not any(op in template_lower for op in ['.{', '>=', '<=', '<>', '=', '>', '<']):
            return False, f"Not a constraint: oclAsType cast without property access or comparison"
    
    # Arithmetic operations without comparisons
    if '.min(' in template_lower or '.max(' in template_lower or '.abs()' in template_lower:
        # These return numeric values, need comparison
        # Check that there's ACTUALLY a comparison operator (not just in parameter names)
        has_comparison = any(f' {op} ' in template or f'{op} ' in template or f' {op}' in template 
                           for op in ['=', '>', '<', '>=', '<=', '<>'])
        if not has_comparison:
            return False, f"Not a constraint: Arithmetic operation (min/max/abs) without comparison"
    
    # Redundant conditionals: if X then Y else Y
    if 'if ' in template_lower and ' then ' in template_lower and ' else ' in template_lower:
        import re
        match = re.search(r'then\s+(.+?)\s+else\s+(.+?)\s+endif', template_lower)
        if match:
            then_part = match.group(1).strip()
            else_part = match.group(2).strip()
            if then_part == else_part:
                return False, f"Redundant conditional: Both branches are identical"
    
    return True, ""


# ===== PATTERN-SPECIFIC RULES =====

class SemanticValidator:
    """Validates parameters based on semantic rules."""
    
    @staticmethod
    def validate_two_attribute_pattern(pattern_id: str, params: Dict) -> Tuple[bool, str]:
        """
        Validate patterns that compare two attributes.
        
        Args:
            pattern_id: Pattern identifier
            params: Generated parameters
            
        Returns:
            (is_valid, reason)
        """
        if pattern_id == 'two_attributes_equal':
            attr1 = params.get('first_attribute', '')
            attr2 = params.get('second_attribute', '')
            
            if not is_valid_equality_pair(attr1, attr2):
                return False, f"Semantic violation: {attr1} = {attr2} doesn't make business sense"
        
        elif pattern_id == 'numeric_comparison':
            attr1 = params.get('attr1', '')
            attr2 = params.get('attr2', '')
            operator = params.get('operator', '=')
            
            # Suggest better operator
            preferred_op = get_preferred_operator_for_pair(attr1, attr2)
            if preferred_op and operator == '=' and preferred_op != '=':
                return False, f"Use '{preferred_op}' instead of '=' for {attr1} and {attr2}"
        
        return True, ""
    
    @staticmethod
    def validate_parameters(pattern_id: str, context: str, params: Dict) -> Tuple[bool, str]:
        """
        Main validation entry point.
        
        Returns:
            (is_valid, reason)
        """
        # Check two-attribute patterns
        if pattern_id in ['two_attributes_equal', 'two_attributes_not_equal', 'numeric_comparison']:
            return SemanticValidator.validate_two_attribute_pattern(pattern_id, params)
        
        # Add more pattern-specific rules here
        
        return True, ""


# ===== USAGE EXAMPLE =====

if __name__ == '__main__':
    print("=== Semantic Rules Test ===\n")
    
    # Test cases
    test_pairs = [
        ('dateFrom', 'dateTo'),
        ('mileage', 'tankLevel'),
        ('firstName', 'lastName'),
        ('price', 'totalAmount'),
        ('status', 'priority'),
    ]
    
    for attr1, attr2 in test_pairs:
        valid = is_valid_equality_pair(attr1, attr2)
        group1 = get_attribute_semantic_group(attr1)
        group2 = get_attribute_semantic_group(attr2)
        preferred_op = get_preferred_operator_for_pair(attr1, attr2)
        
        print(f"{attr1} = {attr2}")
        print(f"  Groups: {group1} vs {group2}")
        print(f"  Valid for equality? {valid}")
        print(f"  Preferred operator: {preferred_op or 'any'}")
        print()
