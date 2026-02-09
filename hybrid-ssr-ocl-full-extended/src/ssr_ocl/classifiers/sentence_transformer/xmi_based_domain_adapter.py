#!/usr/bin/env python3
"""
XMI-Based Domain Adaptation
Extracts domain vocabulary from XMI models and generates domain-specific OCL examples
Works for ANY domain (CarRental, BookRental, CarWorkshop, etc.)
"""

import re
import json
from typing import List, Tuple, Dict, Set
from pathlib import Path
from xml.etree import ElementTree as ET
from .classifier import OCLPatternType


class XMIModelExtractor:
    """Extract classes, attributes, and associations from XMI models"""
    
    def __init__(self, xmi_file: str):
        """
        Initialize extractor with XMI file
        
        Args:
            xmi_file: Path to XMI model file
        """
        self.xmi_file = xmi_file
        self.classes = {}  # {class_name: {attributes, associations}}
        self.attributes = {}  # {class_name: [attr_names]}
        self.associations = {}  # {class_name: [related_classes]}
        self.parse_xmi()
    
    def parse_xmi(self):
        """Parse XMI file and extract model information (Ecore-compatible)"""
        try:
            # --- FIX 1: Register Namespaces ---
            # We must tell the parser what 'ecore', 'xsi', etc. mean
            namespaces = {
                'ecore': 'http://www.eclipse.org/emf/2002/Ecore',
                'xmi': 'http://www.omg.org/XMI',
                'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
            }
            for prefix, uri in namespaces.items():
                ET.register_namespace(prefix, uri)

            tree = ET.parse(self.xmi_file)
            root = tree.getroot()

            # --- FIX 2: Find Ecore Classes ---
            # We look for <eClassifiers> tags that have an xsi:type of "ecore:EClass"
            # This is the correct way to find classes in Ecore-based XMI files
            for elem in root.findall(".//eClassifiers[@{http://www.w3.org/2001/XMLSchema-instance}type='ecore:EClass']"):
                class_name = elem.get('name')
                if class_name:
                    self.classes[class_name] = {
                        'attributes': [],
                        'associations': []
                    }

                    # --- FIX 3: Find Ecore Attributes ---
                    # We look for <eStructuralFeatures> tags inside the class
                    # that have an xsi:type of "ecore:EAttribute"
                    for attr in elem.findall("./eStructuralFeatures[@{http://www.w3.org/2001/XMLSchema-instance}type='ecore:EAttribute']"):
                        attr_name = attr.get('name')
                        attr_type = attr.get('eType', 'String')  # Get type info
                        
                        if attr_name:
                            self.classes[class_name]['attributes'].append({
                                'name': attr_name,
                                'type': attr_type
                            })
                            if class_name not in self.attributes:
                                self.attributes[class_name] = []
                            self.attributes[class_name].append(attr_name)
                    
                    # --- FIX 4: Find Ecore References (Associations) ---
                    # OCL constraints can navigate through EReferences too!
                    for ref in elem.findall("./eStructuralFeatures[@{http://www.w3.org/2001/XMLSchema-instance}type='ecore:EReference']"):
                        ref_name = ref.get('name')
                        ref_type = ref.get('eType', '')
                        
                        if ref_name:
                            self.classes[class_name]['associations'].append({
                                'name': ref_name,
                                'type': ref_type
                            })
                            # Add references to attributes list so they're available for validation
                            if class_name not in self.attributes:
                                self.attributes[class_name] = []
                            self.attributes[class_name].append(ref_name)
                            
                            # Also track in associations dict
                            if class_name not in self.associations:
                                self.associations[class_name] = []
                            self.associations[class_name].append(ref_name)
            
            print(f" Extracted {len(self.classes)} classes from {self.xmi_file}")
            if not self.classes:
                print(f"  No classes found. Check XMI structure and parser queries.")
            
            for cls, info in self.classes.items():
                attr_count = len(info['attributes'])
                assoc_count = len(info['associations'])
                print(f"   - {cls}: {attr_count} attributes, {assoc_count} references")
                
        except Exception as e:
            print(f"  Could not parse XMI: {e}")
            print(f"   Will use fallback extraction from model structure")
    
    def get_classes(self) -> List[str]:
        """Get list of class names"""
        return list(self.classes.keys()) if self.classes else self._fallback_classes()
    
    def get_attributes(self, class_name: str) -> List[str]:
        """Get attributes for a class"""
        if class_name in self.attributes:
            return self.attributes[class_name]
        return self._fallback_attributes(class_name)
    
    def _fallback_classes(self) -> List[str]:
        """Fallback class names if XMI parsing fails"""
        return ['Entity', 'Object', 'Item', 'Record', 'Relation']
    
    def _fallback_attributes(self, class_name: str) -> List[str]:
        """Fallback attributes if XMI parsing fails"""
        return ['id', 'name', 'code', 'status', 'date', 'amount']


class GenericDomainDataGenerator:
    """
    Generate domain-specific OCL examples using XMI model
    Works for ANY domain without hardcoding
    """
    
    def __init__(self, xmi_file: str, examples_per_pattern: int = 10):
        """
        Initialize generator
        
        Args:
            xmi_file: Path to XMI model file
            examples_per_pattern: Number of examples per OCL pattern
        """
        self.xmi_file = xmi_file
        self.extractor = XMIModelExtractor(xmi_file)
        self.examples_per_pattern = examples_per_pattern
        self.examples = []
    
    def generate_domain_data(self) -> List[Tuple[str, str]]:
        """
        Generate domain-specific OCL examples for all patterns
        
        Returns:
            List of (ocl_text, pattern_name) tuples
        """
        self.examples = []
        
        # Get domain vocabulary from XMI
        classes = self.extractor.get_classes()
        
        if not classes:
            print(" No classes extracted from XMI model")
            return []
        
        print(f"\n🔄 Generating OCL examples using {len(classes)} domain classes...")
        
        # For each class, generate examples for key patterns
        for class_idx, class_name in enumerate(classes):
            attributes = self.extractor.get_attributes(class_name)
            
            # Pluralize class name for collections
            collection_name = self._pluralize(class_name)
            
            # Pattern 1: PAIRWISE_UNIQUENESS
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var1, var2 = f"x{i%3}", f"y{i%3}"
                    self.examples.append((
                        f"self.{collection_name}->forAll({var1}, {var2} | {var1} <> {var2} implies {var1}.{attr} <> {var2}.{attr})",
                        OCLPatternType.PAIRWISE_UNIQUENESS.value
                    ))
            
            # Pattern 2: SIZE_CONSTRAINT
            for i in range(self.examples_per_pattern):
                op = ['>', '>=', '<', '<=', '='][i % 5]
                val = [0, 1, 5, 10, 20][i % 5]
                self.examples.append((
                    f"self.{collection_name}->size() {op} {val}",
                    OCLPatternType.SIZE_CONSTRAINT.value
                ))
            
            # Pattern 3: UNIQUENESS_CONSTRAINT
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var = f"e{i%4}"
                    self.examples.append((
                        f"self.{collection_name}->isUnique({var} | {var}.{attr})",
                        OCLPatternType.UNIQUENESS_CONSTRAINT.value
                    ))
            
            # Pattern 4: NUMERIC_COMPARISON (domain-generic)
            # NOTE: Single comparisons only - conjunctions belong to BOOLEAN_OPERATIONS
            # ENHANCED: More attribute-to-attribute comparisons for better classification
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    op = ['>=', '<=', '>', '<', '=', '<>'][i % 6]
                    val = i % 50
                    variants = [
                        f"self.{attr} {op} {val}",
                        f"self.{attr} {op} self.{attributes[(i+1) % len(attributes)]}",
                        f"self.{attr} + {val} {op} self.{attributes[(i+1) % len(attributes)]}",
                        f"self.{attr} {op} {val}",  # Removed 'and' - moved to BOOLEAN_OPERATIONS
                    ]
                    self.examples.append((
                        variants[i % len(variants)],
                        OCLPatternType.NUMERIC_COMPARISON.value
                    ))
            
            # Pattern 4b: NUMERIC_COMPARISON - Additional attr vs attr examples (BOOST)
            for i in range(self.examples_per_pattern * 2):  # 2x more examples for attr-vs-attr
                if len(attributes) >= 2:
                    attr1 = attributes[i % len(attributes)]
                    attr2 = attributes[(i+1) % len(attributes)]
                    ops = ['>', '>=', '<', '<=', '=', '<>']
                    op = ops[i % len(ops)]
                    attr_vs_attr_variants = [
                        f"self.{attr1} {op} self.{attr2}",  # Direct comparison
                        f"self.{attr2} {op} self.{attr1}",  # Reversed
                        f"self.{attr1} - self.{attr2} {op} 0",  # With arithmetic
                        f"self.{attr1} * 2 {op} self.{attr2}",  # With multiplication
                    ]
                    self.examples.append((
                        attr_vs_attr_variants[i % len(attr_vs_attr_variants)],
                        OCLPatternType.NUMERIC_COMPARISON.value
                    ))
            
            # Pattern 4c: NUMERIC_COMPARISON - Ultra-simple direct comparisons (MEGA BOOST)
            # Focus: Simple attribute vs attribute with NO arithmetic - exactly like DatesOrder
            for i in range(self.examples_per_pattern * 4):  # 4x more for pure direct comparisons!
                if len(attributes) >= 2:
                    attr1 = attributes[i % len(attributes)]
                    attr2 = attributes[(i+1) % len(attributes)]
                    # Only use simple comparison operators - most common in constraints
                    simple_ops = ['>', '>=', '<', '<=']
                    op = simple_ops[i % len(simple_ops)]
                    # Pure, simple, direct comparison - NO arithmetic!
                    self.examples.append((
                        f"self.{attr1} {op} self.{attr2}",
                        OCLPatternType.NUMERIC_COMPARISON.value
                    ))
            
            # Pattern 5: NULL_CHECK
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    op = '<>' if i % 2 == 0 else '='
                    self.examples.append((
                        f"self.{attr} {op} null",
                        OCLPatternType.NULL_CHECK.value
                    ))
            
            # Pattern 6: COLLECTION_MEMBERSHIP
            for i in range(self.examples_per_pattern):
                negation = "not " if i % 2 == 0 else ""
                item = f"item{i%3}"
                self.examples.append((
                    f"{negation}self.{collection_name}->includes({item})",
                    OCLPatternType.COLLECTION_MEMBERSHIP.value
                ))
            
            # Pattern 7: FOR_ALL_NESTED
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var = f"r{i%4}" if i % 2 == 0 else f"item{i%3}"
                    op = ['>',  '>=', '<', '<=', '=', '<>'][i % 6]
                    variants = [
                        f"self.{collection_name}->forAll({var} | {var}.{attr} <> null)",
                        f"self.{collection_name}->forAll({var} | {var}.{attr} {op} {i % 30})",
                        f"self.{collection_name}->forAll({var} | {var}.{attr} {op} self.{attr})",
                    ]
                    self.examples.append((
                        variants[i % len(variants)],
                        OCLPatternType.FOR_ALL_NESTED.value
                    ))
            
            # Pattern 8: SELECT_REJECT
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var = f"x{i%3}"
                    op = 'select' if i % 2 == 0 else 'reject'
                    self.examples.append((
                        f"self.{collection_name}->{op}({var} | {var}.{attr} <> null)->size() > 0",
                        OCLPatternType.SELECT_REJECT.value
                    ))
            
            # Pattern 9: EXISTS_NESTED
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var = f"e{i%4}"
                    self.examples.append((
                        f"self.{collection_name}->exists({var} | {var}.{attr} <> null)",
                        OCLPatternType.EXISTS_NESTED.value
                    ))
            
            # Pattern 10: EXACT_COUNT_SELECTION
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var = f"x{i%3}"
                    num = [1, 2, 3][i % 3]
                    self.examples.append((
                        f"self.{collection_name}->select({var} | {var}.{attr} = {num})->size() = {num}",
                        OCLPatternType.EXACT_COUNT_SELECTION.value
                    ))
            
            # Pattern 11: GLOBAL_COLLECTION
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"{class_name}.allInstances()->size() > 0",
                    OCLPatternType.GLOBAL_COLLECTION.value
                ))
            
            # Pattern 12: SET_INTERSECTION
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"self.{collection_name}->intersection(self.other{collection_name})->notEmpty()",
                    OCLPatternType.SET_INTERSECTION.value
                ))
            
            # Pattern 13: EXACTLY_ONE
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var = f"x{i%3}"
                    self.examples.append((
                        f"self.{collection_name}->one({var} | {var}.{attr} = true)",
                        OCLPatternType.EXACTLY_ONE.value
                    ))
            
            # Pattern 14: CLOSURE_TRANSITIVE
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"self->closure(parent)->includes(root)",
                    OCLPatternType.CLOSURE.value
                ))
            
            # Pattern 15: ACYCLICITY
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"not self->closure(parent)->includes(self)",
                    OCLPatternType.ACYCLICITY.value
                ))
            
            # Pattern 16: AGGREGATION_ITERATE
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"self.{collection_name}->iterate(x; acc : Integer = 0 | acc + x.{attr})",
                        OCLPatternType.ITERATE.value
                    ))
            
            # Pattern 17: BOOLEAN_GUARD_IMPLIES
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    variants = [
                        f"self.{attr1} <> null implies self.{attr2} > 0",
                        f"self.{collection_name}->notEmpty() implies self.{attr1} >= {i % 50}",
                        f"self.{attr1} > 0 implies self.{attr2} <> null",
                    ]
                    self.examples.append((
                        variants[i % len(variants)],
                        OCLPatternType.IMPLIES.value
                    ))
            
            # Pattern 17b: BOOLEAN_GUARD_IMPLIES - Enhanced implies patterns (BOOST)
            for i in range(self.examples_per_pattern * 2):  # 2x more examples
                if len(attributes) >= 3:
                    attr1, attr2, attr3 = (
                        attributes[i % len(attributes)], 
                        attributes[(i+1) % len(attributes)], 
                        attributes[(i+2) % len(attributes)]
                    )
                    val = i % 100
                    implies_variants = [
                        # Guard-based implications
                        f"self.{attr1} <> null implies self.{attr2} >= {val}",
                        f"self.{attr1} > {val} implies self.{attr2} <> null",
                        # Collection guard implications
                        f"self.{collection_name}->notEmpty() implies self.{attr1} >= self.{attr2}",
                        f"self.{collection_name}->isEmpty() implies self.{attr1} = 0",
                        # Complex guard implications
                        f"self.{attr1} and self.{attr2} implies self.{attr3} > {val}",
                        f"self.{attr1} >= {val} implies self.{attr2} <= self.{attr3}",
                        # Nested implications
                        f"self.{collection_name}->forAll(x | x.{attr1} > 0) implies self.{attr2} >= {val}",
                    ]
                    self.examples.append((
                        implies_variants[i % len(implies_variants)],
                        OCLPatternType.IMPLIES.value
                    ))
            
            # Pattern 18: SAFE_NAVIGATION
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"self.{attr}?.toString()",
                        OCLPatternType.SAFE_NAVIGATION.value
                    ))
            
            # Pattern 19: TYPE_CHECK_CASTING
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"self.oclIsKindOf({class_name})",
                    OCLPatternType.TYPE_CHECK.value
                ))
            
            # Pattern 20: SUBSET_DISJOINTNESS
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"self.{collection_name}->intersection(self.other{collection_name})->isEmpty()",
                    OCLPatternType.SUBSET_DISJOINT.value
                ))
            
            # Pattern 21: ORDERING_RANKING
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"self.{collection_name}->sortedBy({attr})->first()",
                        OCLPatternType.ORDERING.value
                    ))
            
            # Pattern 22: CONTRACTUAL_TEMPORAL
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[0], attributes[1]
                    self.examples.append((
                        f"self.{attr1}->notEmpty() implies self.{attr2} = self.{attr1}",
                        OCLPatternType.CONTRACTUAL.value
                    ))
            
            # Pattern 23: COLLECT_FLATTEN
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var = f"x{i%3}"
                    self.examples.append((
                        f"self.{collection_name}->collect({var} | {var}.{attr})->flatten()",
                        OCLPatternType.COLLECT_FLATTEN.value
                    ))
            
            # Pattern 24: ANY_OPERATION
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    var = f"x{i%3}"
                    self.examples.append((
                        f"self.{collection_name}->any({var} | {var}.{attr} = true)",
                        OCLPatternType.ANY_OPERATION.value
                    ))
            
            # Pattern 25: COLLECT_NESTED
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    self.examples.append((
                        f"self.{collection_name}->collect({attr1})->collect({attr2})",
                        OCLPatternType.COLLECT_NESTED.value
                    ))
            
            # Pattern 26: AS_SET_AS_BAG
            for i in range(self.examples_per_pattern):
                op = 'asSet' if i % 2 == 0 else 'asBag'
                self.examples.append((
                    f"self.{collection_name}->{op}()->size()",
                    OCLPatternType.AS_SET_AS_BAG.value
                ))
            
            # Pattern 27: SUM_PRODUCT
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"self.{collection_name}.{attr}->sum() > 0",
                        OCLPatternType.SUM_PRODUCT.value
                    ))
            
            # Pattern 28: STRING_CONCAT
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[0], attributes[1]
                    self.examples.append((
                        f"self.{attr1}.concat(self.{attr2})",
                        OCLPatternType.STRING_CONCAT.value
                    ))
            
            # Pattern 29: STRING_OPERATIONS
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    ops = ['toUpper()', 'toLower()', 'size()', 'substring(1,5)']
                    self.examples.append((
                        f"self.{attr}.{ops[i % len(ops)]}",
                        OCLPatternType.STRING_OPERATIONS.value
                    ))
            
            # Pattern 30: STRING_COMPARISON
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    self.examples.append((
                        f"self.{attr1} = self.{attr2}",
                        OCLPatternType.STRING_COMPARISON.value
                    ))
            
            # Pattern 31: STRING_PATTERN
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"self.{attr}.matches('[a-zA-Z]+')",
                        OCLPatternType.STRING_PATTERN.value
                    ))
            
            # Pattern 32: ARITHMETIC_EXPRESSION
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    ops = ['+', '-', '*', '/']
                    self.examples.append((
                        f"self.{attr1} {ops[i % len(ops)]} self.{attr2} > 0",
                        OCLPatternType.ARITHMETIC_EXPRESSION.value
                    ))
            
            # Pattern 33: DIV_MOD_OPERATIONS
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    op = 'div' if i % 2 == 0 else 'mod'
                    self.examples.append((
                        f"self.{attr} {op} 10",
                        OCLPatternType.DIV_MOD_OPERATIONS.value
                    ))
            
            # Pattern 34: ABS_MIN_MAX
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    ops = ['abs()', f'max(self.{attr2})', f'min(self.{attr2})']
                    self.examples.append((
                        f"self.{attr1}.{ops[i % len(ops)]}",
                        OCLPatternType.ABS_MIN_MAX.value
                    ))
            
            # Pattern 35: BOOLEAN_OPERATIONS
            # IMPORTANT: All conjunctions/disjunctions belong here, not NUMERIC_COMPARISON
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    ops = ['and', 'or', 'xor', 'and', 'or', 'and']  # More 'and' for range constraints
                    val1, val2 = i % 100, (i+7) % 100
                    variants = [
                        f"self.{attr1} >= {val1} {ops[i % len(ops)]} self.{attr2} <= {val2}",
                        f"self.{attr1} > 0 {ops[i % len(ops)]} self.{attr1} < 100",  # Range pattern!
                        f"self.{attr1} <> null {ops[i % len(ops)]} self.{attr2} > {val1}",
                        f"self.{attr1} >= {val1} and self.{attr1} <= {val2}",  # Explicit range
                        f"self.{attr1} {ops[i % len(ops)]} self.{attr2}",  # Boolean attrs
                    ]
                    self.examples.append((
                        variants[i % len(variants)],
                        OCLPatternType.BOOLEAN_OPERATIONS.value
                    ))
            
            # Pattern 35b: BOOLEAN_OPERATIONS - Complex conjunctions (BOOST)
            for i in range(self.examples_per_pattern * 2):  # 2x more examples for complex patterns
                if len(attributes) >= 3:
                    attr1, attr2, attr3 = (
                        attributes[i % len(attributes)], 
                        attributes[(i+1) % len(attributes)], 
                        attributes[(i+2) % len(attributes)]
                    )
                    val1, val2 = i % 100, (i+20) % 100
                    complex_variants = [
                        # Range constraints with different patterns
                        f"self.{attr1} >= {val1} and self.{attr1} <= {val2}",
                        f"self.{attr2} > 0 and self.{attr2} < 1000",
                        # Multi-condition checks
                        f"self.{attr1} > self.{attr2} and self.{attr2} > {val1}",
                        f"self.{attr1} <> null and self.{attr2} >= self.{attr3}",
                        # Complex guards with navigation
                        f"self.{collection_name}->isEmpty() or self.{attr1} > {val1}",
                        f"self.{attr1} > self.{attr2} and (self.{collection_name}->isEmpty() or self.{attr3} = {val1})",
                        # Multiple conjunctions
                        f"self.{attr1} >= {val1} and self.{attr2} <= {val2} and self.{attr3} <> null",
                        f"self.{attr1} or self.{attr2} or self.{attr3}",
                        # Mixed operators
                        f"self.{attr1} and (self.{attr2} > {val1} or self.{attr3} < {val2})",
                        f"(self.{attr1} > {val1} and self.{attr1} < {val2}) or self.{attr2} = 0",
                    ]
                    self.examples.append((
                        complex_variants[i % len(complex_variants)],
                        OCLPatternType.BOOLEAN_OPERATIONS.value
                    ))
            
            # Pattern 36: IF_THEN_ELSE
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 3:
                    attr1, attr2, attr3 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)], attributes[(i+2) % len(attributes)]
                    self.examples.append((
                        f"if self.{attr1} > 0 then self.{attr2} else self.{attr3} endif",
                        OCLPatternType.IF_THEN_ELSE.value
                    ))
            
            # Pattern 37: TUPLE_LITERAL
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    self.examples.append((
                        f"Tuple{{a = self.{attr1}, b = self.{attr2}}}",
                        OCLPatternType.TUPLE_LITERAL.value
                    ))
            
            # Pattern 38: LET_EXPRESSION
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"let temp : Integer = self.{attr} in temp > 0",
                        OCLPatternType.LET_EXPRESSION.value
                    ))
            
            # Pattern 39: LET_NESTED
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    self.examples.append((
                        f"let x : Integer = self.{attr1} in let y : Integer = self.{attr2} in x + y",
                        OCLPatternType.LET_NESTED.value
                    ))
            
            # Pattern 40: UNION_INTERSECTION
            for i in range(self.examples_per_pattern):
                op = 'union' if i % 2 == 0 else 'intersection'
                self.examples.append((
                    f"self.{collection_name}->{op}(self.other{collection_name})",
                    OCLPatternType.UNION_INTERSECTION.value
                ))
            
            # Pattern 41: SYMMETRIC_DIFFERENCE
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"self.{collection_name}->symmetricDifference(self.other{collection_name})",
                    OCLPatternType.SYMMETRIC_DIFFERENCE.value
                ))
            
            # Pattern 42: INCLUDING_EXCLUDING
            for i in range(self.examples_per_pattern):
                op = 'including' if i % 2 == 0 else 'excluding'
                self.examples.append((
                    f"self.{collection_name}->{op}(newItem)",
                    OCLPatternType.INCLUDING_EXCLUDING.value
                ))
            
            # Pattern 43: FLATTEN_OPERATION
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"self.{collection_name}->flatten()->size()",
                    OCLPatternType.FLATTEN_OPERATION.value
                ))
            
            # Pattern 44: NAVIGATION_CHAIN
            for i in range(self.examples_per_pattern):
                if len(attributes) >= 2:
                    attr1, attr2 = attributes[i % len(attributes)], attributes[(i+1) % len(attributes)]
                    self.examples.append((
                        f"self.{attr1}.{attr2} <> null",
                        OCLPatternType.NAVIGATION_CHAIN.value
                    ))
            
            # Pattern 44b: NAVIGATION_CHAIN - Enhanced navigation patterns (BOOST)
            for i in range(self.examples_per_pattern * 3):  # 3x more examples for navigation
                if len(attributes) >= 3:
                    attr1, attr2, attr3 = (
                        attributes[i % len(attributes)], 
                        attributes[(i+1) % len(attributes)], 
                        attributes[(i+2) % len(attributes)]
                    )
                    val = i % 100
                    ops = ['>', '>=', '<', '<=', '=', '<>']
                    op = ops[i % len(ops)]
                    navigation_variants = [
                        # Two-level navigation
                        f"self.{attr1}.{attr2} {op} self.{attr3}",
                        f"self.{attr1}.{attr2} {op} {val}",
                        f"self.{attr1}.{attr2} <> null",
                        # Three-level navigation
                        f"self.{attr1}.{attr2}.{attr3} {op} {val}",
                        f"self.{attr1}.{attr2}.{attr3} <> null",
                        # Navigation with comparisons to self attributes
                        f"self.{attr1}.{attr2} >= self.{attr3}",
                        f"self.{attr1}.{attr2} > self.{attr3}",
                        f"self.{attr1}.{attr2} <= self.{attr3}",
                        # Navigation with operations
                        f"self.{attr1}.{attr2} + {val} > self.{attr3}",
                        f"self.{attr1}.{attr2} * 2 >= self.{attr3}",
                    ]
                    self.examples.append((
                        navigation_variants[i % len(navigation_variants)],
                        OCLPatternType.NAVIGATION_CHAIN.value
                    ))
            
            # Pattern 45: OPTIONAL_NAVIGATION
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"self.{attr}->isEmpty() or self.{attr} > 0",
                        OCLPatternType.OPTIONAL_NAVIGATION.value
                    ))
            
            # Pattern 46: COLLECTION_NAVIGATION
            for i in range(self.examples_per_pattern):
                ops = ['first()', 'last()', 'at(1)']
                self.examples.append((
                    f"self.{collection_name}->{ops[i % len(ops)]}",
                    OCLPatternType.COLLECTION_NAVIGATION.value
                ))
            
            # Pattern 47: SHORTHAND_NOTATION
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"self.{collection_name}.{attr}->sum()",
                        OCLPatternType.SHORTHAND_NOTATION.value
                    ))
            
            # Pattern 48: OCL_IS_UNDEFINED
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"self.{attr}.oclIsUndefined()",
                        OCLPatternType.OCL_IS_UNDEFINED.value
                    ))
            
            # Pattern 49: OCL_IS_INVALID
            for i in range(self.examples_per_pattern):
                if attributes:
                    attr = attributes[i % len(attributes)]
                    self.examples.append((
                        f"not self.{attr}.oclIsInvalid()",
                        OCLPatternType.OCL_IS_INVALID.value
                    ))
            
            # Pattern 50: OCL_AS_TYPE
            for i in range(self.examples_per_pattern):
                self.examples.append((
                    f"self.oclAsType({class_name})",
                    OCLPatternType.OCL_AS_TYPE.value
                ))
            
            # Only generate for first 3-4 classes to avoid too many examples
            if class_idx >= 3:
                break
        
        print(f" Generated {len(self.examples)} domain-specific OCL examples")
        return self.examples
    
    def _pluralize(self, word: str) -> str:
        """Simple pluralization"""
        if word.endswith('y'):
            return word[:-1] + 'ies'
        elif word.endswith('s'):
            return word + 'es'
        else:
            return word + 's'
    
    def save_to_json(self, output_file: str) -> str:
        """Save generated examples to JSON"""
        data = {
            "metadata": {
                "source": "XMI model extraction",
                "total_examples": len(self.examples),
                "patterns": len(set(ex[1] for ex in self.examples)),
                "xmi_file": self.xmi_file
            },
            "examples": [
                {
                    "id": idx + 1,
                    "ocl_text": ocl_text,
                    "pattern": pattern
                }
                for idx, (ocl_text, pattern) in enumerate(self.examples)
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f" Saved {len(self.examples)} domain examples to {output_file}")
        return output_file


class DomainAdaptationTrainer:
    """Train classifier with domain adaptation"""
    
    @staticmethod
    def merge_and_retrain(base_classifier, 
                         generic_data_file: str,
                         domain_data_file: str,
                         model_output_dir: str) -> float:
        """
        Merge domain data with generic data and retrain classifier
        
        Args:
            base_classifier: Pre-trained classifier
            generic_data_file: Path to generic 5000 examples
            domain_data_file: Path to domain-specific examples
            model_output_dir: Where to save adapted model
        
        Returns:
            New average confidence on domain data
        """
        # Load data
        with open(generic_data_file, 'r') as f:
            generic = json.load(f)
        
        with open(domain_data_file, 'r') as f:
            domain = json.load(f)
        
        # Merge
        combined_examples = []
        for item in generic['examples']:
            combined_examples.append((item['ocl_text'], item['pattern']))
        
        for item in domain['examples']:
            combined_examples.append((item['ocl_text'], item['pattern']))
        
        print(f"\n Dataset composition:")
        print(f"   Generic: {len(generic['examples'])}")
        print(f"   Domain-specific: {len(domain['examples'])}")
        print(f"   Total: {len(combined_examples)}")
        
        # Retrain
        print(f"\n🔄 Retraining classifier on combined dataset...")
        base_classifier.model_dir = model_output_dir
        base_classifier.train(combined_examples)
        
        return 0.0  # Will be computed after evaluation


if __name__ == "__main__":
    import sys
    sys.path.insert(0, './src')
    
    print("🔄 XMI-Based Domain Adaptation\n")
    print("=" * 80)
    
    # Example: CarRental
    xmi_file = 'examples/carrentalsystem/model.xmi'
    
    if Path(xmi_file).exists():
        print(f"📁 Found XMI model: {xmi_file}")
        
        # Generate domain data
        print("\n1️⃣  Extracting domain vocabulary from XMI...")
        generator = GenericDomainDataGenerator(xmi_file, examples_per_pattern=10)
        domain_examples = generator.generate_domain_data()
        
        # Save
        print("\n2️⃣  Saving domain data...")
        domain_file = generator.save_to_json('ocl_domain_adapted.json')
        
        print("\n" + "=" * 80)
        print(" Domain adaptation data ready!")
        print("\nNext steps:")
        print("1. Load generic classifier")
        print("2. Call: merge_and_retrain(classifier, 'ocl_training_data.json', 'ocl_domain_adapted.json', model_dir)")
        print("3. Test confidence on domain constraints")
        
    else:
        print(f" XMI file not found: {xmi_file}")
