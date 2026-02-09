#!/usr/bin/env python3
"""
Generate 5000 OCL Training Examples and Save to JSON

Generates 5000 OCL constraint examples (100 per pattern × 50 patterns)
and saves them to a JSON file for reproducible training.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

try:
    # Try relative import first (when used as module)
    from .pattern_classifier import OCLPatternType
except ImportError:
    # Fallback to absolute import
    sys.path.insert(0, str(Path(__file__).parent))
    from pattern_classifier import OCLPatternType


def generate_training_data() -> List[Tuple[str, str]]:
    """
    Generate 5000 OCL training examples
    Returns: List of (ocl_text, pattern_value) tuples
    """
    
    # Domain vocabularies for variation
    ENTITIES = ['students', 'employees', 'courses', 'products', 'accounts', 
                'orders', 'members', 'vehicles', 'books', 'departments']
    
    PROPERTIES = ['id', 'code', 'name', 'email', 'status', 'type', 'category']
    
    NUMERIC_PROPS = ['age', 'salary', 'gpa', 'credits', 'balance', 'total', 
                     'count', 'size', 'capacity', 'duration']
    
    BOOLEAN_PROPS = ['active', 'enabled', 'verified', 'approved', 'completed']
    
    RELATION_PROPS = ['manager', 'supervisor', 'parent', 'owner', 'advisor', 
                      'department', 'course', 'account', 'license']
    
    examples = []
    
    # Pattern 1: PAIRWISE_UNIQUENESS (100 examples)
    for i in range(100):
        entity = ENTITIES[i % len(ENTITIES)]
        prop = PROPERTIES[i % len(PROPERTIES)]
        var1, var2 = ['x', 'a', 'e1', 's1'][i % 4], ['y', 'b', 'e2', 's2'][i % 4]
        examples.append((
            f"self.{entity}->forAll({var1}, {var2} | {var1} <> {var2} implies {var1}.{prop} <> {var2}.{prop})",
            OCLPatternType.PAIRWISE_UNIQUENESS.value
        ))
    
    # Pattern 2: EXACT_COUNT_SELECTION (100 examples)
    for i in range(100):
        entity = ENTITIES[i % len(ENTITIES)]
        prop = PROPERTIES[i % len(PROPERTIES)]
        op = ['=', '<=', '>=', '<', '>'][i % 5]
        val = [0, 1, 2, 3, 5][i % 5]
        examples.append((
            f"self.{entity}->select(x | x.{prop} <> null)->size() {op} {val}",
            OCLPatternType.EXACT_COUNT_SELECTION.value
        ))
    
    # Pattern 3: GLOBAL_COLLECTION (100 examples)
    for i in range(100):
        classes = ['Student', 'Employee', 'Course', 'Product', 'Account'][i % 5]
        prop = PROPERTIES[i % len(PROPERTIES)]
        examples.append((
            f"{classes}.allInstances()->collect({prop})->size() > {i % 10}",
            OCLPatternType.GLOBAL_COLLECTION.value
        ))
    
    # Pattern 4: SET_INTERSECTION (100 examples)
    for i in range(100):
        entity1 = ENTITIES[i % len(ENTITIES)]
        entity2 = ENTITIES[(i + 1) % len(ENTITIES)]
        examples.append((
            f"self.{entity1}->intersection(self.{entity2})->size() >= {i % 5}",
            OCLPatternType.SET_INTERSECTION.value
        ))
    
    # Pattern 5: SIZE_CONSTRAINT (100 examples)
    for i in range(100):
        entity = ENTITIES[i % len(ENTITIES)]
        op = ['>', '>=', '<', '<=', '='][i % 5]
        val = [0, 1, 5, 10, 20][i % 5]
        examples.append((
            f"self.{entity}->size() {op} {val}",
            OCLPatternType.SIZE_CONSTRAINT.value
        ))
    
    # Pattern 6: UNIQUENESS_CONSTRAINT (100 examples)
    for i in range(100):
        entity = ENTITIES[i % len(ENTITIES)]
        prop = PROPERTIES[i % len(PROPERTIES)]
        var = ['x', 'e', 's', 'p', 'c'][i % 5]
        examples.append((
            f"self.{entity}->isUnique({var} | {var}.{prop})",
            OCLPatternType.UNIQUENESS_CONSTRAINT.value
        ))
    
    # Pattern 7: COLLECTION_MEMBERSHIP (100 examples)
    for i in range(100):
        entity = ENTITIES[i % len(ENTITIES)]
        negation = "not " if i % 2 == 0 else ""
        examples.append((
            f"{negation}self.{entity}->includes(item)",
            OCLPatternType.COLLECTION_MEMBERSHIP.value
        ))
    
    # Pattern 8: NULL_CHECK (100 examples)
    for i in range(100):
        prop = RELATION_PROPS[i % len(RELATION_PROPS)]
        op = '<>' if i % 2 == 0 else '='
        connector = "" if i % 3 == 0 else f" and self.{RELATION_PROPS[(i+1) % len(RELATION_PROPS)]} <> null"
        examples.append((
            f"self.{prop} {op} null{connector}",
            OCLPatternType.NULL_CHECK.value
        ))
    
    # Pattern 9: NUMERIC_COMPARISON (100 examples)
    for i in range(100):
        prop = NUMERIC_PROPS[i % len(NUMERIC_PROPS)]
        if i % 3 == 0:
            examples.append((
                f"self.{prop} >= {i % 100} and self.{prop} <= {100 + i % 100}",
                OCLPatternType.NUMERIC_COMPARISON.value
            ))
        else:
            op = ['>=', '<=', '>', '<', '='][i % 5]
            examples.append((
                f"self.{prop} {op} {i % 50}",
                OCLPatternType.NUMERIC_COMPARISON.value
            ))
    
    # Pattern 10: EXACTLY_ONE (100 examples)
    for i in range(100):
        entity = ENTITIES[i % len(ENTITIES)]
        prop = PROPERTIES[i % len(PROPERTIES)]
        var = ['x', 'e', 's', 'item'][i % 4]
        value = ['true', "'admin'", "'primary'", "'active'"][i % 4]
        examples.append((
            f"self.{entity}->one({var} | {var}.{prop} = {value})",
            OCLPatternType.EXACTLY_ONE.value
        ))
    
    # Patterns 11-50: Template-based (4100 examples total, ~82 each)
    advanced_patterns = [
        (OCLPatternType.CLOSURE.value, "self.prerequisites->closure(prereq)->includes(item)"),
        (OCLPatternType.ACYCLICITY.value, "not self->closure(parent)->includes(self)"),
        (OCLPatternType.ITERATE.value, "self.grades->iterate(sum:Integer=0 | sum + value) >= 50"),
        (OCLPatternType.IMPLIES.value, "self.isStudent implies self.age >= 16"),
        (OCLPatternType.SAFE_NAVIGATION.value, "self.department->notEmpty() implies self.department.name <> null"),
        (OCLPatternType.TYPE_CHECK.value, "self.element.oclIsKindOf(Student)"),
        (OCLPatternType.SUBSET_DISJOINT.value, "self.adminUsers->includesAll(self.supervisors)"),
        (OCLPatternType.ORDERING.value, "self.grades->sortedBy(score)->first().score >= 90"),
        (OCLPatternType.CONTRACTUAL.value, "self.balance@pre + self.deposit = self.balance"),
        (OCLPatternType.SELECT_REJECT.value, "self.users->select(u | u.age > 18)->size()"),
        (OCLPatternType.COLLECT_FLATTEN.value, "self.students->collect(courses)->flatten()->size()"),
        (OCLPatternType.ANY_OPERATION.value, "self.items->any(i | i.price > 100)"),
        (OCLPatternType.FOR_ALL_NESTED.value, "self.students->forAll(s | s.courses->notEmpty())"),
        (OCLPatternType.EXISTS_NESTED.value, "self.students->exists(s | s.gpa = 4.0)"),
        (OCLPatternType.COLLECT_NESTED.value, "self.numbers->collect(n | n * 2)->sum() = 100"),
        (OCLPatternType.AS_SET_AS_BAG.value, "Bag{1,2,2,3}->asSet()->size() = 3"),
        (OCLPatternType.SUM_PRODUCT.value, "self.prices->sum() > 1000"),
        (OCLPatternType.STRING_CONCAT.value, "self.firstName + ' ' + self.lastName"),
        (OCLPatternType.STRING_OPERATIONS.value, "self.email.toUpper() = 'USER@EXAMPLE.COM'"),
        (OCLPatternType.STRING_COMPARISON.value, "self.email.matches('.*@.*\\\\..*')"),
        (OCLPatternType.STRING_PATTERN.value, "self.text.matches('[0-9]+')"),
        (OCLPatternType.ARITHMETIC_EXPRESSION.value, "self.x + self.y = 10"),
        (OCLPatternType.DIV_MOD_OPERATIONS.value, "self.value mod 2 = 0"),
        (OCLPatternType.ABS_MIN_MAX.value, "self.value.abs() <= 100"),
        (OCLPatternType.BOOLEAN_OPERATIONS.value, "self.isActive and self.isVerified"),
        (OCLPatternType.IF_THEN_ELSE.value, "if self.age >= 18 then 'Adult' else 'Minor' endif"),
        (OCLPatternType.TUPLE_LITERAL.value, "Tuple{first=1, second=2}"),
        (OCLPatternType.LET_EXPRESSION.value, "let x = self.value in x * 2"),
        (OCLPatternType.LET_NESTED.value, "let sum = self->iterate(s = 0 | s + value) in sum > 100"),
        (OCLPatternType.UNION_INTERSECTION.value, "Set{1,2,3}->union(Set{3,4,5})->size() = 5"),
        (OCLPatternType.SYMMETRIC_DIFFERENCE.value, "Set{1,2,3}->symmetricDifference(Set{2,3,4})->size() = 2"),
        (OCLPatternType.INCLUDING_EXCLUDING.value, "self.items->including(newItem)->size() > self.items->size()"),
        (OCLPatternType.FLATTEN_OPERATION.value, "self.nested->flatten()->notEmpty()"),
        (OCLPatternType.NAVIGATION_CHAIN.value, "self.department.manager.salary > 50000"),
        (OCLPatternType.OPTIONAL_NAVIGATION.value, "self.supervisor?->isEmpty() or self.supervisor.level > 0"),
        (OCLPatternType.COLLECTION_NAVIGATION.value, "self.students->collect(address)->collect(city)->asSet()->size()"),
        (OCLPatternType.SHORTHAND_NOTATION.value, "self.location = self.homeAddress"),
        (OCLPatternType.OCL_IS_UNDEFINED.value, "self.value.oclIsUndefined()"),
        (OCLPatternType.OCL_IS_INVALID.value, "self.value.oclIsInvalid()"),
        (OCLPatternType.OCL_AS_TYPE.value, "self.element.oclAsType(Student)"),
    ]
    
    # Generate 82-83 examples per advanced pattern
    examples_per_pattern = 4100 // 41
    remainder = 4100 % 41
    
    for pattern_idx, (pattern_type, template) in enumerate(advanced_patterns):
        count = examples_per_pattern + (1 if pattern_idx < remainder else 0)
        for i in range(count):
            # Vary the template slightly
            variant = template.replace("self", f"obj{i % 3}")
            variant = variant.replace("item", ENTITIES[i % len(ENTITIES)])
            examples.append((variant, pattern_type))
    
    return examples


def save_training_data(output_file: str = "ocl_training_data.json"):
    """
    Generate and save 5000 OCL training examples to JSON file
    
    Args:
        output_file: Path to save the JSON file
    """
    print("\n" + "="*80)
    print(" Generating 5000 OCL Training Examples")
    print("="*80)
    
    print("\n🔄 Generating examples...")
    examples = generate_training_data()
    
    print(f" Generated {len(examples)} examples")
    
    # Convert to JSON-serializable format
    data = {
        "metadata": {
            "total_examples": len(examples),
            "total_patterns": 50,
            "examples_per_pattern": 100,
            "source": "generate_training_data.py"
        },
        "examples": [
            {
                "id": idx + 1,
                "ocl_text": ocl_text,
                "pattern": pattern,
            }
            for idx, (ocl_text, pattern) in enumerate(examples)
        ]
    }
    
    # Verify distribution
    print("\n Pattern distribution:")
    pattern_counts = {}
    for _, pattern in examples:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    for pattern in sorted(pattern_counts.keys()):
        count = pattern_counts[pattern]
        print(f"   {pattern}: {count}")
    
    # Save to JSON
    output_path = Path(output_file)
    print(f"\n💾 Saving to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f" Saved {len(examples)} examples to {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*80)
    print("🎉 TRAINING DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"File: {output_path}")
    print(f"Examples: {len(examples)}")
    print(f"Patterns: {len(pattern_counts)}")
    print("="*80 + "\n")
    
    return output_path


if __name__ == "__main__":
    output_file = "ocl_training_data.json"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    save_training_data(output_file)
