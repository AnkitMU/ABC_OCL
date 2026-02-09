#!/usr/bin/env python3
"""
Model Consistency Checker
Validates that XMI model and OCL constraints belong to the same model/class diagram
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of model consistency validation"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    stats: Dict[str, any]


class OCLConstraintParser:
    """Parse OCL constraints to extract referenced elements"""
    
    @staticmethod
    def extract_context_classes(ocl_file: str) -> Set[str]:
        """Extract all context classes from OCL file"""
        context_classes = set()
        
        with open(ocl_file, 'r') as f:
            content = f.read()
        
        # Find all "context ClassName" declarations
        pattern = r'context\s+(\w+)'
        matches = re.findall(pattern, content)
        context_classes.update(matches)
        
        return context_classes
    
    @staticmethod
    def extract_referenced_classes(ocl_file: str) -> Set[str]:
        """Extract all classes referenced in OCL constraints (e.g., self.customer.license)"""
        referenced = set()
        
        with open(ocl_file, 'r') as f:
            content = f.read()
        
        # Pattern: self.relationName (likely a class reference)
        # We'll look for navigation paths like self.customer, self.vehicle, etc.
        pattern = r'self\.(\w+)'
        matches = re.findall(pattern, content)
        
        # Capitalize first letter (common class naming convention)
        for match in matches:
            # Convert to PascalCase if it looks like a class
            if match and match[0].islower():
                referenced.add(match.capitalize())
            else:
                referenced.add(match)
        
        return referenced
    
    @staticmethod
    def extract_properties(ocl_file: str) -> Set[str]:
        """Extract all properties/attributes referenced in OCL"""
        properties = set()
        
        with open(ocl_file, 'r') as f:
            content = f.read()
        
        # Pattern: .propertyName (attributes/associations)
        pattern = r'\.(\w+)'
        matches = re.findall(pattern, content)
        properties.update(matches)
        
        return properties
    
    @staticmethod
    def parse_full_constraint_info(ocl_file: str) -> List[Dict]:
        """Parse complete constraint information"""
        constraints = []
        
        with open(ocl_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('context '):
                context_class = line.split('context ')[1].strip()
                i += 1
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    if line.startswith('inv '):
                        inv_name = line.split('inv ')[1].split(':')[0].strip()
                        constraint_lines = []
                        i += 1
                        
                        # Collect constraint text
                        while i < len(lines) and not lines[i].strip().startswith('context '):
                            l = lines[i].strip()
                            if l and not l.startswith('--'):
                                constraint_lines.append(l)
                            i += 1
                        
                        constraint_text = ' '.join(constraint_lines).strip()
                        
                        # Extract navigation paths
                        nav_pattern = r'self\.(\w+(?:\.\w+)*)'
                        navigations = re.findall(nav_pattern, constraint_text)
                        
                        constraints.append({
                            'context': context_class,
                            'name': inv_name,
                            'text': constraint_text,
                            'navigations': navigations
                        })
                        break
                    i += 1
            else:
                i += 1
        
        return constraints


class ModelConsistencyChecker:
    """Check consistency between XMI model and OCL constraints"""
    
    def __init__(self, xmi_file: str, ocl_file: str):
        """
        Initialize checker
        
        Args:
            xmi_file: Path to XMI model file
            ocl_file: Path to OCL constraints file
        """
        self.xmi_file = xmi_file
        self.ocl_file = ocl_file
        
        # Import here to avoid circular dependency
        from ..classifiers.sentence_transformer.xmi_based_domain_adapter import XMIModelExtractor
        self.xmi_extractor = XMIModelExtractor(xmi_file)
        self.ocl_parser = OCLConstraintParser()
    
    def validate(self) -> ValidationResult:
        """
        Perform comprehensive validation
        
        Returns:
            ValidationResult with detailed analysis
        """
        issues = []
        warnings = []
        stats = {}
        
        print("\n" + "="*80)
        print(" MODEL CONSISTENCY VALIDATION")
        print("="*80)
        print(f"XMI Model: {Path(self.xmi_file).name}")
        print(f"OCL File:  {Path(self.ocl_file).name}")
        print()
        
        # 1. Extract XMI classes
        xmi_classes = set(self.xmi_extractor.get_classes())
        stats['xmi_classes_count'] = len(xmi_classes)
        print(f" XMI Model Classes: {len(xmi_classes)}")
        print(f"   {', '.join(sorted(xmi_classes))}")
        
        # 2. Extract OCL context classes
        ocl_context_classes = self.ocl_parser.extract_context_classes(self.ocl_file)
        stats['ocl_context_classes_count'] = len(ocl_context_classes)
        print(f"\n OCL Context Classes: {len(ocl_context_classes)}")
        print(f"   {', '.join(sorted(ocl_context_classes))}")
        
        # 3. Check if context classes exist in XMI
        print(f"\n🔎 Checking Context Classes...")
        missing_contexts = ocl_context_classes - xmi_classes
        if missing_contexts:
            for cls in missing_contexts:
                issue = f"Context class '{cls}' not found in XMI model"
                issues.append(issue)
                print(f"    {issue}")
        else:
            print(f"    All {len(ocl_context_classes)} context classes found in XMI model")
        
        stats['missing_context_classes'] = list(missing_contexts)
        
        # 4. Parse full constraint details
        constraints = self.ocl_parser.parse_full_constraint_info(self.ocl_file)
        stats['total_constraints'] = len(constraints)
        print(f"\n Analyzing {len(constraints)} OCL Constraints...")
        
        # 5. Validate each constraint's navigations
        navigation_issues = []
        for constraint in constraints:
            context_class = constraint['context']
            
            # Check if context class exists
            if context_class not in xmi_classes:
                continue  # Already reported above
            
            # Get attributes/associations for this class
            available_props = set(self.xmi_extractor.get_attributes(context_class))
            
            # Check each navigation
            for nav in constraint['navigations']:
                parts = nav.split('.')
                first_prop = parts[0]
                
                # Check if first navigation step exists
                if first_prop not in available_props and first_prop not in [c.lower() for c in xmi_classes]:
                    # Property doesn't exist and isn't a known class
                    warning = f"Constraint '{constraint['name']}': Property '{first_prop}' not found in {context_class}"
                    navigation_issues.append(warning)
        
        if navigation_issues:
            print(f"     {len(navigation_issues)} navigation warnings:")
            for i, warning in enumerate(navigation_issues[:5], 1):
                print(f"      {i}. {warning}")
                warnings.append(warning)
            if len(navigation_issues) > 5:
                print(f"      ... and {len(navigation_issues) - 5} more")
        else:
            print(f"    All navigations appear valid")
        
        stats['navigation_issues_count'] = len(navigation_issues)
        
        # 6. Check coverage: Are there unused XMI classes?
        unused_classes = xmi_classes - ocl_context_classes
        if unused_classes:
            stats['unused_classes'] = list(unused_classes)
            print(f"\n Coverage Analysis:")
            print(f"     {len(unused_classes)} XMI classes have no OCL constraints:")
            print(f"      {', '.join(sorted(unused_classes))}")
        else:
            stats['unused_classes'] = []
            print(f"\n Coverage: All XMI classes have OCL constraints")
        
        # 7. Final verdict
        print("\n" + "="*80)
        is_valid = len(issues) == 0
        
        if is_valid:
            print(" VALIDATION PASSED: XMI and OCL belong to the same model")
            print(f"   • All {len(ocl_context_classes)} context classes found")
            print(f"   • {len(constraints)} constraints validated")
            if len(warnings) > 0:
                print(f"     {len(warnings)} warnings (non-critical)")
        else:
            print(" VALIDATION FAILED: XMI and OCL may be from different models")
            print(f"   • {len(issues)} critical issues found")
            print(f"   • {len(missing_contexts)} missing context classes")
        
        print("="*80 + "\n")
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            stats=stats
        )
    
    def validate_and_raise(self):
        """Validate and raise exception if validation fails"""
        result = self.validate()
        
        if not result.is_valid:
            error_msg = "Model consistency validation failed:\n"
            for issue in result.issues:
                error_msg += f"  - {issue}\n"
            raise ValueError(error_msg)
        
        return result


def quick_validate(xmi_file: str, ocl_file: str) -> bool:
    """
    Quick validation check
    
    Args:
        xmi_file: Path to XMI file
        ocl_file: Path to OCL file
    
    Returns:
        True if valid, False otherwise
    """
    checker = ModelConsistencyChecker(xmi_file, ocl_file)
    result = checker.validate()
    return result.is_valid


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python model_consistency_checker.py <xmi_file> <ocl_file>")
        sys.exit(1)
    
    xmi_file = sys.argv[1]
    ocl_file = sys.argv[2]
    
    checker = ModelConsistencyChecker(xmi_file, ocl_file)
    result = checker.validate()
    
    if result.is_valid:
        print("\n🎉 Models are consistent!")
        sys.exit(0)
    else:
        print(f"\n Found {len(result.issues)} issue(s)")
        sys.exit(1)
