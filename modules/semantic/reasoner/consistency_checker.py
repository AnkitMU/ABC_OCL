"""
Consistency Checker Module
Detects inconsistencies and conflicts between OCL constraints.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from ...core.models import Metamodel, OCLConstraint


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue between constraints."""
    issue_type: str  # 'conflict', 'contradiction', 'redundancy'
    severity: str  # 'critical', 'warning', 'info'
    description: str
    constraints_involved: List[str]  # Constraint IDs or descriptions
    suggestion: str
    context_class: str


class ConsistencyChecker:
    """Checks consistency between multiple OCL constraints."""
    
    def __init__(self, metamodel: Metamodel):
        self.metamodel = metamodel
        self.issues: List[ConsistencyIssue] = []
    
    def check_consistency(self, constraints: List[OCLConstraint]) -> List[ConsistencyIssue]:
        """
        Check consistency among a set of constraints.
        
        Args:
            constraints: List of OCL constraints to check
            
        Returns:
            List of detected consistency issues
        """
        self.issues = []
        
        # Group constraints by context
        by_context = self._group_by_context(constraints)
        
        for context, ctx_constraints in by_context.items():
            # Check for direct conflicts
            self.issues.extend(self._check_direct_conflicts(ctx_constraints))
            
            # Check for impossible bounds
            self.issues.extend(self._check_impossible_bounds(ctx_constraints))
            
            # Check for redundancies
            self.issues.extend(self._check_redundancies(ctx_constraints))
            
            # Check for type conflicts
            self.issues.extend(self._check_type_conflicts(ctx_constraints))
            
            # Check for range conflicts
            self.issues.extend(self._check_range_conflicts(ctx_constraints))
        
        # Check cross-context issues
        self.issues.extend(self._check_cross_context_issues(constraints))
        
        return self.issues
    
    def _group_by_context(self, constraints: List[OCLConstraint]) -> Dict[str, List[OCLConstraint]]:
        """Group constraints by their context class."""
        by_context = {}
        for constraint in constraints:
            if constraint.context not in by_context:
                by_context[constraint.context] = []
            by_context[constraint.context].append(constraint)
        return by_context
    
    def _check_direct_conflicts(self, constraints: List[OCLConstraint]) -> List[ConsistencyIssue]:
        """Check for direct conflicts (e.g., x > 5 and x < 3)."""
        issues = []
        
        for i, c1 in enumerate(constraints):
            for c2 in constraints[i+1:]:
                # Check if both constraints reference the same attribute
                if self._reference_same_element(c1, c2):
                    # Check for conflicting operators
                    if self._are_conflicting(c1, c2):
                        issues.append(ConsistencyIssue(
                            issue_type='conflict',
                            severity='critical',
                            description=f"Conflicting constraints on {self._extract_target(c1)}",
                            constraints_involved=[c1.ocl, c2.ocl],
                            suggestion="Review and adjust constraint bounds to be compatible",
                            context_class=c1.context
                        ))
        
        return issues
    
    def _check_impossible_bounds(self, constraints: List[OCLConstraint]) -> List[ConsistencyIssue]:
        """Check for impossible combinations (e.g., size > 10 and size = 5)."""
        issues = []
        
        # Track bounds for each element
        element_bounds = {}
        
        for constraint in constraints:
            expr = constraint.ocl
            
            # Extract size constraints
            if '->size()' in expr:
                target = self._extract_collection_target(expr)
                if target:
                    if target not in element_bounds:
                        element_bounds[target] = []
                    element_bounds[target].append(constraint)
        
        # Check if bounds are impossible
        for target, target_constraints in element_bounds.items():
            if len(target_constraints) >= 2:
                # Extract numeric bounds
                bounds_info = []
                for c in target_constraints:
                    bound_info = self._extract_bound_info(c.ocl)
                    if bound_info:
                        bounds_info.append((c, bound_info))
                
                # Check for impossible combinations
                for i, (c1, (op1, val1)) in enumerate(bounds_info):
                    for c2, (op2, val2) in bounds_info[i+1:]:
                        if self._bounds_impossible(op1, val1, op2, val2):
                            issues.append(ConsistencyIssue(
                                issue_type='contradiction',
                                severity='critical',
                                description=f"Impossible bounds on {target}: {op1} {val1} and {op2} {val2}",
                                constraints_involved=[c1.ocl, c2.ocl],
                                suggestion=f"Constraints cannot both be satisfied",
                                context_class=c1.context
                            ))
        
        return issues
    
    def _check_redundancies(self, constraints: List[OCLConstraint]) -> List[ConsistencyIssue]:
        """Check for redundant constraints."""
        issues = []
        
        for i, c1 in enumerate(constraints):
            for c2 in constraints[i+1:]:
                if self._is_redundant(c1, c2):
                    issues.append(ConsistencyIssue(
                        issue_type='redundancy',
                        severity='info',
                        description=f"Redundant constraint detected",
                        constraints_involved=[c1.ocl, c2.ocl],
                        suggestion="One constraint implies the other; consider removing redundancy",
                        context_class=c1.context
                    ))
        
        return issues
    
    def _check_type_conflicts(self, constraints: List[OCLConstraint]) -> List[ConsistencyIssue]:
        """Check for type conflicts."""
        issues = []
        
        for constraint in constraints:
            expr = constraint.ocl
            
            # Check for type mismatches (simple heuristic)
            if 'oclIsTypeOf' in expr or 'oclAsType' in expr:
                # Extract type requirements
                # Check against metamodel
                pass  # Complex type inference
        
        return issues
    
    def _check_range_conflicts(self, constraints: List[OCLConstraint]) -> List[ConsistencyIssue]:
        """Check for range conflicts on numeric attributes."""
        issues = []
        
        # Group constraints by attribute
        attr_constraints = {}
        
        for constraint in constraints:
            # Simple pattern matching for attribute comparisons
            for attr_name in self._extract_attributes(constraint):
                if attr_name not in attr_constraints:
                    attr_constraints[attr_name] = []
                attr_constraints[attr_name].append(constraint)
        
        # Check each attribute's constraints
        for attr, attr_cons in attr_constraints.items():
            if len(attr_cons) >= 2:
                # Check if ranges are compatible
                ranges = []
                for c in attr_cons:
                    range_info = self._extract_range(c.ocl)
                    if range_info:
                        ranges.append((c, range_info))
                
                # Check for empty intersection
                if self._ranges_incompatible(ranges):
                    issues.append(ConsistencyIssue(
                        issue_type='conflict',
                        severity='critical',
                        description=f"Incompatible ranges on attribute {attr}",
                        constraints_involved=[c.ocl for c, _ in ranges],
                        suggestion="Adjust ranges to have non-empty intersection",
                        context_class=attr_cons[0].context
                    ))
        
        return issues
    
    def _check_cross_context_issues(self, constraints: List[OCLConstraint]) -> List[ConsistencyIssue]:
        """Check for issues across different contexts."""
        issues = []
        
        # Check for bidirectional consistency
        for c1 in constraints:
            for c2 in constraints:
                if c1.context != c2.context:
                    # Check if they reference each other
                    if self._is_bidirectional_inconsistent(c1, c2):
                        issues.append(ConsistencyIssue(
                            issue_type='conflict',
                            severity='warning',
                            description=f"Potential bidirectional inconsistency between {c1.context} and {c2.context}",
                            constraints_involved=[c1.ocl, c2.ocl],
                            suggestion="Ensure bidirectional associations are consistent",
                            context_class=f"{c1.context}/{c2.context}"
                        ))
        
        return issues
    
    # Helper methods
    
    def _reference_same_element(self, c1: OCLConstraint, c2: OCLConstraint) -> bool:
        """Check if two constraints reference the same element."""
        targets1 = set(self._extract_attributes(c1))
        targets2 = set(self._extract_attributes(c2))
        return bool(targets1 & targets2)
    
    def _are_conflicting(self, c1: OCLConstraint, c2: OCLConstraint) -> bool:
        """Check if two constraints directly conflict."""
        expr1 = c1.ocl
        expr2 = c2.ocl
        
        # Simple conflict detection
        # e.g., "x > 5" and "x < 3"
        if '>' in expr1 and '<' in expr2:
            vals1 = self._extract_numbers(expr1)
            vals2 = self._extract_numbers(expr2)
            if vals1 and vals2:
                return vals1[0] > vals2[0]
        
        return False
    
    def _is_redundant(self, c1: OCLConstraint, c2: OCLConstraint) -> bool:
        """Check if one constraint makes the other redundant."""
        expr1 = c1.ocl
        expr2 = c2.ocl
        
        # Simple redundancy detection
        # e.g., "x >= 10" makes "x >= 5" redundant
        if '>=' in expr1 and '>=' in expr2:
            if self._extract_target(c1) == self._extract_target(c2):
                vals1 = self._extract_numbers(expr1)
                vals2 = self._extract_numbers(expr2)
                if vals1 and vals2:
                    return vals1[0] > vals2[0]
        
        return False
    
    def _is_bidirectional_inconsistent(self, c1: OCLConstraint, c2: OCLConstraint) -> bool:
        """Check for bidirectional inconsistency."""
        # Complex check for bidirectional associations
        return False  # Placeholder
    
    def _extract_target(self, constraint: OCLConstraint) -> str:
        """Extract the target element from constraint."""
        expr = constraint.ocl
        # Simple extraction
        parts = expr.split('.')
        if len(parts) > 1:
            return parts[1].split('->')[0].split(' ')[0]
        return "unknown"
    
    def _extract_collection_target(self, expr: str) -> Optional[str]:
        """Extract collection name from size constraint."""
        if '->size()' in expr:
            parts = expr.split('->size()')[0]
            return parts.split('.')[-1].strip()
        return None
    
    def _extract_bound_info(self, expr: str) -> Optional[Tuple[str, int]]:
        """Extract operator and value from bound expression."""
        import re
        # Match patterns like ">= 5", "< 10", "= 3"
        match = re.search(r'([><=!]+)\s*(\d+)', expr)
        if match:
            return (match.group(1), int(match.group(2)))
        return None
    
    def _bounds_impossible(self, op1: str, val1: int, op2: str, val2: int) -> bool:
        """Check if two bounds are impossible together."""
        # e.g., "> 10" and "< 5"
        if op1 in ['>', '>='] and op2 in ['<', '<=']:
            return val1 >= val2
        if op1 in ['<', '<='] and op2 in ['>', '>=']:
            return val2 >= val1
        if op1 == '=' and op2 == '=' and val1 != val2:
            return True
        return False
    
    def _extract_attributes(self, constraint: OCLConstraint) -> List[str]:
        """Extract attribute names from constraint."""
        import re
        expr = constraint.ocl
        # Simple pattern: self.attribute
        matches = re.findall(r'self\.(\w+)', expr)
        return matches
    
    def _extract_range(self, expr: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """Extract numeric range from expression."""
        import re
        numbers = re.findall(r'\d+', expr)
        if numbers:
            if '>=' in expr or '>' in expr:
                return (int(numbers[0]), None)
            elif '<=' in expr or '<' in expr:
                return (None, int(numbers[0]))
        return None
    
    def _ranges_incompatible(self, ranges: List[Tuple[OCLConstraint, Tuple]]) -> bool:
        """Check if ranges have empty intersection."""
        if len(ranges) < 2:
            return False
        
        # Extract min and max from all ranges
        min_val = None
        max_val = None
        
        for _, (lower, upper) in ranges:
            if lower is not None:
                min_val = lower if min_val is None else max(min_val, lower)
            if upper is not None:
                max_val = upper if max_val is None else min(max_val, upper)
        
        # Check if min > max
        if min_val is not None and max_val is not None:
            return min_val > max_val
        
        return False
    
    def _extract_numbers(self, expr: str) -> List[int]:
        """Extract all numbers from expression."""
        import re
        return [int(n) for n in re.findall(r'\d+', expr)]
    
    def get_critical_issues(self) -> List[ConsistencyIssue]:
        """Get only critical issues."""
        return [i for i in self.issues if i.severity == 'critical']
    
    def get_issues_by_context(self, context: str) -> List[ConsistencyIssue]:
        """Get issues for specific context."""
        return [i for i in self.issues if context in i.context_class]
    
    def export_report(self) -> Dict:
        """Export consistency check report."""
        return {
            'total_issues': len(self.issues),
            'by_severity': {
                'critical': len([i for i in self.issues if i.severity == 'critical']),
                'warning': len([i for i in self.issues if i.severity == 'warning']),
                'info': len([i for i in self.issues if i.severity == 'info'])
            },
            'by_type': {
                'conflict': len([i for i in self.issues if i.issue_type == 'conflict']),
                'contradiction': len([i for i in self.issues if i.issue_type == 'contradiction']),
                'redundancy': len([i for i in self.issues if i.issue_type == 'redundancy'])
            },
            'issues': [
                {
                    'type': i.issue_type,
                    'severity': i.severity,
                    'description': i.description,
                    'constraints': i.constraints_involved,
                    'suggestion': i.suggestion
                }
                for i in self.issues
            ]
        }


if __name__ == "__main__":
    print("Consistency Checker Module - Ready for testing")
