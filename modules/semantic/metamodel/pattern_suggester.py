"""
Pattern Suggester Module
Suggests relevant OCL patterns based on metamodel analysis.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from ...core.models import Metamodel, Class, Attribute, Association
from .structure_analyzer import StructureAnalyzer
from .invariant_detector import InvariantDetector


@dataclass
class PatternSuggestion:
    """A suggested pattern for constraint generation."""
    pattern_id: str
    pattern_name: str
    context_class: str
    confidence: float
    priority: str  # critical, high, medium, low
    reason: str
    parameters: Dict
    related_elements: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class PatternSuggester:
    """Suggests OCL patterns based on metamodel structure."""
    
    def __init__(self, metamodel: Metamodel):
        self.metamodel = metamodel
        self.structure_analyzer = StructureAnalyzer(metamodel)
        self.invariant_detector = InvariantDetector(metamodel)
        self.suggestions: List[PatternSuggestion] = []
    
    def suggest_all_patterns(self) -> List[PatternSuggestion]:
        """Generate all pattern suggestions for the metamodel."""
        self.suggestions = []
        
        # Analyze structure
        structural_patterns = self.structure_analyzer.detect_patterns()
        
        # Detect invariants
        detected_invariants = self.invariant_detector.detect_all_invariants()
        
        # Generate suggestions from structural patterns
        for pattern in structural_patterns:
            self.suggestions.extend(self._suggest_from_structural_pattern(pattern))
        
        # Generate suggestions from detected invariants
        for invariant in detected_invariants:
            self.suggestions.append(self._suggest_from_invariant(invariant))
        
        # Generate suggestions from class analysis
        for cls in self.metamodel.classes.values():
            self.suggestions.extend(self._suggest_for_class(cls))
        
        # Sort by priority and confidence
        self.suggestions.sort(key=lambda s: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(s.priority, 0),
            s.confidence
        ), reverse=True)
        
        return self.suggestions
    
    def _suggest_from_structural_pattern(self, pattern) -> List[PatternSuggestion]:
        """Generate suggestions from a structural pattern."""
        suggestions = []
        
        for constraint_type in pattern.suggested_constraints:
            for class_name in pattern.classes:
                cls = self.metamodel.get_class(class_name)
                if not cls:
                    continue
                
                suggestion = PatternSuggestion(
                    pattern_id=constraint_type,
                    pattern_name=constraint_type.replace('_', ' ').title(),
                    context_class=class_name,
                    confidence=pattern.confidence,
                    priority='high' if pattern.confidence > 0.9 else 'medium',
                    reason=f"Based on {pattern.pattern_type} pattern: {pattern.description}",
                    parameters=self._infer_parameters(constraint_type, cls, pattern),
                    related_elements=pattern.classes,
                    tags=[pattern.pattern_type, 'structural']
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_from_invariant(self, invariant) -> PatternSuggestion:
        """Generate suggestion from detected invariant."""
        return PatternSuggestion(
            pattern_id=invariant.ocl_pattern,
            pattern_name=invariant.invariant_type.replace('_', ' ').title(),
            context_class=invariant.class_name,
            confidence=invariant.confidence,
            priority=invariant.priority,
            reason=invariant.rationale,
            parameters=invariant.parameters,
            related_elements=invariant.related_elements,
            examples=[],
            tags=['invariant', invariant.invariant_type]
        )
    
    def _suggest_for_class(self, cls: Class) -> List[PatternSuggestion]:
        """Generate pattern suggestions for a specific class."""
        suggestions = []
        
        # Get complexity metrics
        metrics = self.structure_analyzer.analyze_class_complexity(cls.name)
        if not metrics:
            return suggestions
        
        # Suggest based on complexity
        if metrics['complexity_level'] in ['high', 'very_high']:
            # High complexity classes need more constraints
            suggestions.extend(self._suggest_for_high_complexity_class(cls, metrics))
        
        # Suggest based on coupling
        if metrics['coupling'] > 5:
            suggestions.extend(self._suggest_for_highly_coupled_class(cls, metrics))
        
        # Suggest based on attributes
        suggestions.extend(self._suggest_for_attributes(cls))
        
        # Suggest based on associations
        suggestions.extend(self._suggest_for_associations(cls))
        
        return suggestions
    
    def _suggest_for_high_complexity_class(self, cls: Class, metrics: Dict) -> List[PatternSuggestion]:
        """Suggest patterns for high complexity classes."""
        suggestions = []
        
        # Suggest comprehensive validation
        suggestions.append(PatternSuggestion(
            pattern_id="forall_nested",
            pattern_name="Comprehensive Validation",
            context_class=cls.name,
            confidence=0.80,
            priority='high',
            reason=f"High complexity class (score: {metrics['complexity_score']:.1f}) needs comprehensive validation",
            parameters={},
            related_elements=[cls.name],
            tags=['complexity', 'validation']
        ))
        
        return suggestions
    
    def _suggest_for_highly_coupled_class(self, cls: Class, metrics: Dict) -> List[PatternSuggestion]:
        """Suggest patterns for highly coupled classes."""
        suggestions = []
        
        # Suggest consistency checks
        suggestions.append(PatternSuggestion(
            pattern_id="forall_nested",
            pattern_name="Consistency Check",
            context_class=cls.name,
            confidence=0.75,
            priority='medium',
            reason=f"High coupling (value: {metrics['coupling']}) suggests need for consistency constraints",
            parameters={},
            related_elements=[cls.name],
            tags=['coupling', 'consistency']
        ))
        
        return suggestions
    
    def _suggest_for_attributes(self, cls: Class) -> List[PatternSuggestion]:
        """Suggest patterns based on class attributes."""
        suggestions = []
        
        for attr in cls.attributes:
            # Numeric attributes
            if attr.type in ['Integer', 'Real', 'Double', 'Float']:
                suggestions.append(PatternSuggestion(
                    pattern_id="numeric_comparison",
                    pattern_name="Numeric Range Constraint",
                    context_class=cls.name,
                    confidence=0.70,
                    priority='medium',
                    reason=f"Numeric attribute '{attr.name}' can have range constraints",
                    parameters={'attribute': attr.name, 'operator': '>', 'value': 0},
                    related_elements=[attr.name],
                    tags=['numeric', 'attribute']
                ))
            
            # String attributes
            if attr.type == 'String':
                # Non-empty constraint
                suggestions.append(PatternSuggestion(
                    pattern_id="string_operation",
                    pattern_name="Non-Empty String",
                    context_class=cls.name,
                    confidence=0.65,
                    priority='low',
                    reason=f"String attribute '{attr.name}' can be validated for non-empty",
                    parameters={'attribute': attr.name, 'operation': 'size()', 'operator': '>', 'value': 0},
                    related_elements=[attr.name],
                    tags=['string', 'attribute']
                ))
                
                # Pattern matching for special strings
                if 'email' in attr.name.lower():
                    suggestions.append(PatternSuggestion(
                        pattern_id="string_pattern",
                        pattern_name="Email Validation",
                        context_class=cls.name,
                        confidence=0.90,
                        priority='high',
                        reason=f"Email attribute '{attr.name}' should match email pattern",
                        parameters={'attribute': attr.name, 'pattern': r'.*@.*\..*'},
                        related_elements=[attr.name],
                        tags=['string', 'validation', 'email']
                    ))
            
            # Boolean attributes
            if attr.type == 'Boolean':
                suggestions.append(PatternSuggestion(
                    pattern_id="boolean_guard_implies",
                    pattern_name="Boolean Guard Implication",
                    context_class=cls.name,
                    confidence=0.60,
                    priority='low',
                    reason=f"Boolean '{attr.name}' can guard other constraints",
                    parameters={'guard': attr.name, 'consequence': 'true'},
                    related_elements=[attr.name],
                    tags=['boolean', 'guard']
                ))
        
        return suggestions
    
    def _suggest_for_associations(self, cls: Class) -> List[PatternSuggestion]:
        """Suggest patterns based on class associations."""
        suggestions = []
        
        for assoc in self.metamodel.get_all_associations():
            if assoc.source_class != cls.name:
                continue
            
            # Collection associations
            if assoc.is_collection:
                # Size constraint
                suggestions.append(PatternSuggestion(
                    pattern_id="size_constraint",
                    pattern_name="Collection Size Constraint",
                    context_class=cls.name,
                    confidence=0.80,
                    priority='high',
                    reason=f"Collection '{assoc.ref_name}' should have size constraints",
                    parameters={'collection': assoc.ref_name, 'operator': '>=', 'value': 0},
                    related_elements=[assoc.ref_name, assoc.target_class],
                    tags=['collection', 'size'],
                    examples=[
                        f"self.{assoc.ref_name}->size() >= 1",
                        f"self.{assoc.ref_name}->size() <= 100"
                    ]
                ))
                
                # ForAll constraint
                suggestions.append(PatternSuggestion(
                    pattern_id="forall_nested",
                    pattern_name="Collection Element Constraint",
                    context_class=cls.name,
                    confidence=0.75,
                    priority='medium',
                    reason=f"All elements in '{assoc.ref_name}' can have constraints",
                    parameters={'collection': assoc.ref_name, 'variable': 'item', 'condition': 'true'},
                    related_elements=[assoc.ref_name, assoc.target_class],
                    tags=['collection', 'forall'],
                    examples=[
                        f"self.{assoc.ref_name}->forAll(item | item.oclIsTypeOf({assoc.target_class}))"
                    ]
                ))
                
                # Uniqueness constraint
                suggestions.append(PatternSuggestion(
                    pattern_id="uniqueness_constraint",
                    pattern_name="Unique Collection Elements",
                    context_class=cls.name,
                    confidence=0.70,
                    priority='medium',
                    reason=f"Collection '{assoc.ref_name}' elements should be unique",
                    parameters={'collection': assoc.ref_name},
                    related_elements=[assoc.ref_name, assoc.target_class],
                    tags=['collection', 'uniqueness'],
                    examples=[
                        f"self.{assoc.ref_name}->isUnique(item | item)"
                    ]
                ))
            
            # Composition relationships
            if assoc.is_composition:
                suggestions.append(PatternSuggestion(
                    pattern_id="null_check",
                    pattern_name="Composition Non-Null",
                    context_class=cls.name,
                    confidence=1.0,
                    priority='critical',
                    reason=f"Composition '{assoc.ref_name}' must not be null",
                    parameters={'attribute': assoc.ref_name},
                    related_elements=[assoc.ref_name, assoc.target_class],
                    tags=['composition', 'null_check']
                ))
        
        return suggestions
    
    def _infer_parameters(self, pattern_type: str, cls: Class, structural_pattern) -> Dict:
        """Infer parameters for a pattern based on class and structural pattern."""
        params = {}
        
        if pattern_type == 'size_constraint':
            # Find collection in class
            for assoc in self.metamodel.get_all_associations():
                if assoc.source_class == cls.name and assoc.is_collection:
                    params['collection'] = assoc.ref_name
                    params['operator'] = '>='
                    params['value'] = 1
                    break
        
        elif pattern_type == 'null_check':
            # Find first attribute or association
            if cls.attributes:
                params['attribute'] = cls.attributes[0].name
            else:
                for assoc in self.metamodel.get_all_associations():
                    if assoc.source_class == cls.name:
                        params['attribute'] = assoc.ref_name
                        break
        
        elif pattern_type == 'uniqueness_constraint':
            # Find collection
            for assoc in self.metamodel.get_all_associations():
                if assoc.source_class == cls.name and assoc.is_collection:
                    params['collection'] = assoc.ref_name
                    break
        
        elif pattern_type == 'type_check':
            if cls.parent_class:
                params['type'] = cls.parent_class
        
        return params
    
    def suggest_for_class(self, class_name: str) -> List[PatternSuggestion]:
        """Get all pattern suggestions for a specific class."""
        cls = self.metamodel.get_class(class_name)
        if not cls:
            return []
        
        return [s for s in self.suggest_all_patterns() if s.context_class == class_name]
    
    def suggest_by_priority(self, priority: str) -> List[PatternSuggestion]:
        """Get suggestions filtered by priority."""
        return [s for s in self.suggest_all_patterns() if s.priority == priority]
    
    def suggest_by_pattern_id(self, pattern_id: str) -> List[PatternSuggestion]:
        """Get all suggestions for a specific pattern."""
        return [s for s in self.suggest_all_patterns() if s.pattern_id == pattern_id]
    
    def suggest_by_tag(self, tag: str) -> List[PatternSuggestion]:
        """Get suggestions filtered by tag."""
        return [s for s in self.suggest_all_patterns() if tag in s.tags]
    
    def get_top_suggestions(self, n: int = 10) -> List[PatternSuggestion]:
        """Get top N suggestions by priority and confidence."""
        all_suggestions = self.suggest_all_patterns()
        return all_suggestions[:n]
    
    def export_suggestions(self) -> Dict:
        """Export suggestions in structured format."""
        all_suggestions = self.suggest_all_patterns()
        
        return {
            'total_count': len(all_suggestions),
            'by_priority': {
                'critical': len([s for s in all_suggestions if s.priority == 'critical']),
                'high': len([s for s in all_suggestions if s.priority == 'high']),
                'medium': len([s for s in all_suggestions if s.priority == 'medium']),
                'low': len([s for s in all_suggestions if s.priority == 'low'])
            },
            'by_class': {
                cls.name: len([s for s in all_suggestions if s.context_class == cls.name])
                for cls in self.metamodel.classes.values()
            },
            'suggestions': [
                {
                    'pattern_id': s.pattern_id,
                    'pattern_name': s.pattern_name,
                    'context': s.context_class,
                    'confidence': s.confidence,
                    'priority': s.priority,
                    'reason': s.reason,
                    'parameters': s.parameters,
                    'examples': s.examples
                }
                for s in all_suggestions
            ]
        }


if __name__ == "__main__":
    print("Pattern Suggester Module - Ready for testing")
