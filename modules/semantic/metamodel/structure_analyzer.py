"""
Structure Analyzer Module
Analyzes metamodel structure to detect patterns and relationships.
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from ...core.models import Metamodel, Class, Association


@dataclass
class StructuralPattern:
    """Detected structural pattern in metamodel."""
    pattern_type: str
    classes: List[str]
    confidence: float
    description: str
    suggested_constraints: List[str]
    metadata: Optional[Dict] = None


class StructureAnalyzer:
    """Analyzes metamodel structure for patterns and relationships."""
    
    def __init__(self, metamodel: Metamodel):
        self.metamodel = metamodel
        self.dependency_graph = {}
        self.inheritance_tree = {}
        self.association_matrix = {}
        self.class_metrics = {}
        self._build_graphs()
        self._compute_metrics()
    
    def _build_graphs(self):
        """Build dependency and inheritance graphs."""
        # Initialize dependency graph
        for cls in self.metamodel.classes.values():
            self.dependency_graph[cls.name] = []
        
        # Build dependency graph from associations
        for assoc in self.metamodel.get_all_associations():
            if assoc.source_class in self.dependency_graph:
                self.dependency_graph[assoc.source_class].append({
                    'target': assoc.target_class,
                    'name': assoc.ref_name,
                    'type': 'composition' if assoc.is_composition else 'association',
                    'multiplicity': assoc.multiplicity
                })
        
        # Build inheritance tree
        for cls in self.metamodel.classes.values():
            if cls.parent_class:
                if cls.parent_class not in self.inheritance_tree:
                    self.inheritance_tree[cls.parent_class] = []
                self.inheritance_tree[cls.parent_class].append(cls.name)
        
        # Build association matrix
        for assoc in self.metamodel.get_all_associations():
            key = (assoc.source_class, assoc.target_class)
            if key not in self.association_matrix:
                self.association_matrix[key] = []
            self.association_matrix[key].append(assoc)
    
    def _compute_metrics(self):
        """Compute complexity metrics for each class."""
        for cls in self.metamodel.classes.values():
            num_attributes = len(cls.attributes)
            num_outgoing = len(self.dependency_graph.get(cls.name, []))
            num_incoming = sum(1 for deps in self.dependency_graph.values() 
                             for dep in deps if dep['target'] == cls.name)
            num_children = len(self.inheritance_tree.get(cls.name, []))
            
            # Weighted complexity score
            complexity = (
                num_attributes * 1.0 +
                num_outgoing * 2.0 +
                num_incoming * 1.5 +
                num_children * 1.5
            )
            
            self.class_metrics[cls.name] = {
                'num_attributes': num_attributes,
                'num_outgoing_associations': num_outgoing,
                'num_incoming_associations': num_incoming,
                'num_children': num_children,
                'complexity_score': complexity,
                'complexity_level': self._classify_complexity(complexity),
                'coupling': num_outgoing + num_incoming,
                'cohesion': self._calculate_cohesion(cls)
            }
    
    def _classify_complexity(self, score: float) -> str:
        """Classify complexity level."""
        if score > 15:
            return 'very_high'
        elif score > 10:
            return 'high'
        elif score > 5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_cohesion(self, cls: Class) -> float:
        """Calculate class cohesion (0-1 scale)."""
        if not cls.attributes:
            return 1.0
        
        # Simple cohesion: ratio of attributes to theoretical max
        max_attributes = 20  # Theoretical max for good design
        return min(1.0, len(cls.attributes) / max_attributes)
    
    def detect_patterns(self) -> List[StructuralPattern]:
        """Detect structural patterns in metamodel."""
        patterns = []
        
        # Detect composition patterns
        patterns.extend(self._detect_composition())
        
        # Detect aggregation patterns
        patterns.extend(self._detect_aggregation())
        
        # Detect inheritance hierarchies
        patterns.extend(self._detect_inheritance())
        
        # Detect association patterns
        patterns.extend(self._detect_association_patterns())
        
        # Detect circular dependencies
        patterns.extend(self._detect_circular_dependencies())
        
        # Detect singleton candidates
        patterns.extend(self._detect_singleton_candidates())
        
        # Detect many-to-many relationships
        patterns.extend(self._detect_many_to_many())
        
        return patterns
    
    def _detect_composition(self) -> List[StructuralPattern]:
        """Detect composition relationships."""
        patterns = []
        
        for assoc in self.metamodel.get_all_associations():
            if assoc.is_composition:
                patterns.append(StructuralPattern(
                    pattern_type="composition",
                    classes=[assoc.source_class, assoc.target_class],
                    confidence=1.0,
                    description=f"{assoc.source_class} composes {assoc.target_class}",
                    suggested_constraints=[
                        "null_check",  # Target must not be null
                        "uniqueness_constraint"  # Unique ownership
                    ],
                    metadata={
                        'association': assoc.name,
                        'multiplicity': assoc.multiplicity
                    }
                ))
        
        return patterns
    
    def _detect_aggregation(self) -> List[StructuralPattern]:
        """Detect aggregation relationships."""
        patterns = []
        
        for assoc in self.metamodel.get_all_associations():
            if not assoc.is_composition and assoc.is_collection:
                patterns.append(StructuralPattern(
                    pattern_type="aggregation",
                    classes=[assoc.source_class, assoc.target_class],
                    confidence=0.9,
                    description=f"{assoc.source_class} aggregates multiple {assoc.target_class}",
                    suggested_constraints=[
                        "size_constraint",  # Size constraints on collection
                        "forall_nested",  # Constraints on all elements
                        "uniqueness_constraint"  # Unique elements
                    ],
                    metadata={
                        'association': assoc.name,
                        'multiplicity': assoc.multiplicity
                    }
                ))
        
        return patterns
    
    def _detect_inheritance(self) -> List[StructuralPattern]:
        """Detect inheritance hierarchies."""
        patterns = []
        
        for parent, children in self.inheritance_tree.items():
            patterns.append(StructuralPattern(
                pattern_type="inheritance",
                classes=[parent] + children,
                confidence=1.0,
                description=f"{parent} is inherited by {', '.join(children)}",
                suggested_constraints=[
                    "type_check",  # Type checking
                    "oclIsKindOf",  # Kind checking
                    "oclAsType"  # Type casting
                ],
                metadata={
                    'parent': parent,
                    'children': children,
                    'depth': 1  # Single level detected
                }
            ))
        
        return patterns
    
    def _detect_association_patterns(self) -> List[StructuralPattern]:
        """Detect common association patterns."""
        patterns = []
        
        # Detect bidirectional associations
        all_assocs = self.metamodel.get_all_associations()
        for assoc1 in all_assocs:
            for assoc2 in all_assocs:
                if (assoc1.source_class == assoc2.target_class and 
                    assoc1.target_class == assoc2.source_class and
                    assoc1.name != assoc2.name):
                    patterns.append(StructuralPattern(
                        pattern_type="bidirectional",
                        classes=[assoc1.source_class, assoc1.target_class],
                        confidence=1.0,
                        description=f"Bidirectional association between {assoc1.source_class} and {assoc1.target_class}",
                        suggested_constraints=[
                            "consistency_check",
                            "inverse_constraint"
                        ],
                        metadata={
                            'forward': assoc1.name,
                            'backward': assoc2.name
                        }
                    ))
        
        return patterns
    
    def _detect_circular_dependencies(self) -> List[StructuralPattern]:
        """Detect circular dependencies."""
        patterns = []
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            
            for dep in self.dependency_graph.get(node, []):
                target = dep['target']
                if target not in visited:
                    cycle = has_cycle(target, path + [target])
                    if cycle:
                        return cycle
                elif target in rec_stack:
                    # Found cycle
                    cycle_start = path.index(target) if target in path else 0
                    return path[cycle_start:] + [target]
            
            rec_stack.remove(node)
            return None
        
        for class_name in self.metamodel.get_class_names():
            if class_name not in visited:
                cycle = has_cycle(class_name, [class_name])
                if cycle:
                    patterns.append(StructuralPattern(
                        pattern_type="circular_dependency",
                        classes=cycle,
                        confidence=1.0,
                        description=f"Circular dependency detected: {' -> '.join(cycle)}",
                        suggested_constraints=[
                            "acyclicity",
                            "null_check"
                        ],
                        metadata={
                            'cycle': cycle
                        }
                    ))
        
        return patterns
    
    def _detect_singleton_candidates(self) -> List[StructuralPattern]:
        """Detect classes that might be singletons."""
        patterns = []
        
        for cls in self.metamodel.classes.values():
            # Heuristic: class with no incoming associations and few attributes
            incoming_count = sum(1 for deps in self.dependency_graph.values()
                               for dep in deps if dep['target'] == cls.name)
            
            if incoming_count == 0 and len(cls.attributes) <= 3:
                patterns.append(StructuralPattern(
                    pattern_type="singleton_candidate",
                    classes=[cls.name],
                    confidence=0.6,
                    description=f"{cls.name} might be a singleton",
                    suggested_constraints=[
                        "uniqueness_constraint",
                        "size_constraint"
                    ],
                    metadata={
                        'num_attributes': len(cls.attributes)
                    }
                ))
        
        return patterns
    
    def _detect_many_to_many(self) -> List[StructuralPattern]:
        """Detect many-to-many relationships."""
        patterns = []
        
        for assoc in self.metamodel.get_all_associations():
            if assoc.is_collection:
                # Check if reverse is also a collection (many-to-many)
                reverse_assocs = [a for a in self.metamodel.get_all_associations() 
                                if a.source_class == assoc.target_class and a.target_class == assoc.source_class]
                if any(a.is_collection for a in reverse_assocs):
                    patterns.append(StructuralPattern(
                        pattern_type="many_to_many",
                        classes=[assoc.source_class, assoc.target_class],
                        confidence=1.0,
                        description=f"Many-to-many between {assoc.source_class} and {assoc.target_class}",
                        suggested_constraints=[
                            "size_constraint",
                            "uniqueness_constraint",
                            "forall_nested"
                        ],
                        metadata={
                            'association': assoc.name
                        }
                    ))
        
        return patterns
    
    def analyze_class_complexity(self, class_name: str) -> Optional[Dict]:
        """Analyze complexity metrics for a class."""
        return self.class_metrics.get(class_name)
    
    def get_related_classes(self, class_name: str, depth: int = 1) -> Set[str]:
        """Get all classes related to given class within depth."""
        related = set()
        to_visit = [(class_name, 0)]
        visited = set()
        
        while to_visit:
            current, current_depth = to_visit.pop(0)
            if current in visited or current_depth > depth:
                continue
            
            visited.add(current)
            if current != class_name:
                related.add(current)
            
            # Add neighbors from dependency graph
            for dep in self.dependency_graph.get(current, []):
                target = dep['target']
                if target not in visited:
                    to_visit.append((target, current_depth + 1))
        
        return related
    
    def suggest_constraints_for_class(self, class_name: str) -> List[Dict]:
        """Suggest appropriate constraints for a class."""
        suggestions = []
        
        cls = self.metamodel.get_class(class_name)
        if not cls:
            return suggestions
        
        # Analyze class complexity
        metrics = self.class_metrics.get(class_name, {})
        
        # Suggest constraints for attributes
        for attr in cls.attributes:
            if attr.type in ['Integer', 'Real', 'Double', 'Float']:
                suggestions.append({
                    'pattern': 'numeric_comparison',
                    'target': attr.name,
                    'reason': f'Numeric attribute {attr.name} could have range constraint',
                    'priority': 'medium',
                    'parameters': {'attribute': attr.name, 'operator': '>', 'value': 0}
                })
            
            if attr.type == 'String':
                suggestions.append({
                    'pattern': 'null_check',
                    'target': attr.name,
                    'reason': f'String attribute {attr.name} should be non-null',
                    'priority': 'high' if not attr.is_optional else 'low',
                    'parameters': {'attribute': attr.name}
                })
            
            if attr.type == 'Boolean':
                suggestions.append({
                    'pattern': 'boolean_guard',
                    'target': attr.name,
                    'reason': f'Boolean attribute {attr.name} can guard other constraints',
                    'priority': 'low',
                    'parameters': {'guard': attr.name}
                })
        
        # Suggest constraints for associations
        for dep in self.dependency_graph.get(class_name, []):
            if '*' in dep.get('multiplicity', ''):
                suggestions.append({
                    'pattern': 'size_constraint',
                    'target': dep['name'],
                    'reason': f'Collection {dep["name"]} should have size constraint',
                    'priority': 'high',
                    'parameters': {'collection': dep['name'], 'operator': '>=', 'value': 1}
                })
                
                suggestions.append({
                    'pattern': 'forall_nested',
                    'target': dep['name'],
                    'reason': f'All elements in {dep["name"]} should satisfy constraints',
                    'priority': 'medium',
                    'parameters': {'collection': dep['name'], 'variable': 'item'}
                })
        
        # High complexity warnings
        if metrics.get('complexity_level') in ['high', 'very_high']:
            suggestions.append({
                'pattern': 'custom',
                'target': class_name,
                'reason': f'High complexity class - consider refactoring or additional constraints',
                'priority': 'info',
                'parameters': {'complexity': metrics['complexity_score']}
            })
        
        return suggestions
    
    def get_class_dependencies(self, class_name: str) -> Dict[str, List[str]]:
        """Get detailed dependencies for a class."""
        dependencies = {
            'outgoing': [],
            'incoming': [],
            'children': [],
            'parent': None
        }
        
        # Outgoing dependencies
        for dep in self.dependency_graph.get(class_name, []):
            dependencies['outgoing'].append(dep['target'])
        
        # Incoming dependencies
        for source, deps in self.dependency_graph.items():
            for dep in deps:
                if dep['target'] == class_name:
                    dependencies['incoming'].append(source)
        
        # Inheritance
        dependencies['children'] = self.inheritance_tree.get(class_name, [])
        
        cls = self.metamodel.get_class(class_name)
        if cls and cls.parent_class:
            dependencies['parent'] = cls.parent_class
        
        return dependencies
    
    def export_analysis(self) -> Dict:
        """Export complete structural analysis."""
        return {
            'patterns': [
                {
                    'type': p.pattern_type,
                    'classes': p.classes,
                    'confidence': p.confidence,
                    'description': p.description,
                    'constraints': p.suggested_constraints
                }
                for p in self.detect_patterns()
            ],
            'class_metrics': self.class_metrics,
            'dependency_graph': self.dependency_graph,
            'inheritance_tree': self.inheritance_tree
        }


if __name__ == "__main__":
    print("Structure Analyzer Module - Ready for testing")
