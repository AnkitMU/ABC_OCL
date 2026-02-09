"""
Invariant Detector Module
Automatically detects implicit invariants from metamodel structure.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from ...core.models import Metamodel, Class, Attribute, Association


@dataclass
class DetectedInvariant:
    """An automatically detected invariant."""
    class_name: str
    invariant_type: str
    description: str
    ocl_pattern: str
    confidence: float
    parameters: Dict
    priority: str = "medium"  # low, medium, high, critical
    rationale: str = ""
    related_elements: List[str] = field(default_factory=list)


class InvariantDetector:
    """Detects implicit invariants from metamodel."""
    
    def __init__(self, metamodel: Metamodel):
        self.metamodel = metamodel
        self.detected_invariants: List[DetectedInvariant] = []
    
    def detect_all_invariants(self) -> List[DetectedInvariant]:
        """Detect all implicit invariants."""
        self.detected_invariants = []
        
        for cls in self.metamodel.classes.values():
            # Detect attribute-based invariants
            self.detected_invariants.extend(self._detect_attribute_invariants(cls))
            
            # Detect association-based invariants
            self.detected_invariants.extend(self._detect_association_invariants(cls))
            
            # Detect multiplicity invariants
            self.detected_invariants.extend(self._detect_multiplicity_invariants(cls))
            
            # Detect type-based invariants
            self.detected_invariants.extend(self._detect_type_invariants(cls))
            
            # Detect uniqueness invariants
            self.detected_invariants.extend(self._detect_uniqueness_invariants(cls))
            
            # Detect domain-specific invariants
            self.detected_invariants.extend(self._detect_domain_invariants(cls))
        
        # Detect cross-class invariants
        self.detected_invariants.extend(self._detect_cross_class_invariants())
        
        return self.detected_invariants
    
    def _detect_attribute_invariants(self, cls: Class) -> List[DetectedInvariant]:
        """Detect invariants from attributes."""
        invariants = []
        
        for attr in cls.attributes:
            # Non-null invariants for required attributes
            if not attr.is_optional:
                invariants.append(DetectedInvariant(
                    class_name=cls.name,
                    invariant_type="non_null",
                    description=f"{attr.name} must not be null",
                    ocl_pattern="null_check",
                    confidence=1.0,
                    parameters={'attribute': attr.name},
                    priority="high",
                    rationale="Required attribute must have a value",
                    related_elements=[attr.name]
                ))
            
            # Numeric attribute invariants
            if attr.type in ['Integer', 'Real', 'Double', 'Float']:
                # Positive value invariant (heuristic based on naming)
                if any(word in attr.name.lower() for word in ['count', 'size', 'length', 'age', 'quantity']):
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="positive_value",
                        description=f"{attr.name} should be non-negative",
                        ocl_pattern="numeric_comparison",
                        confidence=0.85,
                        parameters={'attribute': attr.name, 'operator': '>=', 'value': 0},
                        priority="high",
                        rationale=f"Attribute name '{attr.name}' suggests non-negative value",
                        related_elements=[attr.name]
                    ))
                
                # Price/cost invariants
                if any(word in attr.name.lower() for word in ['price', 'cost', 'amount', 'fee']):
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="positive_value",
                        description=f"{attr.name} should be positive",
                        ocl_pattern="numeric_comparison",
                        confidence=0.80,
                        parameters={'attribute': attr.name, 'operator': '>', 'value': 0},
                        priority="high",
                        rationale=f"Monetary attribute '{attr.name}' should be positive",
                        related_elements=[attr.name]
                    ))
                
                # Percentage invariants
                if 'percent' in attr.name.lower() or 'ratio' in attr.name.lower():
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="range_constraint",
                        description=f"{attr.name} should be between 0 and 100",
                        ocl_pattern="range_constraint",
                        confidence=0.90,
                        parameters={'attribute': attr.name, 'min_value': 0, 'max_value': 100},
                        priority="high",
                        rationale=f"Percentage attribute '{attr.name}' should be in valid range",
                        related_elements=[attr.name]
                    ))
            
            # String attribute invariants
            if attr.type == 'String':
                # Email validation
                if 'email' in attr.name.lower():
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="string_pattern",
                        description=f"{attr.name} should match email pattern",
                        ocl_pattern="string_pattern",
                        confidence=0.95,
                        parameters={'attribute': attr.name, 'pattern': r'.*@.*\..*'},
                        priority="high",
                        rationale="Email attribute should be valid email format",
                        related_elements=[attr.name]
                    ))
                
                # Non-empty string
                if not attr.is_optional:
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="non_empty_string",
                        description=f"{attr.name} should not be empty",
                        ocl_pattern="string_operation",
                        confidence=0.85,
                        parameters={'attribute': attr.name, 'operation': 'size()', 'operator': '>', 'value': 0},
                        priority="medium",
                        rationale="Required string should not be empty",
                        related_elements=[attr.name]
                    ))
                
                # Name attributes
                if 'name' in attr.name.lower():
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="name_length",
                        description=f"{attr.name} length should be reasonable",
                        ocl_pattern="string_operation",
                        confidence=0.70,
                        parameters={'attribute': attr.name, 'operation': 'size()', 'operator': '<=', 'value': 100},
                        priority="low",
                        rationale="Name should have reasonable length",
                        related_elements=[attr.name]
                    ))
            
            # Boolean attribute implications
            if attr.type == 'Boolean':
                invariants.append(DetectedInvariant(
                    class_name=cls.name,
                    invariant_type="boolean_guard",
                    description=f"{attr.name} can guard other constraints",
                    ocl_pattern="boolean_guard",
                    confidence=0.60,
                    parameters={'guard': attr.name},
                    priority="low",
                    rationale="Boolean can be used as conditional guard",
                    related_elements=[attr.name]
                ))
            
            # Date/Time invariants
            if attr.type in ['Date', 'DateTime', 'Time']:
                # Start/End date relationships
                if 'start' in attr.name.lower():
                    # Look for corresponding end date
                    end_attrs = [a for a in cls.attributes 
                                if 'end' in a.name.lower() and a.type == attr.type]
                    if end_attrs:
                        invariants.append(DetectedInvariant(
                            class_name=cls.name,
                            invariant_type="date_ordering",
                            description=f"{attr.name} must be before {end_attrs[0].name}",
                            ocl_pattern="numeric_comparison",
                            confidence=0.95,
                            parameters={'attribute1': attr.name, 'operator': '<', 'attribute2': end_attrs[0].name},
                            priority="high",
                            rationale="Start date must precede end date",
                            related_elements=[attr.name, end_attrs[0].name]
                        ))
        
        return invariants
    
    def _detect_association_invariants(self, cls: Class) -> List[DetectedInvariant]:
        """Detect invariants from associations."""
        invariants = []
        
        for assoc in self.metamodel.get_all_associations():
            if assoc.source_class != cls.name:
                continue
            
            # Composition implies non-null
            if assoc.is_composition:
                invariants.append(DetectedInvariant(
                    class_name=cls.name,
                    invariant_type="composition_non_null",
                    description=f"{assoc.name} must not be null (composition)",
                    ocl_pattern="null_check",
                    confidence=1.0,
                    parameters={'attribute': assoc.name},
                    priority="critical",
                    rationale="Composition relationship requires owned object",
                    related_elements=[assoc.name, assoc.target_class]
                ))
                
                # Composition implies ownership
                invariants.append(DetectedInvariant(
                    class_name=cls.name,
                    invariant_type="unique_ownership",
                    description=f"{assoc.target_class} can only be owned by one {cls.name}",
                    ocl_pattern="uniqueness_constraint",
                    confidence=1.0,
                    parameters={'collection': assoc.name},
                    priority="critical",
                    rationale="Composition implies exclusive ownership",
                    related_elements=[assoc.name, assoc.target_class]
                ))
            
            # Collection associations
            if assoc.multiplicity and '*' in assoc.multiplicity:
                # Non-empty collection for required associations
                if not assoc.is_optional:
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="non_empty_collection",
                        description=f"{assoc.name} should not be empty",
                        ocl_pattern="size_constraint",
                        confidence=0.90,
                        parameters={'collection': assoc.name, 'operator': '>', 'value': 0},
                        priority="high",
                        rationale="Required collection must have at least one element",
                        related_elements=[assoc.name, assoc.target_class]
                    ))
        
        return invariants
    
    def _detect_multiplicity_invariants(self, cls: Class) -> List[DetectedInvariant]:
        """Detect invariants from multiplicity constraints."""
        invariants = []
        
        for assoc in self.metamodel.get_all_associations():
            if assoc.source_class != cls.name:
                continue
            
            # CRITICAL: Only generate size constraints for collections!
            # Single-valued associations (1..1, 0..1) cannot use ->size()
            if not assoc.is_collection:
                continue
            
            # Parse multiplicity
            if assoc.multiplicity:
                mult = assoc.multiplicity
                
                # Exact multiplicity (e.g., "1", "3")
                if mult.isdigit():
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="exact_multiplicity",
                        description=f"{assoc.name} must have exactly {mult} element(s)",
                        ocl_pattern="size_constraint",
                        confidence=1.0,
                        parameters={'collection': assoc.name, 'operator': '=', 'value': int(mult)},
                        priority="critical",
                        rationale=f"Multiplicity constraint requires exactly {mult}",
                        related_elements=[assoc.name, assoc.target_class]
                    ))
                
                # Range multiplicity (e.g., "1..5")
                elif '..' in mult:
                    parts = mult.split('..')
                    if parts[0].isdigit():
                        min_val = int(parts[0])
                        if min_val > 0:
                            invariants.append(DetectedInvariant(
                                class_name=cls.name,
                                invariant_type="min_multiplicity",
                                description=f"{assoc.name} must have at least {min_val} element(s)",
                                ocl_pattern="size_constraint",
                                confidence=1.0,
                                parameters={'collection': assoc.name, 'operator': '>=', 'value': min_val},
                                priority="critical",
                                rationale=f"Minimum multiplicity is {min_val}",
                                related_elements=[assoc.name, assoc.target_class]
                            ))
                    
                    if parts[1].isdigit():
                        max_val = int(parts[1])
                        invariants.append(DetectedInvariant(
                            class_name=cls.name,
                            invariant_type="max_multiplicity",
                            description=f"{assoc.name} must have at most {max_val} element(s)",
                            ocl_pattern="size_constraint",
                            confidence=1.0,
                            parameters={'collection': assoc.name, 'operator': '<=', 'value': max_val},
                            priority="critical",
                            rationale=f"Maximum multiplicity is {max_val}",
                            related_elements=[assoc.name, assoc.target_class]
                        ))
                
                # Unbounded multiplicity with lower bound (e.g., "2..*")
                elif mult.startswith(tuple(str(i) for i in range(10))) and mult.endswith('*'):
                    min_val = int(mult.split('..')[0])
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="min_multiplicity",
                        description=f"{assoc.name} must have at least {min_val} element(s)",
                        ocl_pattern="size_constraint",
                        confidence=1.0,
                        parameters={'collection': assoc.name, 'operator': '>=', 'value': min_val},
                        priority="critical",
                        rationale=f"Minimum multiplicity is {min_val}",
                        related_elements=[assoc.name, assoc.target_class]
                    ))
        
        return invariants
    
    def _detect_type_invariants(self, cls: Class) -> List[DetectedInvariant]:
        """Detect type-based invariants."""
        invariants = []
        
        # If class has parent, detect inheritance invariants
        if cls.parent_class:
            invariants.append(DetectedInvariant(
                class_name=cls.name,
                invariant_type="inheritance_type_check",
                description=f"{cls.name} is a kind of {cls.parent_class}",
                ocl_pattern="oclIsKindOf",
                confidence=1.0,
                parameters={'type': cls.parent_class},
                priority="medium",
                rationale="Inheritance relationship should be maintained",
                related_elements=[cls.parent_class]
            ))
        
        # Check for polymorphic associations
        for assoc in self.metamodel.get_all_associations():
            if assoc.source_class == cls.name:
                target_cls = self.metamodel.get_class(assoc.target_class)
                if target_cls and target_cls.parent_class:
                    invariants.append(DetectedInvariant(
                        class_name=cls.name,
                        invariant_type="polymorphic_type_check",
                        description=f"{assoc.name} elements should be valid {assoc.target_class} instances",
                        ocl_pattern="oclIsTypeOf",
                        confidence=0.80,
                        parameters={'collection': assoc.name, 'type': assoc.target_class},
                        priority="medium",
                        rationale="Polymorphic relationship requires type checking",
                        related_elements=[assoc.name, assoc.target_class]
                    ))
        
        return invariants
    
    def _detect_uniqueness_invariants(self, cls: Class) -> List[DetectedInvariant]:
        """Detect uniqueness invariants."""
        invariants = []
        
        # Detect ID attributes
        id_attrs = [attr for attr in cls.attributes 
                   if 'id' in attr.name.lower() or 
                      attr.name.lower() in ['key', 'code', 'number']]
        
        for attr in id_attrs:
            invariants.append(DetectedInvariant(
                class_name=cls.name,
                invariant_type="unique_identifier",
                description=f"{attr.name} should be unique across all {cls.name} instances",
                ocl_pattern="uniqueness_constraint",
                confidence=0.90,
                parameters={'attribute': attr.name},
                priority="high",
                rationale=f"Attribute '{attr.name}' appears to be an identifier",
                related_elements=[attr.name]
            ))
        
        # Detect collections that should have unique elements
        for assoc in self.metamodel.get_all_associations():
            if assoc.source_class == cls.name and '*' in assoc.multiplicity:
                invariants.append(DetectedInvariant(
                    class_name=cls.name,
                    invariant_type="unique_collection_elements",
                    description=f"All elements in {assoc.name} should be unique",
                    ocl_pattern="uniqueness_constraint",
                    confidence=0.75,
                    parameters={'collection': assoc.name},
                    priority="medium",
                    rationale="Collections typically should not have duplicates",
                    related_elements=[assoc.name, assoc.target_class]
                ))
        
        return invariants
    
    def _detect_domain_invariants(self, cls: Class) -> List[DetectedInvariant]:
        """Detect domain-specific invariants based on naming conventions."""
        invariants = []
        
        # User/Account classes
        if any(word in cls.name.lower() for word in ['user', 'account', 'person']):
            # Age invariant
            age_attrs = [attr for attr in cls.attributes if 'age' in attr.name.lower()]
            for attr in age_attrs:
                invariants.append(DetectedInvariant(
                    class_name=cls.name,
                    invariant_type="age_range",
                    description=f"{attr.name} should be in valid human age range",
                    ocl_pattern="range_constraint",
                    confidence=0.85,
                    parameters={'attribute': attr.name, 'min_value': 0, 'max_value': 150},
                    priority="medium",
                    rationale="Age should be in realistic range",
                    related_elements=[attr.name]
                ))
        
        # Vehicle/Car classes
        if any(word in cls.name.lower() for word in ['vehicle', 'car', 'truck']):
            # Year invariant
            year_attrs = [attr for attr in cls.attributes if 'year' in attr.name.lower()]
            for attr in year_attrs:
                invariants.append(DetectedInvariant(
                    class_name=cls.name,
                    invariant_type="vehicle_year",
                    description=f"{attr.name} should be reasonable vehicle year",
                    ocl_pattern="range_constraint",
                    confidence=0.80,
                    parameters={'attribute': attr.name, 'min_value': 1900, 'max_value': 2030},
                    priority="medium",
                    rationale="Vehicle year should be in valid range",
                    related_elements=[attr.name]
                ))
        
        # Order/Transaction classes
        if any(word in cls.name.lower() for word in ['order', 'transaction', 'booking', 'rental']):
            # Status transitions
            status_attrs = [attr for attr in cls.attributes if 'status' in attr.name.lower()]
            for attr in status_attrs:
                invariants.append(DetectedInvariant(
                    class_name=cls.name,
                    invariant_type="status_constraint",
                    description=f"{attr.name} should be valid status value",
                    ocl_pattern="membership_check",
                    confidence=0.70,
                    parameters={'attribute': attr.name, 'values': ['pending', 'active', 'completed', 'cancelled']},
                    priority="medium",
                    rationale="Status should be from valid set",
                    related_elements=[attr.name]
                ))
        
        return invariants
    
    def _detect_cross_class_invariants(self) -> List[DetectedInvariant]:
        """Detect invariants that span multiple classes."""
        invariants = []
        
        # Detect bidirectional association consistency
        for assoc1 in self.metamodel.get_all_associations():
            for assoc2 in self.metamodel.get_all_associations():
                if (assoc1.source_class == assoc2.target_class and 
                    assoc1.target_class == assoc2.source_class and
                    assoc1.name != assoc2.name):
                    
                    invariants.append(DetectedInvariant(
                        class_name=assoc1.source_class,
                        invariant_type="bidirectional_consistency",
                        description=f"Bidirectional consistency between {assoc1.name} and {assoc2.name}",
                        ocl_pattern="forall_nested",
                        confidence=0.90,
                        parameters={
                            'collection': assoc1.name,
                            'variable': 'item',
                            'condition': f"item.{assoc2.name}->includes(self)"
                        },
                        priority="high",
                        rationale="Bidirectional associations must be consistent",
                        related_elements=[assoc1.name, assoc2.name, assoc1.target_class, assoc2.target_class]
                    ))
        
        return invariants
    
    def get_invariants_by_priority(self, priority: str) -> List[DetectedInvariant]:
        """Get invariants filtered by priority."""
        return [inv for inv in self.detected_invariants if inv.priority == priority]
    
    def get_invariants_by_class(self, class_name: str) -> List[DetectedInvariant]:
        """Get all invariants for a specific class."""
        return [inv for inv in self.detected_invariants if inv.class_name == class_name]
    
    def get_invariants_by_type(self, invariant_type: str) -> List[DetectedInvariant]:
        """Get invariants of a specific type."""
        return [inv for inv in self.detected_invariants if inv.invariant_type == invariant_type]
    
    def export_invariants(self) -> Dict:
        """Export detected invariants in structured format."""
        return {
            'total_count': len(self.detected_invariants),
            'by_priority': {
                'critical': len(self.get_invariants_by_priority('critical')),
                'high': len(self.get_invariants_by_priority('high')),
                'medium': len(self.get_invariants_by_priority('medium')),
                'low': len(self.get_invariants_by_priority('low'))
            },
            'by_class': {
                cls.name: len(self.get_invariants_by_class(cls.name))
                for cls in self.metamodel.classes
            },
            'invariants': [
                {
                    'class': inv.class_name,
                    'type': inv.invariant_type,
                    'description': inv.description,
                    'pattern': inv.ocl_pattern,
                    'confidence': inv.confidence,
                    'priority': inv.priority,
                    'parameters': inv.parameters
                }
                for inv in self.detected_invariants
            ]
        }


if __name__ == "__main__":
    print("Invariant Detector Module - Ready for testing")
