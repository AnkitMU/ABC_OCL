"""
OCL Generator Engine - Orchestrates constraint generation
"""

from typing import Dict, Optional, Any
from datetime import datetime

from modules.core.models import (
    Pattern, OCLConstraint, ValidationResult, Metamodel
)
from modules.synthesis.pattern_engine.pattern_registry import PatternRegistry, get_registry


class OCLGenerator:
    """
    Main OCL generation engine
    
    Coordinates pattern lookup, parameter validation, and OCL generation
    """
    
    def __init__(self, pattern_registry: Optional[PatternRegistry] = None,
                 metamodel: Optional[Metamodel] = None):
        """
        Initialize OCL generator
        
        Args:
            pattern_registry: Pattern registry (uses singleton if None)
            metamodel: Metamodel for validation (optional)
        """
        self.registry = pattern_registry or get_registry()
        self.metamodel = metamodel
        self.generation_count = 0
    
    def set_metamodel(self, metamodel: Metamodel):
        """Set metamodel for validation"""
        self.metamodel = metamodel
    
    def generate(self, pattern_id: str, context: str, 
                 params: Dict[str, Any]) -> OCLConstraint:
        """
        Generate OCL constraint from pattern ID
        
        Args:
            pattern_id: Pattern identifier
            context: Context class name
            params: Parameter values
            
        Returns:
            OCL constraint with metadata
            
        Raises:
            ValueError: If pattern not found or parameters invalid
        """
        # Get pattern
        pattern = self.registry.get_pattern(pattern_id)
        if pattern is None:
            raise ValueError(f"Pattern not found: {pattern_id}")
        
        return self.generate_from_pattern(pattern, context, params)
    
    def generate_from_pattern(self, pattern: Pattern, context: str,
                              params: Dict[str, Any]) -> OCLConstraint:
        """
        Generate OCL constraint from pattern object
        
        Args:
            pattern: Pattern object
            context: Context class name
            params: Parameter values
            
        Returns:
            OCL constraint with metadata
            
        Raises:
            ValueError: If parameters invalid or generation fails
        """
        # Validate context
        if self.metamodel and pattern.requires_context:
            if not self.metamodel.get_class(context):
                raise ValueError(f"Context class not found in metamodel: {context}")
        
        # Validate parameters
        is_valid, error_msg = pattern.validate_parameters(params)
        if not is_valid:
            raise ValueError(f"Parameter validation failed: {error_msg}")
        
        # Generate OCL
        try:
            ocl = pattern.generate_ocl(context, params)
        except Exception as e:
            raise ValueError(f"OCL generation failed: {e}")
        
        # Create constraint object
        constraint = OCLConstraint(
            ocl=ocl,
            pattern_id=pattern.id,
            pattern_name=pattern.name,
            context=context,
            parameters=params.copy(),
            confidence=1.0,
            timestamp=datetime.now().isoformat(),
            metadata={
                'category': pattern.category.value,
                'complexity': pattern.complexity,
                'generation_number': self.generation_count
            }
        )
        
        self.generation_count += 1
        
        return constraint
    
    def validate_parameters(self, pattern_id: str, context: str,
                           params: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters for a pattern
        
        Args:
            pattern_id: Pattern identifier
            context: Context class name
            params: Parameter values
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        
        # Get pattern
        pattern = self.registry.get_pattern(pattern_id)
        if pattern is None:
            result.add_error(f"Pattern not found: {pattern_id}")
            return result
        
        # Validate context
        if self.metamodel and pattern.requires_context:
            if not self.metamodel.get_class(context):
                result.add_error(f"Context class not found: {context}")
                return result
        
        # Validate parameters
        is_valid, error_msg = pattern.validate_parameters(params)
        if not is_valid:
            result.add_error(error_msg)
            return result
        
        # Additional validation based on metamodel
        if self.metamodel:
            result = self._validate_with_metamodel(pattern, context, params, result)
        
        return result
    
    def _validate_with_metamodel(self, pattern: Pattern, context: str,
                                 params: Dict[str, Any],
                                 result: ValidationResult) -> ValidationResult:
        """
        Additional validation using metamodel
        
        Args:
            pattern: Pattern
            context: Context class
            params: Parameters
            result: Validation result to update
            
        Returns:
            Updated validation result
        """
        # Check if specified collections/attributes exist
        for param in pattern.parameters:
            if param.name in params:
                value = params[param.name]
                
                # Validate collection references
                if param.options == "collection_associations":
                    assocs = self.metamodel.get_collection_associations(context)
                    assoc_names = [a.ref_name for a in assocs]
                    if value not in assoc_names:
                        result.add_error(
                            f"Collection '{value}' not found in {context}. "
                            f"Available: {', '.join(assoc_names)}"
                        )
                
                # Validate attribute references
                elif param.options == "attributes":
                    attrs = self.metamodel.get_attributes_for(context)
                    attr_names = [a.name for a in attrs]
                    if value not in attr_names:
                        result.add_warning(
                            f"Attribute '{value}' not found in {context}. "
                            f"Available: {', '.join(attr_names)}"
                        )
        
        return result
    
    def generate_batch(self, specifications: list) -> list:
        """
        Generate multiple constraints
        
        Args:
            specifications: List of (pattern_id, context, params) tuples
            
        Returns:
            List of OCL constraints
        """
        constraints = []
        
        for spec in specifications:
            try:
                if isinstance(spec, tuple) and len(spec) == 3:
                    pattern_id, context, params = spec
                    constraint = self.generate(pattern_id, context, params)
                    constraints.append(constraint)
                else:
                    raise ValueError(f"Invalid specification format: {spec}")
            except Exception as e:
                print(f"Failed to generate constraint: {e}")
        
        return constraints
    
    def format_constraints(self, constraints: list, 
                          include_comments: bool = True) -> str:
        """
        Format multiple constraints for output
        
        Args:
            constraints: List of OCL constraints
            include_comments: Whether to include comments
            
        Returns:
            Formatted OCL string
        """
        output_lines = []
        
        for i, constraint in enumerate(constraints, 1):
            if include_comments:
                output_lines.append(f"-- Constraint {i}: {constraint.pattern_name}")
                output_lines.append(f"-- Generated: {constraint.timestamp}")
                output_lines.append(f"-- Pattern: {constraint.pattern_id}")
            
            output_lines.append(constraint.ocl)
            output_lines.append("")  # Empty line between constraints
        
        return "\n".join(output_lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'total_patterns': self.registry.get_pattern_count(),
            'constraints_generated': self.generation_count,
            'has_metamodel': self.metamodel is not None,
            'categories': len(self.registry.get_all_categories())
        }


# Convenience function
def generate_constraint(pattern_id: str, context: str, params: Dict[str, Any],
                       metamodel: Optional[Metamodel] = None) -> OCLConstraint:
    """
    Quick function to generate a single constraint
    
    Args:
        pattern_id: Pattern identifier
        context: Context class
        params: Parameter values
        metamodel: Optional metamodel for validation
        
    Returns:
        OCL constraint
    """
    generator = OCLGenerator(metamodel=metamodel)
    return generator.generate(pattern_id, context, params)


# Main function for testing
def main():
    """Test OCL generator"""
    from modules.semantic.metamodel.xmi_extractor import extract_metamodel
    import sys
    
    try:
        # Load metamodel if XMI file provided
        metamodel = None
        if len(sys.argv) > 1:
            xmi_file = sys.argv[1]
            print(f"Loading metamodel from {xmi_file}...")
            metamodel = extract_metamodel(xmi_file)
        
        # Create generator
        generator = OCLGenerator(metamodel=metamodel)
        
        print("\n" + "="*60)
        print("OCL GENERATOR TEST")
        print("="*60)
        
        # Test 1: Size constraint
        print("\nTest 1: Size Constraint")
        try:
            constraint = generator.generate(
                pattern_id="size_constraint",
                context="Branch",
                params={
                    "collection": "vehicles",
                    "operator": ">=",
                    "value": 2
                }
            )
            print(f"Generated:\n{constraint.ocl}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 2: Uniqueness constraint
        print("\nTest 2: Uniqueness Constraint")
        try:
            constraint = generator.generate(
                pattern_id="uniqueness_constraint",
                context="Branch",
                params={
                    "collection": "vehicles",
                    "iterator": "v",
                    "attribute": "licencePlate"
                }
            )
            print(f"Generated:\n{constraint.ocl}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 3: Batch generation
        print("\nTest 3: Batch Generation")
        specifications = [
            ("size_constraint", "Branch", {
                "collection": "vehicles", "operator": ">=", "value": 2
            }),
            ("numeric_comparison", "Rental", {
                "attr1": "startDate", "operator": "<", "attr2": "endDate"
            })
        ]
        
        constraints = generator.generate_batch(specifications)
        print(f"Generated {len(constraints)} constraints")
        print("\nFormatted output:")
        print(generator.format_constraints(constraints))
        
        # Statistics
        print("\nGenerator Statistics:")
        stats = generator.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
