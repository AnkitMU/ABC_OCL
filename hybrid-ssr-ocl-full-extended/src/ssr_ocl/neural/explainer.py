"""
Neural Explanation System for OCL Verification Results
Generates human-readable explanations of counterexamples using language models
"""
import os
import json
from typing import Dict, List, Optional
from .pattern_classifier import OCLPatternType

# Optional OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

class NeuralExplainer:
    """Neural system for explaining OCL verification results"""
    
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        self.openai_client = None
        
        # Initialize OpenAI if available and API key provided
        if use_openai and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def explain_counterexample(
        self, 
        counterexample: Dict, 
        ocl_constraint: str, 
        pattern_type: OCLPatternType,
        confidence: float = 0.0
    ) -> str:
        """Generate human-readable explanation of a counterexample"""
        
        if self.use_openai and self.openai_client:
            return self._openai_explanation(counterexample, ocl_constraint, pattern_type)
        else:
            return self._template_based_explanation(counterexample, ocl_constraint, pattern_type)
    
    def _openai_explanation(self, counterexample: Dict, ocl_constraint: str, pattern_type: OCLPatternType) -> str:
        """Generate explanation using OpenAI GPT"""
        try:
            # Remove Z3 internal info
            clean_counterexample = {k: v for k, v in counterexample.items() if k != '_z3_info'}
            
            prompt = f"""
You are an expert in OCL (Object Constraint Language) and formal verification. 
Explain why this OCL constraint is violated by providing a clear, human-readable explanation.

OCL Constraint: {ocl_constraint}
Pattern Type: {pattern_type.value}
Counterexample: {json.dumps(clean_counterexample, indent=2)}

Provide a concise explanation of:
1. What the constraint is trying to ensure
2. How the counterexample violates it
3. What specific values cause the violation

Keep the explanation under 150 words and use simple, clear language.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"  OpenAI explanation failed: {e}")
            return self._template_based_explanation(counterexample, ocl_constraint, pattern_type)
    
    def _template_based_explanation(self, counterexample: Dict, ocl_constraint: str, pattern_type: OCLPatternType) -> str:
        """Generate explanation using templates"""
        
        # Remove Z3 internal info for analysis
        clean_counterexample = {k: v for k, v in counterexample.items() if k != '_z3_info'}
        
        if pattern_type == OCLPatternType.PAIRWISE_UNIQUENESS:
            return self._explain_pairwise_uniqueness(clean_counterexample, ocl_constraint)
        elif pattern_type == OCLPatternType.EXACT_COUNT_SELECTION:
            return self._explain_exact_count_selection(clean_counterexample, ocl_constraint)
        elif pattern_type == OCLPatternType.SIZE_CONSTRAINT:
            return self._explain_size_constraint(clean_counterexample, ocl_constraint)
        elif pattern_type == OCLPatternType.NUMERIC_COMPARISON:
            return self._explain_numeric_comparison(clean_counterexample, ocl_constraint)
        elif pattern_type == OCLPatternType.NULL_CHECK:
            return self._explain_null_check(clean_counterexample, ocl_constraint)
        elif pattern_type == OCLPatternType.GLOBAL_COLLECTION:
            return self._explain_global_collection(clean_counterexample, ocl_constraint)
        else:
            return self._explain_generic(clean_counterexample, ocl_constraint)
    
    def _explain_pairwise_uniqueness(self, counterexample: Dict, constraint: str) -> str:
        """Explain pairwise uniqueness violations"""
        # Find duplicate values
        id_vars = [k for k in counterexample.keys() if k.startswith(('hold_id_', 'student_id_', 'all_id_'))]
        
        if id_vars:
            # Group by value to find duplicates
            value_groups = {}
            for var in id_vars:
                value = counterexample[var]
                if value not in value_groups:
                    value_groups[value] = []
                value_groups[value].append(var)
            
            duplicates = {v: vars for v, vars in value_groups.items() if len(vars) > 1}
            
            if duplicates:
                explanation = " **Uniqueness Violation**: The constraint requires all IDs to be unique, but duplicates were found:\\n\\n"
                for value, vars in duplicates.items():
                    var_names = [v.replace('_', ' ') for v in vars]
                    explanation += f"• **ID {value}** appears in: {', '.join(var_names)}\\n"
                explanation += f"\\n **Constraint**: `{constraint}`\\n"
                explanation += "🔧 **Fix**: Ensure each entity has a unique identifier."
                return explanation
        
        return f" **Uniqueness Violation**: Multiple entities have the same ID value. Constraint: `{constraint}`"
    
    def _explain_exact_count_selection(self, counterexample: Dict, constraint: str) -> str:
        """Explain exact count selection violations"""
        if 'self_id' in counterexample:
            self_id = counterexample['self_id']
            
            # Count matching IDs
            id_vars = [k for k in counterexample.keys() if k.startswith(('hold_id_', 'all_id_'))]
            matches = [k for k in id_vars if counterexample[k] == self_id]
            
            if len(matches) == 0:
                return f" **Missing Entity**: No entity with ID '{self_id}' was found. The constraint expects exactly one match. Constraint: `{constraint}`"
            elif len(matches) > 1:
                return f" **Duplicate Entities**: Found {len(matches)} entities with ID '{self_id}': {matches}. The constraint expects exactly one match. Constraint: `{constraint}`"
        
        return f" **Count Violation**: The constraint expects exactly one matching entity, but a different count was found. Constraint: `{constraint}`"
    
    def _explain_size_constraint(self, counterexample: Dict, constraint: str) -> str:
        """Explain size constraint violations"""
        size_vars = [k for k in counterexample.keys() if k.endswith('_size')]
        
        if size_vars:
            size_var = size_vars[0]
            actual_size = counterexample[size_var]
            collection_name = size_var.replace('_size', '')
            
            if 'size() > 0' in constraint:
                return f" **Empty Collection**: The {collection_name} collection is empty (size = {actual_size}), but the constraint requires it to have at least one element. Constraint: `{constraint}`"
            elif 'size() <=' in constraint:
                return f" **Collection Too Large**: The {collection_name} collection has {actual_size} elements, exceeding the allowed limit. Constraint: `{constraint}`"
            else:
                return f" **Size Violation**: The {collection_name} collection size ({actual_size}) doesn't meet the constraint requirements. Constraint: `{constraint}`"
        
        return f" **Size Constraint Violation**: Collection size doesn't meet requirements. Constraint: `{constraint}`"
    
    def _explain_numeric_comparison(self, counterexample: Dict, constraint: str) -> str:
        """Explain numeric comparison violations"""
        for var, value in counterexample.items():
            if isinstance(value, (int, float)):
                if var == 'age':
                    if value < 16:
                        return f" **Age Violation**: Age is {value} years, but the constraint requires age ≥ 16. Constraint: `{constraint}`"
                elif var == 'gpa':
                    if value < 0 or value > 4:
                        return f" **GPA Out of Range**: GPA is {value}, but must be between 0.0 and 4.0. Constraint: `{constraint}`"
                elif var == 'credits':
                    return f" **Invalid Credits**: Credits value is {value}, which violates the constraint requirements. Constraint: `{constraint}`"
                elif var == 'experienceYears':
                    if value < 0:
                        return f" **Negative Experience**: Experience years is {value}, but cannot be negative. Constraint: `{constraint}`"
        
        return f" **Numeric Constraint Violation**: A numeric value doesn't meet the constraint requirements. Constraint: `{constraint}`"
    
    def _explain_null_check(self, counterexample: Dict, constraint: str) -> str:
        """Explain null check violations"""
        for var, value in counterexample.items():
            if var.endswith('_present'):
                entity_name = var.replace('_present', '')
                if value is False and '<> null' in constraint:
                    return f" **Missing Required Entity**: The {entity_name} is null/missing, but the constraint requires it to be present. Constraint: `{constraint}`"
                elif value is True and '= null' in constraint:
                    return f" **Unexpected Entity**: The {entity_name} is present, but the constraint expects it to be null. Constraint: `{constraint}`"
        
        return f" **Null Check Violation**: Entity presence doesn't match constraint requirements. Constraint: `{constraint}`"
    
    def _explain_global_collection(self, counterexample: Dict, constraint: str) -> str:
        """Explain global collection violations"""
        if 'allInstances' in constraint:
            return f" **Global Collection Violation**: The constraint checks all instances globally, but the current state violates the requirement. Multiple entities may have conflicting values. Constraint: `{constraint}`"
        
        return f" **Collection Analysis Violation**: The constraint analyzes collection properties that don't meet requirements. Constraint: `{constraint}`"
    
    def _explain_generic(self, counterexample: Dict, constraint: str) -> str:
        """Generic explanation fallback"""
        if counterexample:
            var_summary = []
            for key, value in list(counterexample.items())[:3]:  # Show first 3 variables
                var_summary.append(f"{key}: {value}")
            
            summary = ", ".join(var_summary)
            if len(counterexample) > 3:
                summary += f" (and {len(counterexample) - 3} more)"
            
            return f" **Constraint Violation**: The constraint is violated by the following values: {summary}. Constraint: `{constraint}`"
        else:
            return f" **Constraint Violation**: The constraint `{constraint}` is not satisfied under the current conditions."
    
    def explain_verification_result(self, result_status: str, counterexample: Optional[Dict] = None) -> str:
        """Explain overall verification result"""
        if result_status == "UNSAT":
            return " **Constraint Valid**: The constraint cannot be violated - it holds for all possible instances within the given scope."
        elif result_status == "SAT":
            if counterexample:
                return " **Constraint Violable**: The constraint can be violated. See counterexample details above."
            else:
                return " **Constraint Violable**: The constraint can be violated, but no specific counterexample was generated."
        elif result_status == "UNKNOWN":
            return "❓ **Unknown Result**: The solver could not determine if the constraint can be violated within the time limit."
        else:
            return f"❓ **Unexpected Status**: Verification returned status '{result_status}'"

# Global instance
_explainer_instance = None

def get_neural_explainer(use_openai: bool = False) -> NeuralExplainer:
    """Get singleton neural explainer instance"""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = NeuralExplainer(use_openai=use_openai)
    return _explainer_instance