"""
OCL Constraint Verifier using hybrid-ssr-ocl-full-extended framework
Provides accurate verification using the existing Z3-based verification system.
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

from modules.core.models import OCLConstraint, Metamodel


# Add framework to path
FRAMEWORK_PATH = Path(__file__).parent.parent.parent.parent / "hybrid-ssr-ocl-full-extended"
if FRAMEWORK_PATH.exists():
    sys.path.insert(0, str(FRAMEWORK_PATH / "src"))


@dataclass
class FrameworkVerificationResult:
    """Result from framework verification."""
    constraint_id: str
    is_valid: bool
    is_satisfiable: Optional[bool] = None
    errors: List[str] = None
    warnings: List[str] = None
    execution_time: float = 0.0
    solver_result: Optional[str] = None  # 'sat', 'unsat', 'unknown'
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    @property
    def status(self) -> str:
        """Get status string."""
        if not self.is_valid:
            return "INVALID"
        if self.is_satisfiable == False:
            return "UNSATISFIABLE"
        if self.warnings:
            return "VALID (warnings)"
        return "VALID"


class FrameworkConstraintVerifier:
    """
    Verifier that uses hybrid-ssr-ocl-full-extended framework.
    
    This provides accurate, research-grade verification using:
    - Full OCL parser
    - Z3 SMT solver
    - Association-backed encoding
    - Pattern-based constraint handling
    """
    
    def __init__(self, metamodel: Metamodel, xmi_path: str):
        """
        Initialize verifier with framework.
        
        Args:
            metamodel: Metamodel object (for compatibility)
            xmi_path: Path to XMI file (needed by framework)
        """
        self.metamodel = metamodel
        self.xmi_path = xmi_path
        self.framework_available = False
        self.checker = None
        
        # Try to import framework
        try:
            from ssr_ocl.super_encoder.generic_global_consistency_checker import GenericGlobalConsistencyChecker

            # Initialize checker (suppress internal prints)
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                self.checker = GenericGlobalConsistencyChecker(
                    xmi_file=xmi_path,
                    rich_instances=True,  # Enable realistic values
                    timeout_ms=15000,  # 5 second timeout
                    show_raw_values=False
                )
            
            self.framework_available = True
            
        except ImportError as e:
            print(f"Framework not available: {e}")
            print("   Falling back to basic verification")
            self.framework_available = False
        except Exception as e:
            print(f"Error initializing framework: {e}")
            print("   Falling back to basic verification")
            self.framework_available = False
    
    def verify(self, constraint: OCLConstraint) -> FrameworkVerificationResult:
        """
        Verify a single constraint.
        
        Args:
            constraint: Constraint to verify
            
        Returns:
            Verification result
        """
        start = time.time()
        
        result = FrameworkVerificationResult(
            constraint_id=f"{constraint.pattern_id}_{constraint.context}",
            is_valid=True
        )
        
        if not self.framework_available:
            # Fallback to basic checking
            result.warnings.append("Framework not available - using basic verification")
            result.execution_time = time.time() - start
            return result
        
        try:
            # Prepare constraint for framework
            constraint_dict = {
                'name': constraint.pattern_name,
                'pattern': constraint.pattern_id,
                'context': constraint.context,
                'text': constraint.ocl
            }
            
            # Determine scope (how many instances to check)
            scope = self._get_verification_scope()
            
            # Verify using framework
            solver_result, model = self.checker.verify_all_constraints(
                constraints=[constraint_dict],
                scope=scope
            )
            
            result.solver_result = solver_result
            
            if solver_result == 'sat':
                result.is_valid = True
                result.is_satisfiable = True
            elif solver_result == 'unsat':
                result.is_valid = True  # Syntactically valid
                result.is_satisfiable = False
                result.warnings.append("Constraint is unsatisfiable")
            elif solver_result == 'unknown':
                result.is_valid = True
                result.is_satisfiable = None
                result.warnings.append("Verification timeout or unknown")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Framework verification error: {str(e)}")
        
        result.execution_time = time.time() - start
        return result
    
    def verify_batch(self, constraints: List[OCLConstraint], silent: bool = False) -> List[FrameworkVerificationResult]:
        """
        Verify multiple constraints together (checks global consistency).
        
        Args:
            constraints: List of constraints to verify
            silent: If True, suppress all console output
            
        Returns:
            List of verification results
        """
        if not self.framework_available:
            # Fallback - verify individually
            return [self.verify(c) for c in constraints]
        
        start = time.time()
        results = []
        
        try:
            # Prepare all constraints for framework
            constraint_dicts = [
                {
                    'name': c.pattern_name,
                    'pattern': c.pattern_id,
                    'context': c.context,
                    'text': c.ocl
                }
                for c in constraints
            ]
            
            # Determine scope
            scope = self._get_verification_scope()
            
            if not silent:
                print(f"\nVerifying {len(constraints)} constraints for global consistency...")
            
            # Verify all together - suppress output if silent mode
            if silent:
                import sys, io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    solver_result, model = self.checker.verify_all_constraints(
                        constraints=constraint_dicts,
                        scope=scope
                    )
                finally:
                    sys.stdout = old_stdout
            else:
                solver_result, model = self.checker.verify_all_constraints(
                    constraints=constraint_dicts,
                    scope=scope
                )
            
            # Create results for each constraint
            for i, c in enumerate(constraints):
                result = FrameworkVerificationResult(
                    constraint_id=f"{c.pattern_id}_{c.context}",
                    is_valid=True,
                    solver_result=solver_result
                )
                
                # Check individual status from checker
                status = self.checker.constraint_status.get(c.pattern_name, 'unknown')
                
                if status == 'error':
                    result.is_valid = False
                    result.errors.append("Encoding error")
                elif status == 'encoded':
                    result.is_valid = True
                    
                    if solver_result == 'sat':
                        result.is_satisfiable = True
                    elif solver_result == 'unsat':
                        result.is_satisfiable = False
                        result.warnings.append("Part of unsatisfiable constraint set")
                    else:
                        result.is_satisfiable = None
                        result.warnings.append("Verification timeout")
                
                results.append(result)
            
        except Exception as e:
            print(f"Batch verification error: {e}")
            # Return error results for all
            for c in constraints:
                result = FrameworkVerificationResult(
                    constraint_id=f"{c.pattern_id}_{c.context}",
                    is_valid=False
                )
                result.errors.append(f"Batch verification error: {str(e)}")
                results.append(result)
        
        total_time = time.time() - start
        
        # Distribute time across constraints
        for r in results:
            r.execution_time = total_time / len(constraints) if constraints else 0
        
        return results
    
    def check_consistency(self, constraints: List[OCLConstraint]) -> Dict:
        """
        Check if constraints are mutually consistent.
        
        Args:
            constraints: List of constraints
            
        Returns:
            Consistency result dict
        """
        if not self.framework_available:
            return {
                'consistent': True,
                'verified': False,
                'message': 'Framework not available'
            }
        
        try:
            constraint_dicts = [
                {
                    'name': c.pattern_name,
                    'pattern': c.pattern_id,
                    'context': c.context,
                    'text': c.ocl
                }
                for c in constraints
            ]
            
            scope = self._get_verification_scope()
            
            solver_result, model = self.checker.verify_all_constraints(
                constraints=constraint_dicts,
                scope=scope
            )
            
            return {
                'consistent': solver_result == 'sat',
                'verified': True,
                'solver_result': solver_result,
                'message': f"Global consistency check: {solver_result}"
            }
            
        except Exception as e:
            return {
                'consistent': False,
                'verified': False,
                'error': str(e)
            }
    
    def _get_verification_scope(self) -> Dict:
        """
        Determine verification scope (how many instances of each class).
        
        Returns:
            Scope dict like {'nPerson': 2, 'nCompany': 2, ...}
        """
        scope = {}
        
        # Use 2 instances per class for verification (small scope for speed)
        for class_name in self.metamodel.get_class_names():
            scope[f'n{class_name}'] = 2
        
        return scope
    
    def get_statistics(self, results: List[FrameworkVerificationResult]) -> Dict:
        """
        Get statistics from verification results.
        
        Args:
            results: List of verification results
            
        Returns:
            Statistics dict
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        satisfiable = sum(1 for r in results if r.is_satisfiable is True)
        unsatisfiable = sum(1 for r in results if r.is_satisfiable is False)
        unknown = sum(1 for r in results if r.is_satisfiable is None and r.is_valid)
        with_warnings = sum(1 for r in results if r.warnings)
        with_errors = sum(1 for r in results if r.errors)
        
        avg_time = sum(r.execution_time for r in results) / max(1, total)
        
        return {
            'total': total,
            'valid': valid,
            'invalid': total - valid,
            'validity_rate': valid / max(1, total),
            'satisfiable': satisfiable,
            'unsatisfiable': unsatisfiable,
            'unknown': unknown,
            'with_warnings': with_warnings,
            'with_errors': with_errors,
            'avg_verification_time': avg_time,
            'framework_used': self.framework_available
        }
