"""
Implication Analyzer Module
Detects logical implications between constraints using semantic reasoning.
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from modules.core.models import OCLConstraint, Class, Attribute, Association


class ImplicationStrength(Enum):
    """Strength of implication between constraints"""
    STRONG = "strong"          # Always implies
    WEAK = "weak"              # Sometimes implies
    CONDITIONAL = "conditional" # Implies under conditions
    NONE = "none"              # No implication


@dataclass
class Implication:
    """Represents an implication between two constraints"""
    premise: OCLConstraint
    conclusion: OCLConstraint
    strength: ImplicationStrength
    conditions: List[str]  # Conditions under which implication holds
    confidence: float      # 0.0 to 1.0
    explanation: str


class ImplicationAnalyzer:
    """
    Analyzes logical implications between OCL constraints.
    
    Features:
    - Detects when one constraint implies another
    - Identifies subsumption relationships
    - Finds constraint hierarchies
    - Supports conditional implications
    """
    
    def __init__(self):
        self.implication_cache: Dict[Tuple[str, str], Implication] = {}
        self._init_implication_patterns()
    
    def _init_implication_patterns(self):
        """Initialize known implication patterns"""
        self.patterns = {
            # Collection size patterns
            'size_zero_implies_empty': {
                'premise_pattern': r'->size\(\)\s*=\s*0',
                'conclusion_pattern': r'->isEmpty\(\)',
                'strength': ImplicationStrength.STRONG,
                'confidence': 1.0
            },
            'notempty_implies_size_gt_zero': {
                'premise_pattern': r'->notEmpty\(\)',
                'conclusion_pattern': r'->size\(\)\s*>\s*0',
                'strength': ImplicationStrength.STRONG,
                'confidence': 1.0
            },
            # Bound implications
            'lower_bound_implies_weaker': {
                'premise_pattern': r'>=\s*(\d+)',
                'conclusion_pattern': r'>=\s*(\d+)',  # with smaller number
                'strength': ImplicationStrength.STRONG,
                'confidence': 0.95
            },
            # Type implications
            'subtype_implies_supertype': {
                'premise_pattern': r'oclIsKindOf\((\w+)\)',
                'conclusion_pattern': r'oclIsKindOf\((\w+)\)',
                'strength': ImplicationStrength.STRONG,
                'confidence': 0.9
            }
        }
    
    def analyze_implication(
        self,
        constraint1: OCLConstraint,
        constraint2: OCLConstraint,
        metamodel: Optional[Dict] = None
    ) -> Implication:
        """
        Analyze if constraint1 implies constraint2.
        
        Args:
            constraint1: Premise constraint
            constraint2: Potential conclusion constraint
            metamodel: Optional metamodel for context
            
        Returns:
            Implication object with analysis results
        """
        # Use timestamp or hash as cache key since constraint_id doesn't exist
        cache_key = (constraint1.timestamp, constraint2.timestamp)
        if cache_key in self.implication_cache:
            return self.implication_cache[cache_key]
        
        # Check various types of implications
        implications = []
        
        # 1. Syntactic implication (identical constraints)
        if self._check_syntactic_equality(constraint1, constraint2):
            impl = Implication(
                premise=constraint1,
                conclusion=constraint2,
                strength=ImplicationStrength.STRONG,
                conditions=[],
                confidence=1.0,
                explanation="Constraints are syntactically identical"
            )
            implications.append(impl)
        
        # 2. Pattern-based implication
        pattern_impl = self._check_pattern_implication(constraint1, constraint2)
        if pattern_impl:
            implications.append(pattern_impl)
        
        # 3. Collection operation implications
        collection_impl = self._check_collection_implication(constraint1, constraint2)
        if collection_impl:
            implications.append(collection_impl)
        
        # 4. Numeric bound implications
        numeric_impl = self._check_numeric_implication(constraint1, constraint2)
        if numeric_impl:
            implications.append(numeric_impl)
        
        # 5. Logical operator implications
        logical_impl = self._check_logical_implication(constraint1, constraint2)
        if logical_impl:
            implications.append(logical_impl)
        
        # Select strongest implication
        if implications:
            result = max(implications, key=lambda i: (i.strength.value, i.confidence))
        else:
            result = Implication(
                premise=constraint1,
                conclusion=constraint2,
                strength=ImplicationStrength.NONE,
                conditions=[],
                confidence=0.0,
                explanation="No implication detected"
            )
        
        self.implication_cache[cache_key] = result
        return result
    
    def _check_syntactic_equality(
        self,
        c1: OCLConstraint,
        c2: OCLConstraint
    ) -> bool:
        """Check if constraints are syntactically equal"""
        return c1.ocl.strip() == c2.ocl.strip()
    
    def _check_pattern_implication(
        self,
        c1: OCLConstraint,
        c2: OCLConstraint
    ) -> Optional[Implication]:
        """Check pattern-based implications"""
        expr1 = c1.ocl.strip()
        expr2 = c2.ocl.strip()
        
        # size() = 0 implies isEmpty()
        if '->size()' in expr1 and '= 0' in expr1 and '->isEmpty()' in expr2:
            return Implication(
                premise=c1,
                conclusion=c2,
                strength=ImplicationStrength.STRONG,
                conditions=[],
                confidence=1.0,
                explanation="size() = 0 always implies isEmpty()"
            )
        
        # notEmpty() implies size() > 0
        if '->notEmpty()' in expr1 and '->size()' in expr2 and '> 0' in expr2:
            return Implication(
                premise=c1,
                conclusion=c2,
                strength=ImplicationStrength.STRONG,
                conditions=[],
                confidence=1.0,
                explanation="notEmpty() always implies size() > 0"
            )
        
        # forAll(...) implies exists(...)
        if '->forAll(' in expr1 and '->exists(' in expr2:
            # Extract conditions
            if self._extract_lambda_body(expr1) == self._extract_lambda_body(expr2):
                return Implication(
                    premise=c1,
                    conclusion=c2,
                    strength=ImplicationStrength.STRONG,
                    conditions=["Collection is non-empty"],
                    confidence=0.9,
                    explanation="forAll(condition) implies exists(condition) for non-empty collections"
                )
        
        return None
    
    def _check_collection_implication(
        self,
        c1: OCLConstraint,
        c2: OCLConstraint
    ) -> Optional[Implication]:
        """Check collection operation implications"""
        expr1 = c1.ocl
        expr2 = c2.ocl
        
        # includesAll implies includes
        if '->includesAll(' in expr1 and '->includes(' in expr2:
            return Implication(
                premise=c1,
                conclusion=c2,
                strength=ImplicationStrength.STRONG,
                conditions=["Element is in the includesAll set"],
                confidence=0.85,
                explanation="includesAll(set) implies includes(element) for each element in set"
            )
        
        # excludesAll implies excludes
        if '->excludesAll(' in expr1 and '->excludes(' in expr2:
            return Implication(
                premise=c1,
                conclusion=c2,
                strength=ImplicationStrength.STRONG,
                conditions=["Element is in the excludesAll set"],
                confidence=0.85,
                explanation="excludesAll(set) implies excludes(element) for each element in set"
            )
        
        return None
    
    def _check_numeric_implication(
        self,
        c1: OCLConstraint,
        c2: OCLConstraint
    ) -> Optional[Implication]:
        """Check numeric bound implications"""
        import re
        
        expr1 = c1.ocl
        expr2 = c2.ocl
        
        # Extract numeric comparisons
        pattern = r'(\w+(?:\.\w+)*)\s*(>=|<=|>|<|=)\s*(\d+)'
        
        match1 = re.search(pattern, expr1)
        match2 = re.search(pattern, expr2)
        
        if match1 and match2:
            var1, op1, val1 = match1.groups()
            var2, op2, val2 = match2.groups()
            
            # Same variable
            if var1 == var2:
                v1, v2 = int(val1), int(val2)
                
                # x >= 10 implies x >= 5
                if op1 == '>=' and op2 == '>=' and v1 > v2:
                    return Implication(
                        premise=c1,
                        conclusion=c2,
                        strength=ImplicationStrength.STRONG,
                        conditions=[],
                        confidence=1.0,
                        explanation=f"Lower bound {v1} is stronger than {v2}"
                    )
                
                # x <= 5 implies x <= 10
                if op1 == '<=' and op2 == '<=' and v1 < v2:
                    return Implication(
                        premise=c1,
                        conclusion=c2,
                        strength=ImplicationStrength.STRONG,
                        conditions=[],
                        confidence=1.0,
                        explanation=f"Upper bound {v1} is stronger than {v2}"
                    )
        
        return None
    
    def _check_logical_implication(
        self,
        c1: OCLConstraint,
        c2: OCLConstraint
    ) -> Optional[Implication]:
        """Check logical operator implications"""
        expr1 = c1.ocl
        expr2 = c2.ocl
        
        # A and B implies A
        if ' and ' in expr1:
            parts = [p.strip() for p in expr1.split(' and ')]
            if expr2.strip() in parts:
                return Implication(
                    premise=c1,
                    conclusion=c2,
                    strength=ImplicationStrength.STRONG,
                    conditions=[],
                    confidence=1.0,
                    explanation="Conjunction implies each conjunct"
                )
        
        # A implies A or B
        if ' or ' in expr2:
            parts = [p.strip() for p in expr2.split(' or ')]
            if expr1.strip() in parts:
                return Implication(
                    premise=c1,
                    conclusion=c2,
                    strength=ImplicationStrength.STRONG,
                    conditions=[],
                    confidence=1.0,
                    explanation="Any statement implies its disjunction with others"
                )
        
        return None
    
    def _extract_lambda_body(self, expr: str) -> str:
        """Extract the body of a lambda expression"""
        import re
        match = re.search(r'\|\s*(.+?)\s*\)', expr)
        return match.group(1) if match else ""
    
    def find_all_implications(
        self,
        constraints: List[OCLConstraint],
        min_confidence: float = 0.7
    ) -> List[Implication]:
        """
        Find all implications among a set of constraints.
        
        Args:
            constraints: List of constraints to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected implications
        """
        implications = []
        
        for i, c1 in enumerate(constraints):
            for j, c2 in enumerate(constraints):
                if i != j:
                    impl = self.analyze_implication(c1, c2)
                    if impl.strength != ImplicationStrength.NONE and \
                       impl.confidence >= min_confidence:
                        implications.append(impl)
        
        return implications
    
    def build_implication_graph(
        self,
        constraints: List[OCLConstraint]
    ) -> Dict[str, List[str]]:
        """
        Build a directed graph of constraint implications.
        
        Returns:
            Adjacency list where edges represent implications
        """
        graph = {c.constraint_id: [] for c in constraints}
        
        implications = self.find_all_implications(constraints)
        
        for impl in implications:
            if impl.strength in [ImplicationStrength.STRONG, ImplicationStrength.CONDITIONAL]:
                graph[impl.premise.constraint_id].append(impl.conclusion.constraint_id)
        
        return graph
    
    def find_subsumption_chains(
        self,
        constraints: List[OCLConstraint]
    ) -> List[List[str]]:
        """
        Find chains of constraints where each implies the next.
        
        Returns:
            List of constraint ID chains
        """
        graph = self.build_implication_graph(constraints)
        chains = []
        
        def dfs(node: str, path: List[str], visited: Set[str]):
            if node in visited:
                if len(path) > 1:
                    chains.append(path[:])
                return
            
            visited.add(node)
            path.append(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    dfs(neighbor, path, visited)
            
            if len(path) > 1:
                chains.append(path[:])
            
            path.pop()
            visited.remove(node)
        
        for constraint in constraints:
            dfs(constraint.constraint_id, [], set())
        
        # Remove duplicate chains
        unique_chains = []
        seen = set()
        for chain in chains:
            chain_tuple = tuple(chain)
            if chain_tuple not in seen:
                seen.add(chain_tuple)
                unique_chains.append(chain)
        
        return unique_chains
    
    def get_strongest_implications(
        self,
        constraint: OCLConstraint,
        candidates: List[OCLConstraint],
        top_k: int = 5
    ) -> List[Implication]:
        """
        Get the top-k strongest implications from a constraint.
        
        Args:
            constraint: Premise constraint
            candidates: Potential conclusion constraints
            top_k: Number of top implications to return
            
        Returns:
            List of top implications sorted by strength and confidence
        """
        implications = []
        
        for candidate in candidates:
            impl = self.analyze_implication(constraint, candidate)
            if impl.strength != ImplicationStrength.NONE:
                implications.append(impl)
        
        # Sort by strength and confidence
        implications.sort(
            key=lambda i: (
                ['none', 'weak', 'conditional', 'strong'].index(i.strength.value),
                i.confidence
            ),
            reverse=True
        )
        
        return implications[:top_k]
