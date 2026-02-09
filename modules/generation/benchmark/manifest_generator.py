"""
Manifest Generator for ML-Friendly Output

Generates manifest.jsonl files where each line is a complete JSON object
representing one constraint with all metadata. This format is ideal for:
- Machine learning datasets
- Streaming data processing
- Line-by-line parsing without loading entire file
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from modules.core.models import OCLConstraint


def generate_manifest(
    constraints: List[OCLConstraint],
    model_name: str,
    profile_name: str,
    output_path: Path,
    suite_metadata: Optional[Dict[str, Any]] = None,
    verification_results: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generate manifest.jsonl file with one JSON object per line.
    
    Each line contains:
    - constraint_id: Unique identifier (index-based)
    - ocl: The OCL constraint text
    - pattern_id: Pattern used
    - pattern_name: Pattern name
    - context: Context class
    - parameters: Parameter values
    - operators_used: List of OCL operators
    - navigation_depth: Max navigation depth
    - quantifier_depth: Max quantifier nesting
    - difficulty: Difficulty label
    - families: Constraint families
    - solver_result: SAT/UNSAT/UNKNOWN (if verified)
    - solver_time_ms: Verification time (if verified)
    - model: Model name
    - profile: Profile name
    - timestamp: Generation timestamp
    
    Args:
        constraints: List of OCLConstraints
        model_name: Name of the model
        profile_name: Name of the profile
        output_path: Path to save manifest.jsonl
        suite_metadata: Optional suite-level metadata
        verification_results: Optional verification results dict
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, constraint in enumerate(constraints):
            # Base constraint data
            manifest_entry = {
                'constraint_id': idx,
                'ocl': constraint.ocl,
                'pattern_id': constraint.pattern_id,
                'pattern_name': constraint.pattern_name,
                'context': constraint.context,
                'parameters': constraint.parameters,
                'confidence': constraint.confidence,
                'timestamp': constraint.timestamp,
                'model': model_name,
                'profile': profile_name
            }
            
            # Add rich metadata (from metadata_enricher)
            if 'operators_used' in constraint.metadata:
                manifest_entry['operators_used'] = constraint.metadata['operators_used']
            
            if 'navigation_depth' in constraint.metadata:
                manifest_entry['navigation_depth'] = constraint.metadata['navigation_depth']
            
            if 'quantifier_depth' in constraint.metadata:
                manifest_entry['quantifier_depth'] = constraint.metadata['quantifier_depth']
            
            if 'difficulty' in constraint.metadata:
                manifest_entry['difficulty'] = constraint.metadata['difficulty']
            
            if 'operator_count' in constraint.metadata:
                manifest_entry['operator_count'] = constraint.metadata['operator_count']
            
            if 'families' in constraint.metadata:
                manifest_entry['families'] = constraint.metadata['families']
            
            # Add verification results if available
            if verification_results and 'constraints' in verification_results:
                # Match by index or OCL text
                if idx < len(verification_results['constraints']):
                    verif = verification_results['constraints'][idx]
                    manifest_entry['solver_result'] = verif.get('result', 'UNKNOWN')
                    manifest_entry['solver_time_ms'] = verif.get('time_ms', None)
                    manifest_entry['solver_error'] = verif.get('error', None)
            
            # Add any other metadata fields
            for key, value in constraint.metadata.items():
                if key not in ['operators_used', 'navigation_depth', 'quantifier_depth', 
                               'difficulty', 'operator_count', 'families']:
                    manifest_entry[f'meta_{key}'] = value
            
            # Write as single-line JSON
            json.dump(manifest_entry, f, ensure_ascii=False)
            f.write('\n')


def generate_manifest_summary(
    manifest_path: Path,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate summary statistics from manifest.jsonl file.
    
    Args:
        manifest_path: Path to manifest.jsonl
        output_path: Optional path to save summary JSON
        
    Returns:
        Dictionary with summary statistics
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    constraints = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                constraints.append(json.loads(line))
    
    if not constraints:
        return {'total_constraints': 0}
    
    # Aggregate statistics
    summary = {
        'total_constraints': len(constraints),
        'model': constraints[0].get('model', 'unknown'),
        'profile': constraints[0].get('profile', 'unknown'),
        'patterns_used': {},
        'difficulty_distribution': {},
        'family_distribution': {},
        'verification_results': {'SAT': 0, 'UNSAT': 0, 'UNKNOWN': 0, 'ERROR': 0},
        'navigation_depth': {'min': float('inf'), 'max': 0, 'avg': 0},
        'quantifier_depth': {'min': float('inf'), 'max': 0, 'avg': 0},
        'operators': {},
        'solver_time': {'total_ms': 0, 'avg_ms': 0, 'min_ms': float('inf'), 'max_ms': 0}
    }
    
    nav_depths = []
    quant_depths = []
    solver_times = []
    all_operators = set()
    
    for c in constraints:
        # Pattern usage
        pattern_id = c.get('pattern_id', 'unknown')
        summary['patterns_used'][pattern_id] = summary['patterns_used'].get(pattern_id, 0) + 1
        
        # Difficulty distribution
        difficulty = c.get('difficulty', 'unknown')
        summary['difficulty_distribution'][difficulty] = \
            summary['difficulty_distribution'].get(difficulty, 0) + 1
        
        # Family distribution
        families = c.get('families', [])
        for fam in families:
            summary['family_distribution'][fam] = summary['family_distribution'].get(fam, 0) + 1
        
        # Verification results
        solver_result = c.get('solver_result', 'UNKNOWN')
        summary['verification_results'][solver_result] = \
            summary['verification_results'].get(solver_result, 0) + 1
        
        # Navigation depth
        nav_depth = c.get('navigation_depth', 0)
        if nav_depth is not None:
            nav_depths.append(nav_depth)
            summary['navigation_depth']['min'] = min(summary['navigation_depth']['min'], nav_depth)
            summary['navigation_depth']['max'] = max(summary['navigation_depth']['max'], nav_depth)
        
        # Quantifier depth
        quant_depth = c.get('quantifier_depth', 0)
        if quant_depth is not None:
            quant_depths.append(quant_depth)
            summary['quantifier_depth']['min'] = min(summary['quantifier_depth']['min'], quant_depth)
            summary['quantifier_depth']['max'] = max(summary['quantifier_depth']['max'], quant_depth)
        
        # Operators
        operators = c.get('operators_used', [])
        for op in operators:
            summary['operators'][op] = summary['operators'].get(op, 0) + 1
            all_operators.add(op)
        
        # Solver time
        solver_time = c.get('solver_time_ms')
        if solver_time is not None and solver_time > 0:
            solver_times.append(solver_time)
            summary['solver_time']['total_ms'] += solver_time
            summary['solver_time']['min_ms'] = min(summary['solver_time']['min_ms'], solver_time)
            summary['solver_time']['max_ms'] = max(summary['solver_time']['max_ms'], solver_time)
    
    # Calculate averages
    if nav_depths:
        summary['navigation_depth']['avg'] = sum(nav_depths) / len(nav_depths)
    if quant_depths:
        summary['quantifier_depth']['avg'] = sum(quant_depths) / len(quant_depths)
    if solver_times:
        summary['solver_time']['avg_ms'] = summary['solver_time']['total_ms'] / len(solver_times)
    
    # Clean up infinity values
    if summary['navigation_depth']['min'] == float('inf'):
        summary['navigation_depth']['min'] = 0
    if summary['quantifier_depth']['min'] == float('inf'):
        summary['quantifier_depth']['min'] = 0
    if summary['solver_time']['min_ms'] == float('inf'):
        summary['solver_time']['min_ms'] = 0
    
    summary['unique_operators'] = len(all_operators)
    summary['unique_patterns'] = len(summary['patterns_used'])
    
    # Save summary if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    return summary


def read_manifest_stream(manifest_path: Path):
    """
    Generator to stream constraints from manifest.jsonl line by line.
    Memory-efficient for large files.
    
    Args:
        manifest_path: Path to manifest.jsonl
        
    Yields:
        Dictionary for each constraint
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue


def filter_manifest(
    manifest_path: Path,
    output_path: Path,
    difficulty: Optional[str] = None,
    families: Optional[List[str]] = None,
    solver_result: Optional[str] = None,
    min_nav_depth: Optional[int] = None,
    max_nav_depth: Optional[int] = None
) -> int:
    """
    Filter manifest.jsonl based on criteria and write to new file.
    
    Args:
        manifest_path: Input manifest.jsonl
        output_path: Output manifest.jsonl
        difficulty: Filter by difficulty level
        families: Filter by families (any match)
        solver_result: Filter by solver result
        min_nav_depth: Minimum navigation depth
        max_nav_depth: Maximum navigation depth
        
    Returns:
        Number of constraints written
    """
    count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out:
        for constraint in read_manifest_stream(manifest_path):
            # Apply filters
            if difficulty and constraint.get('difficulty') != difficulty:
                continue
            
            if families:
                constraint_families = constraint.get('families', [])
                if not any(f in constraint_families for f in families):
                    continue
            
            if solver_result and constraint.get('solver_result') != solver_result:
                continue
            
            nav_depth = constraint.get('navigation_depth', 0)
            if min_nav_depth is not None and nav_depth < min_nav_depth:
                continue
            if max_nav_depth is not None and nav_depth > max_nav_depth:
                continue
            
            # Write matching constraint
            json.dump(constraint, out, ensure_ascii=False)
            out.write('\n')
            count += 1
    
    return count
