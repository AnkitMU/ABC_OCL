"""
Coverage tracker utilities: compute achieved coverage over generated constraints.
Heuristic string-based analyzers to avoid full AST parsing.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import re

OPERATORS = [
    "forAll", "exists", "select", "collect", "size", "isUnique",
    "implies", "oclIsKindOf", "oclAsType", "oclIsUndefined", "oclIsInvalid"
]


def count_operators(ocl: str) -> Dict[str, int]:
    counts: Dict[str, int] = {op: 0 for op in OPERATORS}
    for op in OPERATORS:
        # rough regex word-boundary
        counts[op] = len(re.findall(r"\b"+re.escape(op)+r"\b", ocl))
    return counts


def nav_hops(ocl: str) -> int:
    """
    Estimate navigation hops.
    Count '->' plus dot-navigation that follows 'self'.
    This avoids counting decimals or enum namespace separators.
    """
    arrow_hops = ocl.count('->')
    dot_hops = len(re.findall(r"\bself\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", ocl))
    return arrow_hops + dot_hops


def quantifier_depth(ocl: str) -> int:
    """
    Approximate nesting depth of quantifiers.
    Count quantifier lambdas like forAll(...|...), exists(...|...), select(...|...), collect(...|...).
    """
    return len(re.findall(r"->(?:forAll|exists|select|collect)\s*\([^|]*\|", ocl))


def types_touched(ocl: str) -> Dict[str, int]:
    # heuristic types via keywords
    real_matches = re.findall(r"\b[0-9]+\.[0-9]+\b", ocl)
    # remove reals before counting integers
    ocl_without_reals = re.sub(r"\b[0-9]+\.[0-9]+\b", " ", ocl)
    return {
        'Integer': len(re.findall(r"\b(?:0|[1-9][0-9]*)\b", ocl_without_reals)),
        'Real': len(real_matches),
        'String': len(re.findall(r"\bsize\(\)|'[^']*'\b|\"[^\"]*\"", ocl)),
        'Boolean': len(re.findall(r"\b(true|false)\b", ocl, flags=re.IGNORECASE)),
        'Enum': len(re.findall(r"::", ocl)),
    }


def compute_coverage(constraints) -> Dict:
    classes = set()
    attributes = set()
    associations = set()
    op_counts = {op: 0 for op in OPERATORS}
    hop_hist = {0:0, 1:0, 2:0}
    depth_hist = {0:0, 1:0, 2:0}
    type_hits = {'Integer':0,'Real':0,'String':0,'Boolean':0,'Enum':0}

    for c in constraints:
        classes.add(c.context)
        ocl = c.ocl
        # naive attr/assoc references (prefer association detection first)
        for token in re.findall(r"\bself\.[A-Za-z_][A-Za-z0-9_]*->", ocl):
            name = token.split('.')[1].split('->')[0]
            associations.add((c.context, name))
        for token in re.findall(r"\bself\.[A-Za-z_][A-Za-z0-9_]*\b", ocl):
            name = token.split('.')[1]
            if (c.context, name) not in associations:
                attributes.add((c.context, name))

        # operators
        ops = count_operators(ocl)
        for k,v in ops.items():
            op_counts[k] += v

        # hops
        hops = nav_hops(ocl)
        hop_hist[0 if hops==0 else 1 if hops==1 else 2] += 1

        # quantifier depth
        depth = quantifier_depth(ocl)
        depth_hist[0 if depth==0 else 1 if depth==1 else 2] += 1

        # types
        th = types_touched(ocl)
        for k,v in th.items():
            type_hits[k] += 1 if v>0 else 0

    return {
        'classes_used': len(classes),
        'attributes_used': len(attributes),
        'associations_used': len(associations),
        'operator_counts': op_counts,
        'hop_hist': hop_hist,
        'depth_hist': depth_hist,
        'types': type_hits,
    }
