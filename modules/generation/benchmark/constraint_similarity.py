"""
Constraint Similarity Utilities

Provides two families of similarity checks for OCL constraints:
1) Structural similarity via lightweight AST + tree-edit distance
2) Semantic similarity via SentenceTransformer embeddings
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from modules.core.models import OCLConstraint


# =========================
# Structural (AST) Similarity
# =========================

@dataclass
class ASTNode:
    """Simple AST node for OCL expressions."""
    type: str
    value: Any
    children: List['ASTNode'] = field(default_factory=list)

    def __repr__(self) -> str:
        if not self.children:
            return f"{self.type}({self.value})"
        children_str = ', '.join(repr(c) for c in self.children)
        return f"{self.type}({self.value})[{children_str}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'value': str(self.value),
            'children': [c.to_dict() for c in self.children]
        }

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def get_node_types(self) -> List[str]:
        types = [self.type]
        for child in self.children:
            types.extend(child.get_node_types())
        return types


class OCLParser:
    """Lightweight OCL parser to build AST for structural similarity."""

    BINARY_OPS = [
        ['implies'],
        ['or', 'xor'],
        ['and'],
        ['=', '<>', '<', '>', '<=', '>='],
        ['+', '-'],
        ['*', '/', 'div', 'mod']
    ]

    UNARY_OPS = ['not', '-']

    COLLECTION_OPS = [
        'forAll', 'exists', 'select', 'reject', 'collect',
        'size', 'isEmpty', 'notEmpty', 'includes', 'excludes',
        'sum', 'count', 'any', 'one', 'isUnique', 'union',
        'intersection', 'including', 'excluding'
    ]

    def __init__(self, ocl: str):
        self.ocl = ocl
        self.tokens: List[str] = []
        self.pos = 0

    def tokenize(self, expr: str) -> List[str]:
        if 'inv:' in expr:
            expr = expr.split('inv:', 1)[1].strip()
        pattern = r'(\w+|->|<>|<=|>=|[=<>+\-*/(){}|.,:\[\]]|\'[^\']*\')'
        tokens = re.findall(pattern, expr)
        return [t for t in tokens if t.strip()]

    def parse(self, expr: str) -> ASTNode:
        self.tokens = self.tokenize(expr)
        self.pos = 0
        if not self.tokens:
            return ASTNode('empty', None)
        try:
            return self.parse_expression()
        except Exception:
            return ASTNode('expression', expr)

    def current_token(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> str:
        token = self.current_token()
        self.pos += 1
        return token

    def parse_expression(self) -> ASTNode:
        return self.parse_binary_op(0)

    def parse_binary_op(self, precedence: int) -> ASTNode:
        if precedence >= len(self.BINARY_OPS):
            return self.parse_unary()
        left = self.parse_binary_op(precedence + 1)
        while self.current_token() in self.BINARY_OPS[precedence]:
            op = self.consume()
            right = self.parse_binary_op(precedence + 1)
            left = ASTNode('binary_op', op, [left, right])
        return left

    def parse_unary(self) -> ASTNode:
        if self.current_token() in self.UNARY_OPS:
            op = self.consume()
            operand = self.parse_unary()
            return ASTNode('unary_op', op, [operand])
        return self.parse_postfix()

    def parse_postfix(self) -> ASTNode:
        node = self.parse_primary()
        while True:
            token = self.current_token()
            if token == '.':
                self.consume()
                attr = self.consume()
                node = ASTNode('navigation', attr, [node])
            elif token == '->':
                self.consume()
                op = self.consume()
                if op in self.COLLECTION_OPS:
                    args = []
                    if self.current_token() == '(':
                        self.consume()
                        while self.current_token() and self.current_token() != ')':
                            if self.current_token() == '|':
                                self.consume()
                            else:
                                args.append(self.parse_expression())
                            if self.current_token() == ',':
                                self.consume()
                            elif self.current_token() != ')':
                                break
                        if self.current_token() == ')':
                            self.consume()
                    node = ASTNode('collection_op', op, [node] + args)
                else:
                    node = ASTNode('collection_op', op, [node])
            elif token == '(':
                self.consume()
                args = []
                while self.current_token() and self.current_token() != ')':
                    args.append(self.parse_expression())
                    if self.current_token() == ',':
                        self.consume()
                if self.current_token() == ')':
                    self.consume()
                node = ASTNode('call', node.value if node.type == 'identifier' else str(node), args)
            else:
                break
        return node

    def parse_primary(self) -> ASTNode:
        token = self.current_token()
        if not token:
            return ASTNode('empty', None)
        if token == '(':
            self.consume()
            node = self.parse_expression()
            if self.current_token() == ')':
                self.consume()
            return node
        if token.startswith("'"):
            self.consume()
            return ASTNode('string_literal', token[1:-1])
        if token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
            self.consume()
            return ASTNode('number_literal', token)
        if token in ['true', 'false']:
            self.consume()
            return ASTNode('boolean_literal', token)
        if token in ['null', 'self']:
            self.consume()
            return ASTNode('keyword', token)
        if token in ['if', 'then', 'else', 'endif', 'let', 'in']:
            self.consume()
            return ASTNode('control', token)
        self.consume()
        return ASTNode('identifier', token)


def tree_edit_distance(node1: ASTNode, node2: ASTNode) -> int:
    """Simplified tree edit distance."""
    if node1 is None and node2 is None:
        return 0
    if node1 is None:
        return node2.size()
    if node2 is None:
        return node1.size()

    cost = 0 if (node1.type == node2.type and node1.value == node2.value) else 1
    len1 = len(node1.children)
    len2 = len(node2.children)
    if len1 == 0 and len2 == 0:
        return cost

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = sum(node1.children[j].size() for j in range(i))
    for j in range(len2 + 1):
        dp[0][j] = sum(node2.children[j].size() for j in range(j))

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            replace_cost = tree_edit_distance(node1.children[i - 1], node2.children[j - 1])
            delete_cost = dp[i - 1][j] + node1.children[i - 1].size()
            insert_cost = dp[i][j - 1] + node2.children[j - 1].size()
            dp[i][j] = min(replace_cost + dp[i - 1][j - 1], delete_cost, insert_cost)

    return cost + dp[len1][len2]


def ast_similarity(c1: OCLConstraint, c2: OCLConstraint) -> float:
    """Compute AST-based structural similarity between two constraints."""
    parser1 = OCLParser(c1.ocl)
    parser2 = OCLParser(c2.ocl)
    tree1 = parser1.parse(c1.ocl)
    tree2 = parser2.parse(c2.ocl)
    distance = tree_edit_distance(tree1, tree2)
    max_size = max(tree1.size(), tree2.size())
    if max_size == 0:
        return 1.0
    similarity = 1.0 - (distance / max_size)
    return max(0.0, min(1.0, similarity))


def ast_node_type_similarity(c1: OCLConstraint, c2: OCLConstraint) -> float:
    """Fast structural similarity using node-type distribution."""
    parser1 = OCLParser(c1.ocl)
    parser2 = OCLParser(c2.ocl)
    tree1 = parser1.parse(c1.ocl)
    tree2 = parser2.parse(c2.ocl)
    types1 = tree1.get_node_types()
    types2 = tree2.get_node_types()
    from collections import Counter
    counter1 = Counter(types1)
    counter2 = Counter(types2)
    intersection = sum((counter1 & counter2).values())
    union = sum((counter1 | counter2).values())
    if union == 0:
        return 1.0
    return intersection / union


def compute_ast_features(constraint: OCLConstraint) -> Dict[str, Any]:
    parser = OCLParser(constraint.ocl)
    tree = parser.parse(constraint.ocl)
    node_types = tree.get_node_types()
    from collections import Counter
    type_counts = Counter(node_types)
    return {
        'ast_depth': tree.depth(),
        'ast_size': tree.size(),
        'node_type_distribution': dict(type_counts),
        'binary_ops': type_counts.get('binary_op', 0),
        'unary_ops': type_counts.get('unary_op', 0),
        'collection_ops': type_counts.get('collection_op', 0),
        'navigation_ops': type_counts.get('navigation', 0),
        'literals': (type_counts.get('string_literal', 0) +
                    type_counts.get('number_literal', 0) +
                    type_counts.get('boolean_literal', 0))
    }


# =========================
# Semantic Similarity
# =========================

_model = None
_model_name = 'all-MiniLM-L6-v2'


def get_sentence_transformer():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")
    return _model


def normalize_ocl_for_embedding(ocl: str) -> str:
    if 'inv:' in ocl:
        ocl = ocl.split('inv:', 1)[1].strip()
    return ' '.join(ocl.split())


def compute_embedding(ocl: str) -> np.ndarray:
    model = get_sentence_transformer()
    normalized = normalize_ocl_for_embedding(ocl)
    return model.encode(normalized, convert_to_numpy=True)


def compute_embeddings_batch(ocl_list: List[str]) -> np.ndarray:
    model = get_sentence_transformer()
    normalized_list = [normalize_ocl_for_embedding(ocl) for ocl in ocl_list]
    return model.encode(normalized_list, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(np.clip(similarity, -1.0, 1.0))


def semantic_similarity(c1: OCLConstraint, c2: OCLConstraint) -> float:
    emb1 = compute_embedding(c1.ocl)
    emb2 = compute_embedding(c2.ocl)
    sim = cosine_similarity(emb1, emb2)
    return (sim + 1) / 2


def semantic_similarity_matrix(constraints: List[OCLConstraint]) -> np.ndarray:
    if not constraints:
        return np.array([])
    ocl_list = [c.ocl for c in constraints]
    embeddings = compute_embeddings_batch(ocl_list)
    n = len(constraints)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            sim = (sim + 1) / 2
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    return similarity_matrix


def find_semantic_duplicates(
    constraints: List[OCLConstraint],
    threshold: float = 0.95
) -> List[tuple[int, int, float]]:
    sim_matrix = semantic_similarity_matrix(constraints)
    duplicates = []
    n = len(constraints)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                duplicates.append((i, j, sim_matrix[i, j]))
    return sorted(duplicates, key=lambda x: -x[2])


def cluster_by_semantic_similarity(
    constraints: List[OCLConstraint],
    threshold: float = 0.8
) -> List[List[int]]:
    sim_matrix = semantic_similarity_matrix(constraints)
    n = len(constraints)
    clusters = [[i] for i in range(n)]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                similarities = []
                for idx1 in clusters[i]:
                    for idx2 in clusters[j]:
                        similarities.append(sim_matrix[idx1, idx2])
                avg_sim = np.mean(similarities) if similarities else 0.0
                if avg_sim >= threshold:
                    clusters[i].extend(clusters[j])
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break
    return clusters


def compute_semantic_diversity_score(constraints: List[OCLConstraint]) -> float:
    if len(constraints) <= 1:
        return 1.0
    sim_matrix = semantic_similarity_matrix(constraints)
    n = len(constraints)
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += sim_matrix[i, j]
            count += 1
    avg_similarity = total_sim / count if count > 0 else 0.0
    return 1.0 - avg_similarity


def get_semantic_features(constraint: OCLConstraint) -> Dict[str, Any]:
    embedding = compute_embedding(constraint.ocl)
    return {
        'semantic_embedding': embedding.tolist(),
        'embedding_dim': len(embedding),
        'embedding_norm': float(np.linalg.norm(embedding))
    }


def is_semantically_similar(
    c1: OCLConstraint,
    c2: OCLConstraint,
    threshold: float = 0.9
) -> bool:
    sim = semantic_similarity(c1, c2)
    return sim >= threshold


_embedding_cache: Dict[str, np.ndarray] = {}


def compute_embedding_cached(ocl: str) -> np.ndarray:
    normalized = normalize_ocl_for_embedding(ocl)
    if normalized not in _embedding_cache:
        _embedding_cache[normalized] = compute_embedding(ocl)
    return _embedding_cache[normalized]


def clear_embedding_cache():
    global _embedding_cache
    _embedding_cache.clear()
