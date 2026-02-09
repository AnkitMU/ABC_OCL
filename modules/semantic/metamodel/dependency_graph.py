"""
Dependency Graph Module
Builds and analyzes dependency relationships between classes and constraints.
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
from ...core.models import Metamodel


@dataclass
class Dependency:
    """Represents a dependency between classes."""
    source: str
    target: str
    dependency_type: str  # 'association', 'inheritance', 'composition', 'aggregation'
    weight: float = 1.0
    name: Optional[str] = None
    metadata: Optional[Dict] = None


class DependencyGraph:
    """Manages dependency graph for metamodel."""
    
    def __init__(self, metamodel: Metamodel):
        self.metamodel = metamodel
        self.graph: Dict[str, List[Dependency]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[Dependency]] = defaultdict(list)
        self.adjacency_matrix: Dict[Tuple[str, str], List[Dependency]] = {}
        self._build_graph()
    
    def _build_graph(self):
        """Build dependency graph from metamodel."""
        # Add association dependencies
        for assoc in self.metamodel.get_all_associations():
            dep_type = 'composition' if assoc.is_composition else 'association'
            weight = 2.0 if assoc.is_composition else 1.0
            
            dep = Dependency(
                source=assoc.source_class,
                target=assoc.target_class,
                dependency_type=dep_type,
                weight=weight,
                name=assoc.name,
                metadata={
                    'multiplicity': assoc.multiplicity,
                    'is_composition': assoc.is_composition
                }
            )
            
            self.graph[assoc.source_class].append(dep)
            self.reverse_graph[assoc.target_class].append(dep)
            
            key = (assoc.source_class, assoc.target_class)
            if key not in self.adjacency_matrix:
                self.adjacency_matrix[key] = []
            self.adjacency_matrix[key].append(dep)
        
        # Add inheritance dependencies
        for cls in self.metamodel.classes.values():
            if cls.parent_class:
                dep = Dependency(
                    source=cls.name,
                    target=cls.parent_class,
                    dependency_type='inheritance',
                    weight=1.5,
                    metadata={'type': 'parent_child'}
                )
                self.graph[cls.name].append(dep)
                self.reverse_graph[cls.parent_class].append(dep)
                
                key = (cls.name, cls.parent_class)
                if key not in self.adjacency_matrix:
                    self.adjacency_matrix[key] = []
                self.adjacency_matrix[key].append(dep)
    
    def get_dependencies(self, class_name: str) -> List[Dependency]:
        """Get all outgoing dependencies for a class."""
        return self.graph.get(class_name, [])
    
    def get_dependents(self, class_name: str) -> List[Dependency]:
        """Get all classes that depend on this class (incoming dependencies)."""
        return self.reverse_graph.get(class_name, [])
    
    def get_dependency_between(self, source: str, target: str) -> List[Dependency]:
        """Get all dependencies between two classes."""
        return self.adjacency_matrix.get((source, target), [])
    
    def has_dependency(self, source: str, target: str) -> bool:
        """Check if there is a dependency from source to target."""
        return (source, target) in self.adjacency_matrix
    
    def find_cycles(self) -> List[List[str]]:
        """Find cyclic dependencies in the graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # Found cycle
                try:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:] + [node]
                    # Normalize cycle (start from smallest element)
                    min_idx = cycle.index(min(cycle[:-1]))
                    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                    if normalized not in cycles:
                        cycles.append(normalized)
                except ValueError:
                    pass
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for dep in self.graph.get(node, []):
                dfs(dep.target, path.copy())
            
            rec_stack.remove(node)
        
        for class_name in self.metamodel.get_class_names():
            if class_name not in visited:
                dfs(class_name, [])
        
        return cycles
    
    def topological_sort(self) -> Optional[List[str]]:
        """
        Perform topological sort of classes.
        Returns None if graph has cycles.
        """
        in_degree = defaultdict(int)
        
        # Initialize in-degrees
        for class_name in self.metamodel.get_class_names():
            in_degree[class_name] = 0
        
        # Calculate in-degrees
        for deps in self.graph.values():
            for dep in deps:
                in_degree[dep.target] += 1
        
        # Perform topological sort using Kahn's algorithm
        queue = deque([n for n in in_degree if in_degree[n] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for dep in self.graph.get(node, []):
                in_degree[dep.target] -= 1
                if in_degree[dep.target] == 0:
                    queue.append(dep.target)
        
        # Check if all nodes are included (no cycles)
        if len(result) != len(self.metamodel.get_class_names()):
            return None  # Graph has cycles
        
        return result
    
    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest dependency path between two classes (BFS)."""
        if source not in self.metamodel.get_class_names() or \
           target not in self.metamodel.get_class_names():
            return None
        
        if source == target:
            return [source]
        
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            for dep in self.graph.get(current, []):
                if dep.target == target:
                    return path + [target]
                
                if dep.target not in visited:
                    visited.add(dep.target)
                    queue.append((dep.target, path + [dep.target]))
        
        return None
    
    def all_paths(self, source: str, target: str, max_depth: int = 10) -> List[List[str]]:
        """Find all paths between two classes (limited by max_depth)."""
        if source not in self.metamodel.get_class_names() or \
           target not in self.metamodel.get_class_names():
            return []
        
        if source == target:
            return [[source]]
        
        paths = []
        
        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            for dep in self.graph.get(current, []):
                if dep.target not in visited:
                    visited.add(dep.target)
                    path.append(dep.target)
                    dfs(dep.target, path, visited)
                    path.pop()
                    visited.remove(dep.target)
        
        dfs(source, [source], {source})
        return paths
    
    def get_strongly_connected_components(self) -> List[Set[str]]:
        """Find strongly connected components using Tarjan's algorithm."""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = defaultdict(bool)
        sccs = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            for dep in self.graph.get(node, []):
                if dep.target not in index:
                    strongconnect(dep.target)
                    lowlinks[node] = min(lowlinks[node], lowlinks[dep.target])
                elif on_stack[dep.target]:
                    lowlinks[node] = min(lowlinks[node], index[dep.target])
            
            if lowlinks[node] == index[node]:
                component = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.add(w)
                    if w == node:
                        break
                sccs.append(component)
        
        for node in self.metamodel.get_class_names():
            if node not in index:
                strongconnect(node)
        
        return sccs
    
    def get_transitive_closure(self, class_name: str) -> Set[str]:
        """Get all classes reachable from given class (transitive closure)."""
        reachable = set()
        to_visit = [class_name]
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            if current != class_name:
                reachable.add(current)
            
            for dep in self.graph.get(current, []):
                if dep.target not in visited:
                    to_visit.append(dep.target)
        
        return reachable
    
    def get_inverse_transitive_closure(self, class_name: str) -> Set[str]:
        """Get all classes that can reach given class (inverse transitive closure)."""
        reachable = set()
        to_visit = [class_name]
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            if current != class_name:
                reachable.add(current)
            
            for dep in self.reverse_graph.get(current, []):
                if dep.source not in visited:
                    to_visit.append(dep.source)
        
        return reachable
    
    def calculate_coupling(self, class_name: str) -> Dict[str, int]:
        """Calculate coupling metrics for a class."""
        outgoing = len(self.graph.get(class_name, []))
        incoming = len(self.reverse_graph.get(class_name, []))
        
        # Efferent coupling (depends on)
        efferent = len(set(dep.target for dep in self.graph.get(class_name, [])))
        
        # Afferent coupling (depended upon by)
        afferent = len(set(dep.source for dep in self.reverse_graph.get(class_name, [])))
        
        # Instability metric (I = Ce / (Ce + Ca))
        total_coupling = efferent + afferent
        instability = efferent / total_coupling if total_coupling > 0 else 0.0
        
        return {
            'outgoing_dependencies': outgoing,
            'incoming_dependencies': incoming,
            'efferent_coupling': efferent,
            'afferent_coupling': afferent,
            'total_coupling': total_coupling,
            'instability': instability
        }
    
    def find_dependency_clusters(self) -> List[Set[str]]:
        """Find clusters of tightly coupled classes."""
        sccs = self.get_strongly_connected_components()
        
        # Filter out single-node components
        clusters = [scc for scc in sccs if len(scc) > 1]
        
        return clusters
    
    def get_dependency_depth(self, class_name: str) -> int:
        """Calculate maximum depth of dependency tree from class."""
        max_depth = 0
        
        def dfs(node: str, depth: int, visited: Set[str]):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            for dep in self.graph.get(node, []):
                if dep.target not in visited:
                    visited.add(dep.target)
                    dfs(dep.target, depth + 1, visited)
                    visited.remove(dep.target)
        
        dfs(class_name, 0, {class_name})
        return max_depth
    
    def get_root_classes(self) -> List[str]:
        """Get classes with no incoming dependencies."""
        roots = []
        for class_name in self.metamodel.get_class_names():
            if not self.reverse_graph.get(class_name):
                roots.append(class_name)
        return roots
    
    def get_leaf_classes(self) -> List[str]:
        """Get classes with no outgoing dependencies."""
        leaves = []
        for class_name in self.metamodel.get_class_names():
            if not self.graph.get(class_name):
                leaves.append(class_name)
        return leaves
    
    def analyze_dependency_patterns(self) -> Dict:
        """Analyze dependency patterns in the graph."""
        cycles = self.find_cycles()
        sccs = self.get_strongly_connected_components()
        roots = self.get_root_classes()
        leaves = self.get_leaf_classes()
        
        # Calculate average coupling
        all_coupling = [self.calculate_coupling(cls) for cls in self.metamodel.get_class_names()]
        avg_coupling = sum(c['total_coupling'] for c in all_coupling) / len(all_coupling) if all_coupling else 0
        
        # Calculate graph density
        num_classes = len(self.metamodel.get_class_names())
        num_edges = sum(len(deps) for deps in self.graph.values())
        max_edges = num_classes * (num_classes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0
        
        return {
            'num_classes': num_classes,
            'num_dependencies': num_edges,
            'graph_density': density,
            'num_cycles': len(cycles),
            'cycles': cycles,
            'num_sccs': len(sccs),
            'sccs_sizes': [len(scc) for scc in sccs],
            'num_roots': len(roots),
            'root_classes': roots,
            'num_leaves': len(leaves),
            'leaf_classes': leaves,
            'average_coupling': avg_coupling,
            'is_acyclic': len(cycles) == 0
        }
    
    def export_graph(self) -> Dict:
        """Export graph in structured format for visualization."""
        nodes = []
        edges = []
        
        for class_name in self.metamodel.get_class_names():
            coupling = self.calculate_coupling(class_name)
            nodes.append({
                'id': class_name,
                'label': class_name,
                'coupling': coupling
            })
        
        edge_id = 0
        for source, deps in self.graph.items():
            for dep in deps:
                edges.append({
                    'id': edge_id,
                    'source': source,
                    'target': dep.target,
                    'type': dep.dependency_type,
                    'weight': dep.weight,
                    'name': dep.name
                })
                edge_id += 1
        
        return {
            'nodes': nodes,
            'edges': edges,
            'analysis': self.analyze_dependency_patterns()
        }


if __name__ == "__main__":
    print("Dependency Graph Module - Ready for testing")
