import re
from z3 import (
    Solver, Int, Real, Bool, And, Or, Not, Distinct, Sum, If, IntVal, RealVal
)
from ssr_ocl.lowering.scopes import resolve_count, resolve_enum
from ssr_ocl.lowering.encodings import int_array, enum_sort, bool_symbol
from ssr_ocl.lowering.unified_smt_encoder import get_unified_encoder
from ssr_ocl.types import Candidate
from ssr_ocl.parsers.xmi_parser import extract_collection_mappings

# Neural-Symbolic Components
try:
    from ssr_ocl.neural import get_neural_classifier, OCLPatternType
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("  Neural components not available, using symbolic-only mode")

# Optional enhanced encoders (guard against missing module)
ENHANCED_ENCODERS_AVAILABLE = False
try:
    from ssr_ocl.lowering.enhanced_encoders import get_enhanced_encoders
    ENHANCED_ENCODERS_AVAILABLE = True
except Exception:
    ENHANCED_ENCODERS_AVAILABLE = False

# ===== IMPROVED OCL REGEX PATTERNS (spec-aligned) =====

# ===== QUANTIFIERS =====
FORALL_RE = re.compile(
    r"->\s*forAll\s*\(\s*(?:(?P<vars>\w+(?:\s*,\s*\w+)*)\s*\|\s*)?(?P<body>.+)\)",
    re.IGNORECASE
)
EXISTS_RE = re.compile(
    r"->\s*exists\s*\(\s*(?:(?P<vars>\w+(?:\s*,\s*\w+)*)\s*\|\s*)?(?P<body>.+)\)",
    re.IGNORECASE
)
ONE_RE = re.compile(
    r"->\s*one\s*\(\s*(\w+)\s*\|\s*(.+)\)",
    re.IGNORECASE
)
IS_UNIQUE_RE = re.compile(
    r"->\s*isUnique\s*\(\s*(\w+)\s*\|\s*(.+)\)",
    re.IGNORECASE
)

# ===== COLLECTION OPERATIONS =====
SELECT_RE = re.compile(
    r"->\s*select\s*\(\s*(\w+)\s*\|\s*([^)]+)\)",
    re.IGNORECASE
)
REJECT_RE = re.compile(
    r"->\s*reject\s*\(\s*(\w+)\s*\|\s*([^)]+)\)",
    re.IGNORECASE
)
COLLECT_RE = re.compile(
    r"->\s*collect\s*\(\s*(\w+)\s*\|\s*([^)]+)\)",
    re.IGNORECASE
)

# ===== SIZE & MEMBERSHIP =====
SIZE_RE = re.compile(
    r"->\s*size\s*\(\)\s*(<=|>=|<>|=|<|>)\s*(-?\d+|[A-Za-z_][\w\.]*)",
    re.IGNORECASE
)
INCLUDES_RE = re.compile(
    r"->\s*includes\s*\(\s*(.+?)\s*\)",
    re.IGNORECASE
)
EXCLUDES_RE = re.compile(
    r"->\s*excludes\s*\(\s*(.+?)\s*\)",
    re.IGNORECASE
)
INCLUDES_ALL_RE = re.compile(
    r"->\s*includesAll\s*\(\s*(.+?)\s*\)",
    re.IGNORECASE
)
EXCLUDES_ALL_RE = re.compile(
    r"->\s*excludesAll\s*\(\s*(.+?)\s*\)",
    re.IGNORECASE
)
COUNT_RE = re.compile(
    r"->\s*count\s*\(\s*(.+?)\s*\)",
    re.IGNORECASE
)

# ===== NULL & TYPE CHECKING =====
NULL_CHECK_RE = re.compile(
    r"([A-Za-z_][\w\.]*)\s*(<>|=)\s*null",
    re.IGNORECASE
)
KINDOF_RE = re.compile(
    r"oclIsKindOf\s*\(\s*([A-Za-z_]\w*(?::{2}\w+)*)\s*\)",
    re.IGNORECASE
)
TYPEOF_RE = re.compile(
    r"oclIsTypeOf\s*\(\s*([A-Za-z_]\w*(?::{2}\w+)*)\s*\)",
    re.IGNORECASE
)

# ===== GLOBAL CONSTRAINTS =====
ALLINST_FORALL_RE = re.compile(
    r"([A-Za-z_]\w*(?::{2}\w+)*)\.allInstances\(\)\s*->\s*forAll\s*\(\s*(\w+(?:\s*,\s*\w+)*)?\s*\|\s*(.+)\)",
    re.IGNORECASE
)
ALLINST_CHAIN_RE = re.compile(
    r"([A-Za-z_]\w*(?::{2}\w+)*)\.allInstances\(\)\s*->\s*collect\(\s*(\w+)\s*\)\s*->\s*select\(\s*\w+\s*\|\s*([^)]+)\s*\)\s*->\s*size\(\)\s*(<=|>=|<>|=|<|>)\s*(-?\d+)",
    re.IGNORECASE
)

# ===== PAIRWISE PATTERNS =====
PAIRWISE_UNIQ_RE = re.compile(
    r"->\s*forAll\s*\(\s*(\w+)\s*,\s*(\w+)\s*\|\s*\1\s*<>\s*\2\s*implies\s*(.+?)\s*\)",
    re.IGNORECASE
)

def guess_collection_class(ocl: str, model_mappings=None):
    # First try model-derived mappings if available
    if model_mappings:
        for pattern, (class_name, collection_name) in model_mappings.items():
            if pattern in ocl:
                return class_name, collection_name
    
    # Enhanced heuristics for university domain
    if "departments->" in ocl: return "Department", "departments"
    if "students->" in ocl: return "Student", "students"
    if "professors->" in ocl: return "Professor", "professors"
    if "courses->" in ocl: return "Course", "courses"
    if "enrollments->" in ocl: return "Enrollment", "enrollments"
    if "rooms->" in ocl: return "Room", "rooms"
    if "buildings->" in ocl: return "Building", "buildings"
    if "prerequisites->" in ocl: return "Course", "prerequisites"
    # Legacy patterns
    if "books->" in ocl: return "Book", "books"
    if "members->" in ocl: return "Member", "members"
    if "loans->" in ocl: return "Loan", "loans"
    if "copies->" in ocl: return "Copy", "copies"
    return "Element", "elements"

def encode_candidate_to_z3(candidate: Candidate, cfg: dict, model_mappings=None):
    """Neural-Symbolic OCL to Z3 encoding using unified SMT encoder"""
    text = candidate.ocl
    unified_encoder = get_unified_encoder()
    
    # === NEURAL PATTERN CLASSIFICATION ===
    pattern_name = "unknown"
    confidence = 0.0
    
    if NEURAL_AVAILABLE:
        try:
            classifier = get_neural_classifier()
            pattern_type, confidence = classifier.predict(text)
            pattern_name = pattern_type.value
            print(f"🧠 Neural classification: {pattern_name} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"  Neural classification failed: {e}")
    
    # === UNIFIED ENCODING ===
    try:
        class_name, collection = guess_collection_class(text, model_mappings)
        context = {
            'collection': collection,
            'scope': resolve_count(class_name, cfg, default=5),
            'inner_scope': 3,
            'constraint_text': text
        }
        
        # Use unified encoder for all patterns
        if pattern_name != "unknown":
            solver, model_vars = unified_encoder.encode(pattern_name, text, context)
            print(f" Encoding successful using unified SMT encoder")
            return solver, model_vars
        else:
            # Fallback: try to encode with generic method
            from z3 import Bool
            solver = Solver()
            model_vars = {"fallback_constraint": Bool("constraint")}
            solver.add(model_vars["fallback_constraint"])
            print(f"  Using fallback encoding")
            return solver, model_vars
            
    except Exception as e:
        print(f" Encoding failed: {e}")
        from z3 import Bool
        solver = Solver()
        model_vars = {"error_constraint": Bool("constraint")}
        solver.add(model_vars["error_constraint"])
        return solver, model_vars

def _neural_symbolic_encode(pattern_type: 'OCLPatternType', text: str, context: dict, solver: Solver, model_vars: dict, cfg: dict):
    """Neural-guided symbolic encoding with support for 24 patterns"""
    try:
        # Basic patterns (original)
        if pattern_type == OCLPatternType.PAIRWISE_UNIQUENESS:
            return _neural_pairwise_uniqueness(text, solver, model_vars, context, cfg)
        elif pattern_type == OCLPatternType.EXACT_COUNT_SELECTION:
            return _neural_exact_count_selection(text, solver, model_vars, context, cfg)
        elif pattern_type == OCLPatternType.SIZE_CONSTRAINT:
            return _neural_size_constraint(text, solver, model_vars, context, cfg)
        elif pattern_type == OCLPatternType.NUMERIC_COMPARISON:
            return _neural_numeric_comparison(text, solver, model_vars, context, cfg)
        elif pattern_type == OCLPatternType.NULL_CHECK:
            return _neural_null_check(text, solver, model_vars, context, cfg)
        elif pattern_type == OCLPatternType.GLOBAL_COLLECTION:
            return _neural_global_collection(text, solver, model_vars, context, cfg)
        
        # Advanced patterns (from enhanced_encoders)
        elif ENHANCED_ENCODERS_AVAILABLE:
            encoders = get_enhanced_encoders()
            if pattern_type == OCLPatternType.EXACTLY_ONE:
                result = encoders.encode_one_exactly_one(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.CLOSURE:
                result = encoders.encode_closure_transitive(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.ACYCLICITY:
                result = encoders.encode_acyclicity(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.ITERATE:
                result = encoders.encode_aggregation_iterate(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.IMPLIES:
                result = encoders.encode_boolean_guard_implies(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.SAFE_NAVIGATION:
                result = encoders.encode_definedness_safe_navigation(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.TYPE_CHECK:
                result = encoders.encode_type_check_casting(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.SUBSET_DISJOINT:
                result = encoders.encode_subset_disjointness(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.ORDERING:
                result = encoders.encode_ordering_ranking(text, context)
                return result if result else None
            elif pattern_type == OCLPatternType.CONTRACTUAL:
                result = encoders.encode_contractual_temporal(text, context)
                return result if result else None
        
        return None
    except Exception as e:
        print(f"Neural encoding failed: {e}")
        return None

def _neural_pairwise_uniqueness(text: str, solver: Solver, model_vars: dict, context: dict, cfg: dict):
    """Neural-guided pairwise uniqueness encoding"""
    n = context.get('scope', 5)
    collection = context.get('collection', 'holdings')
    
    ids = [Int(f"{collection}_id_{i}") for i in range(n)]
    for i, sym in enumerate(ids):
        model_vars[f"{collection}_id_{i}"] = sym
    
    # Neural insight: Use Z3's Distinct for better performance
    constraint = Distinct(ids)
    solver.add(Not(constraint))
    return solver, model_vars

def _neural_exact_count_selection(text: str, solver: Solver, model_vars: dict, context: dict, cfg: dict):
    """Neural-guided exact count selection encoding"""
    n = context.get('scope', 5)
    collection = context.get('collection', 'holdings')
    
    ids = [Int(f"{collection}_id_{i}") for i in range(n)]
    self_id = Int("self_id")
    
    for i, sym in enumerate(ids):
        model_vars[f"{collection}_id_{i}"] = sym
    model_vars["self_id"] = self_id
    
    # Neural insight: Exactly-one constraint
    matches = [ids[i] == self_id for i in range(n)]
    exactly_one = Or([And(matches[i], And([Not(matches[j]) for j in range(n) if j != i])) for i in range(n)])
    solver.add(Not(exactly_one))
    return solver, model_vars

def _neural_size_constraint(text: str, solver: Solver, model_vars: dict, context: dict, cfg: dict):
    """Neural-guided size constraint encoding"""
    collection = context.get('collection', 'elements')
    size_var = Int(f"{collection}_size")
    model_vars[f"{collection}_size"] = size_var
    
    # Extract operator and value using neural insights
    if "> 0" in text:
        constraint = size_var > 0
    elif "<= " in text:
        import re
        match = re.search(r"<= (\d+)", text)
        value = int(match.group(1)) if match else 6
        constraint = size_var <= value
    else:
        constraint = size_var >= 0  # Default
    
    solver.add(Not(constraint))
    return solver, model_vars

def _neural_numeric_comparison(text: str, solver: Solver, model_vars: dict, context: dict, cfg: dict):
    """Neural-guided numeric comparison encoding"""
    import re
    
    # Extract variable and constraint
    for var_name in ['age', 'gpa', 'credits', 'experienceYears', 'capacity']:
        if var_name in text:
            if var_name == 'gpa':
                var = Real(var_name)
            else:
                var = Int(var_name)
            
            model_vars[var_name] = var
            
            # Extract constraint (look for operator without spaces)
            constraint_match = re.search(f"{var_name}\\s*(>=|<=|>|<|==|=|<>|!=)\\s*([\\d.]+)", text)
            if constraint_match:
                op, value_str = constraint_match.group(1), constraint_match.group(2)
                value = float(value_str) if '.' in value_str else int(value_str)
                
                # Match operator correctly (no spaces in op)
                if op == ">=":
                    constraint = var >= value
                elif op == "<=":
                    constraint = var <= value
                elif op == ">":
                    constraint = var > value
                elif op == "<":
                    constraint = var < value
                elif op in ("=", "=="):
                    constraint = var == value
                elif op in ("<>", "!="):
                    constraint = var != value
                else:
                    return None
                
                solver.add(Not(constraint))
                return solver, model_vars
    
    return None

def _neural_null_check(text: str, solver: Solver, model_vars: dict, context: dict, cfg: dict):
    """Neural-guided null check encoding"""
    # Use improved NULL_CHECK_RE which supports dotted navigation
    m = NULL_CHECK_RE.search(text)
    if m:
        attr, op = m.group(1), m.group(2)
        # Use last part of dotted name for variable
        attr_name = attr.split(".")[-1]
        attr_present = Bool(f"{attr_name}_present")
        model_vars[f"{attr_name}_present"] = attr_present
        
        # Determine constraint based on operator
        if op in ("<>", "!="):
            constraint = attr_present  # Must be present (not null)
        else:
            constraint = Not(attr_present)  # Must be null
        
        solver.add(Not(constraint))
        return solver, model_vars
    
    return None

def _neural_global_collection(text: str, solver: Solver, model_vars: dict, context: dict, cfg: dict):
    """Neural-guided global collection encoding"""
    n = context.get('scope', 6)
    
    if "allInstances" in text and "select" in text:
        ids = [Int(f"all_id_{i}") for i in range(n)]
        self_id = Int("self_id")
        
        for i, sym in enumerate(ids):
            model_vars[f"all_id_{i}"] = sym
        model_vars["self_id"] = self_id
        
        # Exactly one match in global collection
        matches = [ids[i] == self_id for i in range(n)]
        exactly_one = Or([And(matches[i], And([Not(matches[j]) for j in range(n) if j != i])) for i in range(n)])
        solver.add(Not(exactly_one))
        return solver, model_vars
    
    return None

def _extract_collection_name(text: str) -> str:
    """Extract collection name from OCL text"""
    collections = ['holdings', 'students', 'professors', 'courses', 'departments', 'enrollments', 'rooms']
    for collection in collections:
        if collection in text.lower():
            return collection
    return 'elements'

def _extract_class_name(text: str) -> str:
    """Extract class name from OCL context"""
    if 'context' in text.lower():
        lines = text.split('\n')
        for line in lines:
            if 'context' in line.lower():
                parts = line.split()
                if len(parts) > 1:
                    return parts[1]
    return 'Element'

def _symbolic_encode(text: str, s: Solver, model_vars: dict, cfg: dict, model_mappings=None):
    """Original symbolic encoding as fallback"""
    # === LIBRARY SYSTEM UNIQUENESS PATTERNS (Legacy - use generalized patterns) ===
    
    # A) Pairwise uniqueness: self.holdings->forAll(x, y | x <> y implies x.id <> y.id)
    # Guard: Use PAIRWISE_UNIQ_RE if UNIQ_LIB_RE is referenced
    if 'UNIQ_LIB_RE' in globals() and UNIQ_LIB_RE.search(text):
        n = resolve_count("Copy", cfg, default=4)
        ids = [Int(f"hold_id_{i}") for i in range(n)]
        for i, sym in enumerate(ids):
            model_vars[f"hold_id_{i}"] = sym
        
        # Invariant: all IDs are distinct
        constraint = Distinct(ids)
        s.add(Not(constraint))  # Look for duplicate IDs
        return s, model_vars
    
    # B) Holdings select size = 1: self.thelib.holdings->select(...)->size() = 1
    # Guard: Check if SEL_SIZE1_RE exists before using
    if 'SEL_SIZE1_RE' in globals() and SEL_SIZE1_RE.search(text):
        n = resolve_count("Copy", cfg, default=4)
        ids = [Int(f"hold_id_{i}") for i in range(n)]
        self_id = Int("self_id")
        for i, sym in enumerate(ids):
            model_vars[f"hold_id_{i}"] = sym
        model_vars["self_id"] = self_id
        
        # Exactly one ID matches self_id
        matches = [ids[i] == self_id for i in range(n)]
        exactly_one = Or([And(matches[i], And([Not(matches[j]) for j in range(n) if j != i])) for i in range(n)])
        s.add(Not(exactly_one))  # Look for 0 or 2+ matches
        return s, model_vars
    
    # C) AllInstances select size = 1: Copy.allInstances()->collect(id)->select(...)->size() = 1
    # Guard: Check if ALLINST_SEL_RE exists before using
    if 'ALLINST_SEL_RE' in globals() and ALLINST_SEL_RE.search(text):
        n = resolve_count("Copy", cfg, default=5)  # Global scope
        ids = [Int(f"all_id_{i}") for i in range(n)]
        self_id = Int("self_id")
        for i, sym in enumerate(ids):
            model_vars[f"all_id_{i}"] = sym
        model_vars["self_id"] = self_id
        
        matches = [ids[i] == self_id for i in range(n)]
        exactly_one = Or([And(matches[i], And([Not(matches[j]) for j in range(n) if j != i])) for i in range(n)])
        s.add(Not(exactly_one))
        return s, model_vars
    
    # D) AllInstances intersection size = 1: Copy.allInstances()->collect(id)->intersection(Bag{self.id})->size() = 1
    # Guard: Check if ALLINST_INT_RE exists before using
    if 'ALLINST_INT_RE' in globals() and ALLINST_INT_RE.search(text):
        n = resolve_count("Copy", cfg, default=5)
        ids = [Int(f"all_id_{i}") for i in range(n)]
        self_id = Int("self_id")
        for i, sym in enumerate(ids):
            model_vars[f"all_id_{i}"] = sym
        model_vars["self_id"] = self_id
        
        # Count occurrences of self_id in global collection
        count = Sum([If(ids[i] == self_id, 1, 0) for i in range(n)])
        constraint = count == 1
        s.add(Not(constraint))
        return s, model_vars

    # === STANDARD PATTERNS ===
    
    # 1) Size constraints (e.g., self.departments->size() > 0)
    size_m = SIZE_RE.search(text)
    if size_m:
        op, value = size_m.group(1), int(size_m.group(2))
        klass, assoc = guess_collection_class(text, model_mappings)
        n = resolve_count(klass, cfg, default=3)
        # For size() constraints, we model collection size directly
        size_var = Int(f"{assoc}_size")
        model_vars[f"{assoc}_size"] = size_var
        if ">" in op:
            constraint = size_var > value
        elif ">=" in op:
            constraint = size_var >= value
        elif "<" in op:
            constraint = size_var < value
        elif "<=" in op:
            constraint = size_var <= value
        else:  # "=" or "=="
            constraint = size_var == value
        s.add(Not(constraint))
        return s, model_vars

    # 2) isUnique constraints (e.g., self.students->isUnique(s | s.studentId))
    unique_m = IS_UNIQUE_RE.search(text)
    if unique_m:
        var, attr = unique_m.group(1), unique_m.group(2)
        klass, assoc = guess_collection_class(text, model_mappings)
        n = resolve_count(klass, cfg, default=3)
        # Model as array where all elements must be distinct
        arr = int_array(f"{assoc}_ids", n)
        for i, sym in enumerate(arr):
            model_vars[f"{assoc}_{i}_id"] = sym
        # Add distinctness constraint
        constraint = Distinct(arr)
        s.add(Not(constraint))
        return s, model_vars

    # 3) includes constraints (e.g., not self.prerequisites->includes(self))
    includes_m = INCLUDES_RE.search(text)
    if includes_m:
        element = includes_m.group(1)
        klass, assoc = guess_collection_class(text, model_mappings)
        n = resolve_count(klass, cfg, default=3)
        arr = int_array(f"{assoc}_elements", n)
        target = Int(f"{element}_value")
        model_vars[f"{element}_value"] = target
        for i, sym in enumerate(arr):
            model_vars[f"{assoc}_{i}"] = sym
        # includes means OR of all elements equals target
        constraint = Or([sym == target for sym in arr])
        if "not " in text.lower():
            s.add(constraint)  # Want to find where it IS included (violates "not includes")
        else:
            s.add(Not(constraint))  # Want to find where it's NOT included
        return s, model_vars

    # 4) Null checks (e.g., self.advisor <> null)
    null_m = NULL_CHECK_RE.search(text)
    if null_m:
        attr, op = null_m.group(1), null_m.group(2)
        attr_present = Bool(f"{attr}_present")
        model_vars[f"{attr}_present"] = attr_present
        if "<>" in op or "!=" in op:
            constraint = attr_present  # Must be present (not null)
        else:  # "=" 
            constraint = Not(attr_present)  # Must be null
        s.add(Not(constraint))
        return s, model_vars

    # 5) Age and numeric constraints (e.g., self.age >= 16)
    if "age" in text and (">=" in text or ">" in text or "<=" in text or "<" in text):
        age = Int('age')
        model_vars['age'] = age
        # Extract the numeric constraint
        age_constraint = re.search(r"age\s*([><=!]+)\s*(\d+)", text)
        if age_constraint:
            op, value = age_constraint.group(1), int(age_constraint.group(2))
            if ">=" in op:
                constraint = age >= value
            elif ">" in op:
                constraint = age > value
            elif "<=" in op:
                constraint = age <= value
            elif "<" in op:
                constraint = age < value
            else:
                constraint = age == value
            s.add(Not(constraint))
            return s, model_vars

    # 6) GPA constraints (e.g., self.gpa >= 0.0 and self.gpa <= 4.0)
    if "gpa" in text and (">=" in text or ">" in text or "<=" in text or "<" in text):
        from z3 import Real
        gpa = Real('gpa')
        model_vars['gpa'] = gpa
        # Handle range constraints
        if "and" in text:
            # Extract both bounds
            lower_match = re.search(r"gpa\s*>=\s*([\d.]+)", text)
            upper_match = re.search(r"gpa\s*<=\s*([\d.]+)", text)
            constraints = []
            if lower_match:
                constraints.append(gpa >= float(lower_match.group(1)))
            if upper_match:
                constraints.append(gpa <= float(upper_match.group(1)))
            if constraints:
                s.add(Not(And(constraints)))
                return s, model_vars

    # 7) Credits and maxSeats constraints
    for attr in ["credits", "maxSeats", "capacity", "experienceYears"]:
        if attr in text and (">=" in text or ">" in text or "<=" in text or "<" in text):
            var = Int(attr)
            model_vars[attr] = var
            constraint_match = re.search(f"{attr}\\s*([><=!]+)\\s*(\\d+)", text)
            if constraint_match:
                op, value = constraint_match.group(1), int(constraint_match.group(2))
                if ">=" in op:
                    constraint = var >= value
                elif ">" in op:
                    constraint = var > value
                elif "<=" in op:
                    constraint = var <= value
                elif "<" in op:
                    constraint = var < value
                else:
                    constraint = var == value
                s.add(Not(constraint))
                return s, model_vars

    # 8) forAll over collections with bounded scope
    m = FORALL_RE.search(text)
    if m:
        var, body = m.group(1), m.group(2)
        klass, assoc = guess_collection_class(text, model_mappings)
        n = resolve_count(klass, cfg, default=3)
        # Better body analysis
        if "." in body:
            attr = body.split(".")[-1].split()[0]  # Extract attribute name
            arr = int_array(f"{assoc}_{attr}", n)
        else:
            arr = int_array(f"{assoc}_elements", n)
        for i, sym in enumerate(arr):
            model_vars[f"{assoc}_{i}"] = sym
        # Parse the body condition more carefully
        if ">" in body and "0" in body:
            constraint = And([sym > 0 for sym in arr])
        elif ">=" in body:
            value_match = re.search(r">=\s*(\d+)", body)
            val = int(value_match.group(1)) if value_match else 0
            constraint = And([sym >= val for sym in arr])
        elif "<" in body:
            value_match = re.search(r"<\s*(\d+)", body)
            val = int(value_match.group(1)) if value_match else 100
            constraint = And([sym < val for sym in arr])
        elif "<>" in body and "null" in body:
            # Handle null checks in forAll
            constraint = And([Bool(f"element_{i}_present") for i in range(len(arr))])
            for i in range(len(arr)):
                model_vars[f"element_{i}_present"] = Bool(f"element_{i}_present")
        else:
            constraint = And([sym > 0 for sym in arr])  # Default
        s.add(Not(constraint))
        return s, model_vars

    # 9) exists over collections
    m2 = EXISTS_RE.search(text)
    if m2:
        var, body = m2.group(1), m2.group(2)
        klass, assoc = guess_collection_class(text, model_mappings)
        n = resolve_count(klass, cfg, default=3)
        arr = int_array(f"{assoc}_attr", n)
        for i, sym in enumerate(arr):
            model_vars[f"{assoc}_{i}"] = sym
        # exists means at least one satisfies condition
        if ">" in body and "0" in body:
            constraint = Or([sym > 0 for sym in arr])
        elif "==" in body or "=" in body:
            value_match = re.search(r"==?\s*(\d+)", body)
            val = int(value_match.group(1)) if value_match else 0
            constraint = Or([sym == val for sym in arr])
        else:
            constraint = Or([sym == 0 for sym in arr])  # Default
        s.add(Not(constraint))
        return s, model_vars

    # 10) oclIsKindOf(Enum) with university enums
    if "oclIsKindOf" in text:
        enum_m = re.search(r"oclIsKindOf\(\s*(\w+)\s*\)", text)
        literal = enum_m.group(1) if enum_m else None
        # Try university enums first
        enum_name = "Semester" if literal in ["Spring", "Summer", "Fall"] else "Degree"
        literals = resolve_enum(enum_name, cfg) or ["Spring", "Summer", "Fall"]
        EnumType, ctors = enum_sort(enum_name, literals)
        selfType = Int("selfType")
        model_vars['selfType'] = selfType
        if literal in literals:
            constraint = selfType == literals.index(literal)
        else:
            constraint = selfType >= 0
        s.add(Not(constraint))
        return s, model_vars

    # Fallback: unknown → satisfiable placeholder
    s.add(True)
    return s, model_vars
