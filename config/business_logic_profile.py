"""
Business Logic Profile - Generates semantically realistic constraints
Optimized for real-world business rules, not just ML training
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.generation.benchmark.bench_config import BenchmarkProfile

def get_business_logic_profile() -> BenchmarkProfile:
    """
    Profile optimized for generating realistic business logic constraints.
    
    Key differences from lib_easy/standard/hard:
    - Suppresses semantically weird patterns (equality of unrelated attributes)
    - Boosts comparison patterns (<=, >=, <, >) for temporal/numeric
    - Prioritizes validation rules (not null, ranges, uniqueness)
    - Emphasizes cardinality and navigation (common in business rules)
    """
    profile = BenchmarkProfile()
    
    # === QUANTITIES ===
    profile.quantities.invariants = 50
    profile.quantities.per_class_min = 2
    profile.quantities.per_class_max = 8
    
    # === FAMILY DISTRIBUTION (Business-Focused) ===
    profile.quantities.families_pct = {
        "cardinality": 30,      # Collection sizes, multiplicity (VERY common)
        "navigation": 20,       # Association traversal (common)
        "arithmetic": 15,       # Numeric comparisons, ranges
        "uniqueness": 15,       # Unique keys, no duplicates (common)
        "quantified": 10,       # forAll, exists (moderately common)
        "string": 5,            # String validation (less common)
        "type_checks": 5,       # Type checking (rare but useful)
        "enum": 0,              # Enums (skip if not in metamodel)
    }
    
    # === PATTERN WEIGHTS (Key to Business Logic Quality) ===
    profile.library.weights = {
        # ============================================
        # SUPPRESS: Semantically Weird Patterns
        # ============================================
        'two_attributes_equal': 0.0,           #  Disable (causes dateFrom = dateTo)
        'two_attributes_not_equal': 0.0,       #  Disable (weird in business logic)
        
        # ============================================
        # BOOST: Comparison Patterns (Business Rules)
        # ============================================
        'numeric_comparison': 10.0,            #  dateFrom <= dateTo, price >= 0
        'numeric_greater_than_value': 8.0,     #  age > 18, amount > 0
        'numeric_less_than_value': 8.0,        #  discount < 100
        'numeric_bounded': 10.0,               #  age >= 18 and age <= 120
        'range_constraint': 10.0,              #  temperature >= 0 and <= 100
        
        # ============================================
        # BOOST: Validation Rules (Very Common)
        # ============================================
        'attribute_not_null_simple': 15.0,     #  name <> null
        'attribute_defined': 12.0,             #  email.isDefined()
        'string_not_empty': 12.0,              #  name <> '' and name.size() > 0
        'string_min_length': 8.0,              #  password.size() >= 8
        'numeric_positive': 10.0,              #  price > 0
        'numeric_non_negative': 10.0,          #  quantity >= 0
        
        # ============================================
        # BOOST: Collection/Cardinality (Very Common)
        # ============================================
        'collection_has_size': 12.0,           #  rentals->size() = 5
        'collection_size_range': 10.0,         #  employees->size() >= 1 and <= 100
        'collection_min_size': 10.0,           #  team->size() >= 2
        'collection_not_empty_simple': 10.0,   #  rentals->notEmpty()
        'size_constraint': 10.0,               #  vehicles->size() >= 3
        
        # ============================================
        # BOOST: Uniqueness (Common in Business)
        # ============================================
        'uniqueness_constraint': 15.0,         #  customers->isUnique(c | c.email)
        'collection_isUnique_attr': 12.0,      #  vehicles->isUnique(v | v.licensePlate)
        
        # ============================================
        # BOOST: Navigation (Common)
        # ============================================
        'association_exists': 10.0,            #  rental.customer <> null
        'navigation_depth_2': 8.0,             #  self.rental.vehicle.status = 'available'
        
        # ============================================
        # BOOST: Implications (Business Rules)
        # ============================================
        'implies_simple': 10.0,                #  isPremium implies discountRate > 0
        'conditional_if_then_else': 8.0,       #  if isPremium then discount > 10 else true
        
        # ============================================
        # MODERATE: Quantifiers (Less Common)
        # ============================================
        'forall_exists': 5.0,                  #  Moderately common
        'select_operation': 5.0,               #  Moderately common
        'collect_operation': 3.0,              #  Less common
        
        # ============================================
        # SUPPRESS: Complex/Rare Patterns
        # ============================================
        'closure_transitive': 0.5,             #  Rare in business logic
        'nested_quantifiers': 0.5,             #  Complex, rare
        'tuple_type': 0.0,                     #  Very rare
        
        # ============================================
        # BOOST: New Type Operations (Useful)
        # ============================================
        'oclIsKindOf_check': 5.0,              #  Useful for inheritance
        'oclIsTypeOf_check': 3.0,              #  Less common
        'allInstances_check': 2.0,             #  Rare in business rules
        
        # ============================================
        # BOOST: New Collection Operations
        # ============================================
        'collection_sortedBy': 5.0,            #  Moderately useful
        'collection_sum': 8.0,                 #  Total amounts, counts
        'collection_first': 6.0,               #  Latest, oldest record
        'collection_last': 6.0,                #  Latest, oldest record
        'collection_any_match': 4.0,           #  Less common than forAll
        
        # ============================================
        # MODERATE: Boolean Logic
        # ============================================
        'boolean_is_true': 8.0,                #  isActive = true
        'boolean_is_false': 6.0,               #  isDeleted = false
        'xor_condition': 2.0,                  #  Rare in business logic
    }
    
    # === COVERAGE TARGETS ===
    profile.coverage.class_context_pct = 90        # Cover most classes
    profile.coverage.operator_mins = {
        'forAll': 3,
        'exists': 2,
        'select': 2,
        'size': 8,
        'implies': 3,
    }
    
    # === DIFFICULTY MIX (Business Logic) ===
    profile.coverage.difficulty_mix = {
        'easy': 50,      # Lots of simple validation rules
        'medium': 35,    # Some navigation, implications
        'hard': 15,      # Few complex quantifiers
    }
    
    # === REDUNDANCY ===
    profile.redundancy.similarity_threshold = 0.80   # Allow some variety
    profile.redundancy.novelty_boost = True          # Encourage diversity
    
    return profile


# === USAGE EXAMPLE ===
if __name__ == '__main__':
    profile = get_business_logic_profile()
    print("Business Logic Profile Created")
    print(f"Target constraints: {profile.quantities.invariants}")
    print(f"Families: {profile.quantities.families_pct}")
    print(f"Key weights: two_attributes_equal={profile.library.weights.get('two_attributes_equal', 1.0)}")
    print(f"Key weights: numeric_comparison={profile.library.weights.get('numeric_comparison', 1.0)}")
