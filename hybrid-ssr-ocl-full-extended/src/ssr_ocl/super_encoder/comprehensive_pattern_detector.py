"""
Comprehensive Pattern Detector with 500+ Regex Patterns
========================================================

This module provides extensive regex pattern matching for all 50 OCL pattern types,
with approximately 10+ regex variations per pattern to handle different syntactic forms.

Pattern coverage:
- All 50 OCL pattern types from OCLPatternType enum
- Multiple regex variants per pattern type
- Handles different OCL syntactic styles
- Priority-ordered for accuracy (specific → general)
"""

import re
from typing import Optional

try:
    from ..classifiers.sentence_transformer.classifier import OCLPatternType
except ImportError:
    from src.ssr_ocl.classifiers.sentence_transformer.classifier import OCLPatternType


class ComprehensivePatternDetector:
    """
    Comprehensive regex-based OCL pattern detector with 500+ patterns.
    
    Features:
    - ~10 regex variants per OCL pattern type
    - Priority ordering (more specific patterns first)
    - Case-insensitive matching for keywords
    - Handles whitespace variations
    """
    
    # 500+ comprehensive regex patterns, ordered by specificity
    COMPREHENSIVE_PATTERNS = [
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. PAIRWISE_UNIQUENESS (14 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'forAll\s*\(\s*\w+\s*,\s*\w+\s*\|[^)]*<>[^)]*implies', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'->forAll\s*\([^|]+,\s*[^|]+\|[^)]*<>[^)]*implies', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'->forAll\([^)]*,\s*[^)]*\|[^)]*!=', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'allInstances\(\)->forAll\([^,]+,\s*[^|]+\|', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'forAll\s*\(\s*\w+\s*:\s*\w+\s*,\s*\w+\s*:\s*\w+', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'->forAll\([a-z]\s*,\s*[a-z]\s*\|.*<>', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'forAll.*\|.*implies.*isUnique', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'->select.*->forAll\([^,]+,\s*[^)]+\|', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'forAll\([^)]*\)\s*and\s*forAll\([^)]*<>', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'allInstances.*forAll.*,.*\|.*<>.*implies', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'->forAll\s*\([ij]\s*,\s*[jk]', OCLPatternType.PAIRWISE_UNIQUENESS),
        (r'->collect.*->forAll\([^,]+,\s*[^|]+\|.*<>', OCLPatternType.PAIRWISE_UNIQUENESS),
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. EXACT_COUNT_SELECTION (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->select\([^)]+\)->size\(\)\s*=\s*\d+', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->select\([^)]+\)->count\([^)]+\)\s*=\s*\d+', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->reject\([^)]+\)->size\(\)\s*=\s*\d+', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->select.*->size\(\)\s*=\s*[0-9]+', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->select.*->size\(\)\s*=\s*self\.', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->select\(\w+\s*\|[^)]+\)->size\(\)\s*>=?\s*\d+', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->select\([^)]+\)->size\(\)\s*<=?\s*\d+', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->select.*->size\(\)\s*<>\s*\d+', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->collect.*->select.*->size\(\)\s*=', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->asSet\(\)->select.*->size\(\)\s*=', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->reject\(\w+\s*\|.*\)->size\(\)\s*=\s*\d+', OCLPatternType.EXACT_COUNT_SELECTION),
        (r'->selectByKind.*->size\(\)\s*=\s*\d+', OCLPatternType.EXACT_COUNT_SELECTION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. GLOBAL_COLLECTION (16 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\w+\.allInstances\(\)', OCLPatternType.GLOBAL_COLLECTION),
        (r'\w+::allInstances\(\)', OCLPatternType.GLOBAL_COLLECTION),
        (r'[A-Z]\w*\.allInstances\(\)->', OCLPatternType.GLOBAL_COLLECTION),
        (r'[A-Z]\w*::allInstances\(\)->', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->select', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->forAll', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->exists', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->collect', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->size\(\)', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->isUnique', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->includes', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->one\(', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->any\(', OCLPatternType.GLOBAL_COLLECTION),
        (r'[A-Z]\w*\.allInstances\(\)->asSet', OCLPatternType.GLOBAL_COLLECTION),
        (r'::allInstances\(\)->reject', OCLPatternType.GLOBAL_COLLECTION),
        (r'allInstances\(\)->selectByType', OCLPatternType.GLOBAL_COLLECTION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. SET_INTERSECTION (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->intersection\(', OCLPatternType.SET_INTERSECTION),
        (r'->intersect\(', OCLPatternType.SET_INTERSECTION),
        (r'->asSet\(\)->intersection\(', OCLPatternType.SET_INTERSECTION),
        (r'\.intersection\(\w+\)', OCLPatternType.SET_INTERSECTION),
        (r'->intersection\([^)]+\)->notEmpty', OCLPatternType.SET_INTERSECTION),
        (r'->intersection\([^)]+\)->isEmpty', OCLPatternType.SET_INTERSECTION),
        (r'->intersection\([^)]+\)->size', OCLPatternType.SET_INTERSECTION),
        (r'->collect.*->intersection', OCLPatternType.SET_INTERSECTION),
        (r'->select.*->intersection', OCLPatternType.SET_INTERSECTION),
        (r'intersection\([^)]+\)->includes', OCLPatternType.SET_INTERSECTION),
        
        # ═══════════════════════════════════════════════════════════════════
        # UNION with size() - MUST come before SIZE_CONSTRAINT to avoid false matching
        # ═══════════════════════════════════════════════════════════════════
        (r'->union\([^)]+\)->size\(\)\s*[<>=]', OCLPatternType.UNION_INTERSECTION),
        (r'->asSet\(\)->union.*->size\(\)', OCLPatternType.UNION_INTERSECTION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. SIZE_CONSTRAINT (18 patterns) - MUST come before SHORTHAND_NOTATION
        # ═══════════════════════════════════════════════════════════════════
        # More specific: direct size() calls with comparison operators
        (r'self\.\w+->size\(\)\s*>=?\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'self\.\w+->size\(\)\s*<=?\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'self\.\w+->size\(\)\s*=\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'self\.\w+->size\(\)\s*<>\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'self\.\w+->size\(\)\s*>\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'self\.\w+->size\(\)\s*<\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*>=?\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*<=?\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*=\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*<>\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*>\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*<\s*\d+', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*=\s*self\.', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*>=\s*self\.', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*<=\s*self\.', OCLPatternType.SIZE_CONSTRAINT),
        (r'->notEmpty\(\)\s*and.*->size\(\)', OCLPatternType.SIZE_CONSTRAINT),
        (r'->notEmpty\(\).*implies.*->size\(\)', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*>\s*0', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*>=\s*1', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*=\s*0', OCLPatternType.SIZE_CONSTRAINT),
        (r'->count\([^)]+\)\s*[<>=]', OCLPatternType.SIZE_CONSTRAINT),
        (r'->isEmpty\(\)\s*or.*->size', OCLPatternType.SIZE_CONSTRAINT),
        (r'->size\(\)\s*in\s*\d+\.\.\d+', OCLPatternType.SIZE_CONSTRAINT),
        
        # ═══════════════════════════════════════════════════════════════════
        # 6. UNIQUENESS_CONSTRAINT (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->isUnique\(', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->isUnique\(\w+\s*\|', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->isUnique\(\s*\w+\s*\)', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->collect.*->isUnique\(', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->select.*->isUnique\(', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'allInstances\(\)->isUnique\(', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->asSet\(\)->size\(\)\s*=.*->size\(\)', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->isUnique\(\w+\s*\|\s*\w+\.\w+\)', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->flatten\(\)->isUnique', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->isUnique\([^)]*oclAsType', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->reject.*->isUnique', OCLPatternType.UNIQUENESS_CONSTRAINT),
        (r'->isUnique\(\w+\s*\|\s*\w+\s*\+', OCLPatternType.UNIQUENESS_CONSTRAINT),
        
        # ═══════════════════════════════════════════════════════════════════
        # 7. COLLECTION_MEMBERSHIP (20 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->includes\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->excludes\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->includesAll\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->excludesAll\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->includes\(self\)', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->excludes\(null\)', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->notEmpty\(\)\s*and.*->includes', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->includesAll\(.*->asSet', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->excludesAll\(.*->asSet', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->includes\(\w+\)\s*and.*->includes', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'not.*->includes\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'not.*->excludes\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->collect.*->includes\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->select.*->includes\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->includes\(.*oclAsType', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->excludes\(.*oclIsKindOf', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'allInstances\(\)->includes\(', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->asSet\(\)->includes', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->flatten\(\)->includes', OCLPatternType.COLLECTION_MEMBERSHIP),
        (r'->union.*->includes', OCLPatternType.COLLECTION_MEMBERSHIP),
        
        # ═══════════════════════════════════════════════════════════════════
        # 8. NULL_CHECK (14 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\w+\s*<>\s*null', OCLPatternType.NULL_CHECK),
        (r'\w+\s*=\s*null', OCLPatternType.NULL_CHECK),
        (r'null\s*<>\s*\w+', OCLPatternType.NULL_CHECK),
        (r'null\s*=\s*\w+', OCLPatternType.NULL_CHECK),
        (r'self\.\w+\s*<>\s*null', OCLPatternType.NULL_CHECK),
        (r'self\.\w+\s*=\s*null', OCLPatternType.NULL_CHECK),
        (r'\w+\s*!=\s*null', OCLPatternType.NULL_CHECK),
        (r'\w+\s*==\s*null', OCLPatternType.NULL_CHECK),
        (r'not\s*\(\s*\w+\s*=\s*null\s*\)', OCLPatternType.NULL_CHECK),
        (r'\w+\.oclIsUndefined\(\)', OCLPatternType.NULL_CHECK),
        (r'isDefined\(\)', OCLPatternType.NULL_CHECK),
        (r'->notEmpty\(\)\s*and.*<>\s*null', OCLPatternType.NULL_CHECK),
        (r'<>\s*null\s*and', OCLPatternType.NULL_CHECK),
        (r'=\s*null\s*or', OCLPatternType.NULL_CHECK),
        
        # ═══════════════════════════════════════════════════════════════════
        # 9. NUMERIC_COMPARISON (moved to end - generic)
        # ═══════════════════════════════════════════════════════════════════
        
        # ═══════════════════════════════════════════════════════════════════
        # 10. EXACTLY_ONE (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->one\(', OCLPatternType.EXACTLY_ONE),
        (r'->one\(\w+\s*\|', OCLPatternType.EXACTLY_ONE),
        (r'->select.*->one\(', OCLPatternType.EXACTLY_ONE),
        (r'allInstances\(\)->one\(', OCLPatternType.EXACTLY_ONE),
        (r'->collect.*->one\(', OCLPatternType.EXACTLY_ONE),
        (r'->asSet\(\)->one\(', OCLPatternType.EXACTLY_ONE),
        (r'->one\([^)]*oclIsKindOf', OCLPatternType.EXACTLY_ONE),
        (r'->flatten\(\)->one\(', OCLPatternType.EXACTLY_ONE),
        (r'->one\(\w+\s*\|\s*\w+\s*=', OCLPatternType.EXACTLY_ONE),
        (r'->reject.*->one\(', OCLPatternType.EXACTLY_ONE),
        
        # ═══════════════════════════════════════════════════════════════════
        # 11. CLOSURE_TRANSITIVE (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->closure\(', OCLPatternType.CLOSURE),
        (r'->closure\(\w+\s*\|', OCLPatternType.CLOSURE),
        (r'->closure\(\s*\w+\s*\)', OCLPatternType.CLOSURE),
        (r'self->closure\(', OCLPatternType.CLOSURE),
        (r'->closure\([^)]+\)->includes', OCLPatternType.CLOSURE),
        (r'->closure\([^)]+\)->excludes', OCLPatternType.CLOSURE),
        (r'->closure\([^)]+\)->forAll', OCLPatternType.CLOSURE),
        (r'->closure\([^)]+\)->exists', OCLPatternType.CLOSURE),
        (r'->closure\([^)]+\)->select', OCLPatternType.CLOSURE),
        (r'->closure\([^)]+\)->size', OCLPatternType.CLOSURE),
        (r'->collect.*->closure', OCLPatternType.CLOSURE),
        (r'->asSet\(\)->closure', OCLPatternType.CLOSURE),
        
        # ═══════════════════════════════════════════════════════════════════
        # 12. ACYCLICITY (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'not\s+self->closure.*->includes\(self\)', OCLPatternType.ACYCLICITY),
        (r'not.*->closure.*->includes\(self\)', OCLPatternType.ACYCLICITY),
        (r'self->closure.*->excludes\(self\)', OCLPatternType.ACYCLICITY),
        (r'->closure.*->excludes\(self\)', OCLPatternType.ACYCLICITY),
        (r'not.*closure.*includes.*self', OCLPatternType.ACYCLICITY),
        (r'->closure\([^)]+\)->excludes\(self\)', OCLPatternType.ACYCLICITY),
        (r'not\s*\(.*->closure.*->includes\(self\)\s*\)', OCLPatternType.ACYCLICITY),
        (r'self\s*not\s*in\s*self->closure', OCLPatternType.ACYCLICITY),
        (r'closure.*excludes.*self', OCLPatternType.ACYCLICITY),
        (r'not.*self.*closure.*includes', OCLPatternType.ACYCLICITY),
        
        # ═══════════════════════════════════════════════════════════════════
        # 13. AGGREGATION_ITERATE (14 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->iterate\(', OCLPatternType.ITERATE),
        (r'->iterate\(\w+\s*;', OCLPatternType.ITERATE),
        (r'->iterate\(\w+\s*:\s*\w+\s*;', OCLPatternType.ITERATE),
        (r'->iterate\([^|]+\|', OCLPatternType.ITERATE),
        (r'->collect.*->iterate', OCLPatternType.ITERATE),
        (r'->select.*->iterate', OCLPatternType.ITERATE),
        (r'allInstances\(\)->iterate', OCLPatternType.ITERATE),
        (r'->iterate\([^)]*acc\s*:', OCLPatternType.ITERATE),
        (r'->iterate\([^)]*result\s*:', OCLPatternType.ITERATE),
        (r'->asSet\(\)->iterate', OCLPatternType.ITERATE),
        (r'->flatten\(\)->iterate', OCLPatternType.ITERATE),
        (r'->iterate\([^)]*;\s*\w+\s*=', OCLPatternType.ITERATE),
        (r'->reject.*->iterate', OCLPatternType.ITERATE),
        (r'->sortedBy.*->iterate', OCLPatternType.ITERATE),
        
        # ═══════════════════════════════════════════════════════════════════
        # 14. BOOLEAN_GUARD_IMPLIES (16 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\w+\s*<>\s*null\s+implies', OCLPatternType.IMPLIES),
        (r'->notEmpty\(\)\s*implies', OCLPatternType.IMPLIES),
        (r'->isEmpty\(\)\s*implies', OCLPatternType.IMPLIES),
        (r'isDefined\(\)\s*implies', OCLPatternType.IMPLIES),
        (r'not.*implies', OCLPatternType.IMPLIES),
        (r'\w+\s*=\s*null\s+implies', OCLPatternType.IMPLIES),
        (r'self\.\w+\s*<>\s*null\s+implies', OCLPatternType.IMPLIES),
        (r'->includes.*implies', OCLPatternType.IMPLIES),
        (r'->excludes.*implies', OCLPatternType.IMPLIES),
        (r'->size\(\)\s*>\s*0\s*implies', OCLPatternType.IMPLIES),
        (r'oclIsKindOf.*implies', OCLPatternType.IMPLIES),
        (r'oclIsTypeOf.*implies', OCLPatternType.IMPLIES),
        (r'\w+\s*and.*implies', OCLPatternType.IMPLIES),
        (r'\w+\s*or.*implies', OCLPatternType.IMPLIES),
        (r'->forAll.*implies.*implies', OCLPatternType.IMPLIES),
        (r'->exists.*implies', OCLPatternType.IMPLIES),
        
        # ═══════════════════════════════════════════════════════════════════
        # 15. SAFE_NAVIGATION (8 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\w+\?\.\w+', OCLPatternType.SAFE_NAVIGATION),
        (r'\w+\?\.\w+\?\.\w+', OCLPatternType.SAFE_NAVIGATION),
        (r'self\?\.\w+', OCLPatternType.SAFE_NAVIGATION),
        (r'\w+\?->\w+', OCLPatternType.SAFE_NAVIGATION),
        (r'\.\w+\?\.\w+', OCLPatternType.SAFE_NAVIGATION),
        (r'\w+\?\.oclAsType', OCLPatternType.SAFE_NAVIGATION),
        (r'\w+\?\.\w+\(\)', OCLPatternType.SAFE_NAVIGATION),
        (r'self\.\w+\?\.\w+', OCLPatternType.SAFE_NAVIGATION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 16. TYPE_CHECK_CASTING (16 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\.oclIsKindOf\(', OCLPatternType.TYPE_CHECK),
        (r'\.oclIsTypeOf\(', OCLPatternType.TYPE_CHECK),
        (r'->select\([^)]*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'->select\([^)]*oclIsTypeOf', OCLPatternType.TYPE_CHECK),
        (r'->forAll\([^)]*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'->exists\([^)]*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'->reject\([^)]*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'if.*oclIsKindOf.*then', OCLPatternType.TYPE_CHECK),
        (r'oclIsKindOf.*implies', OCLPatternType.TYPE_CHECK),
        (r'not.*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'oclIsTypeOf.*implies', OCLPatternType.TYPE_CHECK),
        (r'->any\([^)]*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'->one\([^)]*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'->collect\([^)]*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'oclIsKindOf.*and.*oclIsKindOf', OCLPatternType.TYPE_CHECK),
        (r'self\.oclIsKindOf', OCLPatternType.TYPE_CHECK),
        
        # ═══════════════════════════════════════════════════════════════════
        # 17. SUBSET_DISJOINTNESS (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->intersection\([^)]+\)->isEmpty\(\)', OCLPatternType.SUBSET_DISJOINT),
        (r'->intersection\([^)]+\)->size\(\)\s*=\s*0', OCLPatternType.SUBSET_DISJOINT),
        (r'->intersection.*->notEmpty\(\)', OCLPatternType.SUBSET_DISJOINT),
        (r'not.*->intersection.*->notEmpty', OCLPatternType.SUBSET_DISJOINT),
        (r'->asSet\(\)->intersection.*->isEmpty', OCLPatternType.SUBSET_DISJOINT),
        (r'->includesAll.*and.*->excludesAll', OCLPatternType.SUBSET_DISJOINT),
        (r'->forAll\([^)]*not.*->includes', OCLPatternType.SUBSET_DISJOINT),
        (r'->select.*->intersection.*->isEmpty', OCLPatternType.SUBSET_DISJOINT),
        (r'->collect.*->intersection.*->isEmpty', OCLPatternType.SUBSET_DISJOINT),
        (r'->symmetricDifference.*->size\(\)\s*=.*->size', OCLPatternType.SUBSET_DISJOINT),
        
        # ═══════════════════════════════════════════════════════════════════
        # 18. ORDERING_RANKING (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->sortedBy\(', OCLPatternType.ORDERING),
        (r'->sortedBy\(\w+\s*\|', OCLPatternType.ORDERING),
        (r'->sortedBy\(\s*\w+\s*\)', OCLPatternType.ORDERING),
        (r'->select.*->sortedBy', OCLPatternType.ORDERING),
        (r'->collect.*->sortedBy', OCLPatternType.ORDERING),
        (r'allInstances\(\)->sortedBy', OCLPatternType.ORDERING),
        (r'->asSet\(\)->sortedBy', OCLPatternType.ORDERING),
        (r'->sortedBy.*->first\(\)', OCLPatternType.ORDERING),
        (r'->sortedBy.*->last\(\)', OCLPatternType.ORDERING),
        (r'->sortedBy.*->at\(', OCLPatternType.ORDERING),
        (r'->reject.*->sortedBy', OCLPatternType.ORDERING),
        (r'->flatten\(\)->sortedBy', OCLPatternType.ORDERING),
        
        # ═══════════════════════════════════════════════════════════════════
        # 19. CONTRACTUAL_TEMPORAL (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->notEmpty\(\)\s+implies.*=', OCLPatternType.CONTRACTUAL),
        (r'isDefined\(\)\s*implies.*=', OCLPatternType.CONTRACTUAL),
        (r'<>\s*null\s*implies.*=', OCLPatternType.CONTRACTUAL),
        (r'->size\(\)\s*>\s*0\s*implies.*=', OCLPatternType.CONTRACTUAL),
        (r'->includes.*implies.*=', OCLPatternType.CONTRACTUAL),
        (r'->notEmpty.*implies.*>=', OCLPatternType.CONTRACTUAL),
        (r'->notEmpty.*implies.*<=', OCLPatternType.CONTRACTUAL),
        (r'->forAll.*implies.*=', OCLPatternType.CONTRACTUAL),
        (r'->exists.*implies.*=', OCLPatternType.CONTRACTUAL),
        (r'oclIsKindOf.*implies.*=', OCLPatternType.CONTRACTUAL),
        
        # ═══════════════════════════════════════════════════════════════════
        # 20. SELECT_REJECT (16 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->select\(', OCLPatternType.SELECT_REJECT),
        (r'->reject\(', OCLPatternType.SELECT_REJECT),
        (r'->select\(\w+\s*\|', OCLPatternType.SELECT_REJECT),
        (r'->reject\(\w+\s*\|', OCLPatternType.SELECT_REJECT),
        (r'->select\([^)]*<>.*\)', OCLPatternType.SELECT_REJECT),
        (r'->select\([^)]*=.*\)', OCLPatternType.SELECT_REJECT),
        (r'->reject\([^)]*=.*\)', OCLPatternType.SELECT_REJECT),
        (r'->select\([^)]*>.*\)', OCLPatternType.SELECT_REJECT),
        (r'->select\([^)]*<.*\)', OCLPatternType.SELECT_REJECT),
        (r'->collect.*->select', OCLPatternType.SELECT_REJECT),
        (r'->select.*->select', OCLPatternType.SELECT_REJECT),
        (r'allInstances\(\)->select', OCLPatternType.SELECT_REJECT),
        (r'->asSet\(\)->select', OCLPatternType.SELECT_REJECT),
        (r'->flatten\(\)->select', OCLPatternType.SELECT_REJECT),
        (r'->select.*->reject', OCLPatternType.SELECT_REJECT),
        (r'->selectByType\(', OCLPatternType.SELECT_REJECT),
        
        # ═══════════════════════════════════════════════════════════════════
        # 21. COLLECT_FLATTEN (14 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->collect\([^)]+\)->flatten\(\)', OCLPatternType.COLLECT_FLATTEN),
        (r'->collect.*->collect.*->flatten', OCLPatternType.COLLECT_FLATTEN),
        (r'->flatten\(\)', OCLPatternType.FLATTEN_OPERATION),
        (r'->select.*->collect.*->flatten', OCLPatternType.COLLECT_FLATTEN),
        (r'allInstances\(\)->collect.*->flatten', OCLPatternType.COLLECT_FLATTEN),
        (r'->collect\(\w+\s*\|\s*\w+\.\w+\)->flatten', OCLPatternType.COLLECT_FLATTEN),
        (r'->asSet\(\)->collect.*->flatten', OCLPatternType.COLLECT_FLATTEN),
        (r'->reject.*->collect.*->flatten', OCLPatternType.COLLECT_FLATTEN),
        (r'->flatten\(\)->select', OCLPatternType.FLATTEN_OPERATION),
        (r'->flatten\(\)->forAll', OCLPatternType.FLATTEN_OPERATION),
        (r'->flatten\(\)->exists', OCLPatternType.FLATTEN_OPERATION),
        (r'->flatten\(\)->size', OCLPatternType.FLATTEN_OPERATION),
        (r'->flatten\(\)->asSet', OCLPatternType.FLATTEN_OPERATION),
        (r'->collect.*->flatten\(\)->collect', OCLPatternType.COLLECT_FLATTEN),
        
        # ═══════════════════════════════════════════════════════════════════
        # 22. ANY_OPERATION (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->any\(', OCLPatternType.ANY_OPERATION),
        (r'->any\(\w+\s*\|', OCLPatternType.ANY_OPERATION),
        (r'->any\([^)]*=', OCLPatternType.ANY_OPERATION),
        (r'->select.*->any\(', OCLPatternType.ANY_OPERATION),
        (r'allInstances\(\)->any\(', OCLPatternType.ANY_OPERATION),
        (r'->collect.*->any\(', OCLPatternType.ANY_OPERATION),
        (r'->asSet\(\)->any\(', OCLPatternType.ANY_OPERATION),
        (r'->any\([^)]*oclIsKindOf', OCLPatternType.ANY_OPERATION),
        (r'->any\([^)]*>.*\)', OCLPatternType.ANY_OPERATION),
        (r'->any\([^)]*<.*\)', OCLPatternType.ANY_OPERATION),
        (r'->reject.*->any\(', OCLPatternType.ANY_OPERATION),
        (r'->flatten\(\)->any\(', OCLPatternType.ANY_OPERATION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 23. FORALL_NESTED (18 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->forAll\(', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\(\w+\s*\|', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\([^)]*implies', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\([^)]*>.*\)', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\([^)]*<.*\)', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\([^)]*=.*\)', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\([^)]*<>.*\)', OCLPatternType.FOR_ALL_NESTED),
        (r'->select.*->forAll', OCLPatternType.FOR_ALL_NESTED),
        (r'->collect.*->forAll', OCLPatternType.FOR_ALL_NESTED),
        (r'allInstances\(\)->forAll', OCLPatternType.FOR_ALL_NESTED),
        (r'->asSet\(\)->forAll', OCLPatternType.FOR_ALL_NESTED),
        (r'->flatten\(\)->forAll', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll.*->forAll', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\([^)]*and', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\([^)]*or', OCLPatternType.FOR_ALL_NESTED),
        (r'->reject.*->forAll', OCLPatternType.FOR_ALL_NESTED),
        (r'->forAll\([^)]*oclIsKindOf', OCLPatternType.FOR_ALL_NESTED),
        (r'->union.*->forAll', OCLPatternType.FOR_ALL_NESTED),
        
        # ═══════════════════════════════════════════════════════════════════
        # 24. EXISTS_NESTED (16 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->exists\(', OCLPatternType.EXISTS_NESTED),
        (r'->exists\(\w+\s*\|', OCLPatternType.EXISTS_NESTED),
        (r'->exists\([^)]*=', OCLPatternType.EXISTS_NESTED),
        (r'->exists\([^)]*<>', OCLPatternType.EXISTS_NESTED),
        (r'->exists\([^)]*>', OCLPatternType.EXISTS_NESTED),
        (r'->exists\([^)]*<', OCLPatternType.EXISTS_NESTED),
        (r'->select.*->exists', OCLPatternType.EXISTS_NESTED),
        (r'->collect.*->exists', OCLPatternType.EXISTS_NESTED),
        (r'allInstances\(\)->exists', OCLPatternType.EXISTS_NESTED),
        (r'->asSet\(\)->exists', OCLPatternType.EXISTS_NESTED),
        (r'->flatten\(\)->exists', OCLPatternType.EXISTS_NESTED),
        (r'->exists.*->exists', OCLPatternType.EXISTS_NESTED),
        (r'->exists\([^)]*and', OCLPatternType.EXISTS_NESTED),
        (r'->reject.*->exists', OCLPatternType.EXISTS_NESTED),
        (r'->exists\([^)]*oclIsKindOf', OCLPatternType.EXISTS_NESTED),
        (r'->union.*->exists', OCLPatternType.EXISTS_NESTED),
        
        # ═══════════════════════════════════════════════════════════════════
        # 25. COLLECT_NESTED (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->collect\([^)]+\)->collect\(', OCLPatternType.COLLECT_NESTED),
        (r'->collect\(\w+\s*\|\s*\w+\.\w+\)->collect', OCLPatternType.COLLECT_NESTED),
        (r'->select.*->collect.*->collect', OCLPatternType.COLLECT_NESTED),
        (r'allInstances\(\)->collect.*->collect', OCLPatternType.COLLECT_NESTED),
        (r'->collect.*->collect.*->collect', OCLPatternType.COLLECT_NESTED),
        (r'->asSet\(\)->collect.*->collect', OCLPatternType.COLLECT_NESTED),
        (r'->reject.*->collect.*->collect', OCLPatternType.COLLECT_NESTED),
        (r'->collect\([^)]+\)->select.*->collect', OCLPatternType.COLLECT_NESTED),
        (r'->flatten\(\)->collect.*->collect', OCLPatternType.COLLECT_NESTED),
        (r'->union.*->collect.*->collect', OCLPatternType.COLLECT_NESTED),
        (r'->collect\(\w+\s*\|.*\)->asSet\(\)->collect', OCLPatternType.COLLECT_NESTED),
        (r'->collect\(', OCLPatternType.COLLECT_FLATTEN),
        
        # ═══════════════════════════════════════════════════════════════════
        # 26. AS_SET_AS_BAG (16 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->asSet\(\)', OCLPatternType.AS_SET_AS_BAG),
        (r'->asBag\(\)', OCLPatternType.AS_SET_AS_BAG),
        (r'->asSequence\(\)', OCLPatternType.AS_SET_AS_BAG),
        (r'->asOrderedSet\(\)', OCLPatternType.AS_SET_AS_BAG),
        (r'->collect.*->asSet\(\)', OCLPatternType.AS_SET_AS_BAG),
        (r'->select.*->asSet\(\)', OCLPatternType.AS_SET_AS_BAG),
        (r'allInstances\(\)->asSet', OCLPatternType.AS_SET_AS_BAG),
        (r'->flatten\(\)->asSet', OCLPatternType.AS_SET_AS_BAG),
        (r'->asSet\(\)->select', OCLPatternType.AS_SET_AS_BAG),
        (r'->asSet\(\)->collect', OCLPatternType.AS_SET_AS_BAG),
        (r'->asBag\(\)->select', OCLPatternType.AS_SET_AS_BAG),
        (r'->asSequence\(\)->first', OCLPatternType.AS_SET_AS_BAG),
        (r'->asSequence\(\)->last', OCLPatternType.AS_SET_AS_BAG),
        (r'->asOrderedSet\(\)->first', OCLPatternType.AS_SET_AS_BAG),
        (r'->reject.*->asSet', OCLPatternType.AS_SET_AS_BAG),
        (r'->union.*->asSet', OCLPatternType.AS_SET_AS_BAG),
        
        # ═══════════════════════════════════════════════════════════════════
        # 27. SUM_PRODUCT (14 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->sum\(\)', OCLPatternType.SUM_PRODUCT),
        (r'->product\(', OCLPatternType.SUM_PRODUCT),
        (r'->collect\([^)]+\)->sum\(\)', OCLPatternType.SUM_PRODUCT),
        (r'->select.*->sum\(\)', OCLPatternType.SUM_PRODUCT),
        (r'->select.*->product\(', OCLPatternType.SUM_PRODUCT),
        (r'allInstances\(\)->collect.*->sum', OCLPatternType.SUM_PRODUCT),
        (r'->asSet\(\)->sum', OCLPatternType.SUM_PRODUCT),
        (r'->flatten\(\)->sum', OCLPatternType.SUM_PRODUCT),
        (r'->reject.*->sum', OCLPatternType.SUM_PRODUCT),
        (r'->collect\(\w+\s*\|\s*\w+\.\w+\)->sum', OCLPatternType.SUM_PRODUCT),
        (r'->sum\(\)\s*[<>=]', OCLPatternType.SUM_PRODUCT),
        (r'->product\([^)]+\)->sum', OCLPatternType.SUM_PRODUCT),
        (r'->union.*->sum', OCLPatternType.SUM_PRODUCT),
        (r'->sortedBy.*->sum', OCLPatternType.SUM_PRODUCT),
        
        # ═══════════════════════════════════════════════════════════════════
        # 28. STRING_CONCAT (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\.concat\(', OCLPatternType.STRING_CONCAT),
        (r'->collect\([^)]+\)\.concat', OCLPatternType.STRING_CONCAT),
        (r'\w+\.concat\(\w+\)\.concat', OCLPatternType.STRING_CONCAT),
        (r'self\.\w+\.concat\(', OCLPatternType.STRING_CONCAT),
        (r'\.concat\(["\']', OCLPatternType.STRING_CONCAT),
        (r'\.concat\(self\.', OCLPatternType.STRING_CONCAT),
        (r'toString\(\)\.concat', OCLPatternType.STRING_CONCAT),
        (r'\.concat.*\.concat', OCLPatternType.STRING_CONCAT),
        (r'if.*then.*concat', OCLPatternType.STRING_CONCAT),
        (r'let.*concat', OCLPatternType.STRING_CONCAT),
        (r'\.substring.*\.concat', OCLPatternType.STRING_CONCAT),
        (r'\.concat\(.*toString\(\)', OCLPatternType.STRING_CONCAT),
        
        # ═══════════════════════════════════════════════════════════════════
        # 29. STRING_OPERATIONS (18 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\.toUpper\(\)', OCLPatternType.STRING_OPERATIONS),
        (r'\.toLower\(\)', OCLPatternType.STRING_OPERATIONS),
        (r'\.substring\(', OCLPatternType.STRING_OPERATIONS),
        (r'\.size\(\)', OCLPatternType.STRING_OPERATIONS),
        (r'\.characters\(\)', OCLPatternType.STRING_OPERATIONS),
        (r'\.toUpperCase\(\)', OCLPatternType.STRING_OPERATIONS),
        (r'\.toLowerCase\(\)', OCLPatternType.STRING_OPERATIONS),
        (r'\.trim\(\)', OCLPatternType.STRING_OPERATIONS),
        (r'\.substring\(\d+\s*,\s*\d+\)', OCLPatternType.STRING_OPERATIONS),
        (r'\.indexOf\(', OCLPatternType.STRING_OPERATIONS),
        (r'\.lastIndexOf\(', OCLPatternType.STRING_OPERATIONS),
        (r'\.startsWith\(', OCLPatternType.STRING_OPERATIONS),
        (r'\.endsWith\(', OCLPatternType.STRING_OPERATIONS),
        (r'self\.\w+\.toUpper', OCLPatternType.STRING_OPERATIONS),
        (r'self\.\w+\.toLower', OCLPatternType.STRING_OPERATIONS),
        (r'\.size\(\)\s*[<>=]', OCLPatternType.STRING_OPERATIONS),
        (r'\.substring.*\.toUpper', OCLPatternType.STRING_OPERATIONS),
        (r'->collect.*\.toUpper', OCLPatternType.STRING_OPERATIONS),
        
        
        # ═══════════════════════════════════════════════════════════════════
        # 31. STRING_PATTERN (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\.matches\(', OCLPatternType.STRING_PATTERN),
        (r'\.matches\(["\'][^"\']*["\']', OCLPatternType.STRING_PATTERN),
        (r'->select\([^)]*\.matches', OCLPatternType.STRING_PATTERN),
        (r'->forAll\([^)]*\.matches', OCLPatternType.STRING_PATTERN),
        (r'->exists\([^)]*\.matches', OCLPatternType.STRING_PATTERN),
        (r'->reject\([^)]*\.matches', OCLPatternType.STRING_PATTERN),
        (r'self\.\w+\.matches', OCLPatternType.STRING_PATTERN),
        (r'if.*\.matches.*then', OCLPatternType.STRING_PATTERN),
        (r'not.*\.matches', OCLPatternType.STRING_PATTERN),
        (r'->collect.*\.matches', OCLPatternType.STRING_PATTERN),
        
        
        # ═══════════════════════════════════════════════════════════════════
        # 33. DIV_MOD_OPERATIONS (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\s+div\s+', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'\s+mod\s+', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'self\.\w+\s+div\s+', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'self\.\w+\s+mod\s+', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'\w+\s+div\s+\d+', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'\w+\s+mod\s+\d+', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'->sum\(\)\s+div', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'->size\(\)\s+div', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'->size\(\)\s+mod', OCLPatternType.DIV_MOD_OPERATIONS),
        (r'\(\w+\s+div\s+\w+\)\s+mod', OCLPatternType.DIV_MOD_OPERATIONS),
        
        # ═══════════════════════════════════════════════════════════════════
        # 34. ABS_MIN_MAX (14 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\.abs\(\)', OCLPatternType.ABS_MIN_MAX),
        (r'\.min\(', OCLPatternType.ABS_MIN_MAX),
        (r'\.max\(', OCLPatternType.ABS_MIN_MAX),
        (r'->collect.*\.abs\(\)', OCLPatternType.ABS_MIN_MAX),
        (r'->select.*\.min\(', OCLPatternType.ABS_MIN_MAX),
        (r'self\.\w+\.abs\(\)', OCLPatternType.ABS_MIN_MAX),
        (r'\(\w+\s*-\s*\w+\)\.abs\(\)', OCLPatternType.ABS_MIN_MAX),
        (r'->sum\(\)\.abs\(\)', OCLPatternType.ABS_MIN_MAX),
        (r'\.min\(\w+,\s*\w+\)', OCLPatternType.ABS_MIN_MAX),
        (r'\.max\(\w+,\s*\w+\)', OCLPatternType.ABS_MIN_MAX),
        (r'->collect.*\.max\(', OCLPatternType.ABS_MIN_MAX),
        (r'\.abs\(\)\s*[<>=]', OCLPatternType.ABS_MIN_MAX),
        (r'->forAll.*\.abs\(\)', OCLPatternType.ABS_MIN_MAX),
        (r'if.*\.abs\(\).*then', OCLPatternType.ABS_MIN_MAX),
        
        # ═══════════════════════════════════════════════════════════════════
        # 35. BOOLEAN_OPERATIONS (16 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\s+and\s+.*\s+and\s+', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'\s+or\s+.*\s+or\s+', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'\s+xor\s+', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'\s+and\s+.*\s+or\s+', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'\s+or\s+.*\s+and\s+', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'not\s*\(.*and.*\)', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'not\s*\(.*or.*\)', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'\(.*and.*\)\s+or', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'\(.*or.*\)\s+and', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'->forAll.*and.*and', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'->exists.*or.*or', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'if.*and.*and.*then', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'\w+\s+and\s+\w+\s+and\s+\w+', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'implies.*and', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'->notEmpty\(\)\s+and.*\s+and', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'<>\s*null\s+and.*\s+and', OCLPatternType.BOOLEAN_OPERATIONS),
        
        # ═══════════════════════════════════════════════════════════════════
        # 36. IF_THEN_ELSE (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'if\s+.*\s+then\s+.*\s+else\s+.*\s+endif', OCLPatternType.IF_THEN_ELSE),
        (r'if\s+.*\s+then\s+.*\s+else', OCLPatternType.IF_THEN_ELSE),
        (r'if.*then.*endif', OCLPatternType.IF_THEN_ELSE),
        (r'if\s+\w+\s*<>\s*null\s+then', OCLPatternType.IF_THEN_ELSE),
        (r'if\s+\w+\s*=\s*null\s+then', OCLPatternType.IF_THEN_ELSE),
        (r'if.*oclIsKindOf.*then', OCLPatternType.IF_THEN_ELSE),
        (r'if.*->notEmpty\(\).*then', OCLPatternType.IF_THEN_ELSE),
        (r'if.*->isEmpty\(\).*then', OCLPatternType.IF_THEN_ELSE),
        (r'if.*and.*then', OCLPatternType.IF_THEN_ELSE),
        (r'if.*or.*then', OCLPatternType.IF_THEN_ELSE),
        (r'->collect.*if.*then', OCLPatternType.IF_THEN_ELSE),
        (r'if.*>.*then.*else.*endif', OCLPatternType.IF_THEN_ELSE),
        
        # ═══════════════════════════════════════════════════════════════════
        # 37. TUPLE_LITERAL (8 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'Tuple\s*\{', OCLPatternType.TUPLE_LITERAL),
        (r'Tuple\s*\{\s*\w+\s*:', OCLPatternType.TUPLE_LITERAL),
        (r'Tuple\s*\{[^}]+,\s*[^}]+\}', OCLPatternType.TUPLE_LITERAL),
        (r'->collect.*Tuple\s*\{', OCLPatternType.TUPLE_LITERAL),
        (r'->select.*Tuple\s*\{', OCLPatternType.TUPLE_LITERAL),
        (r'let.*Tuple\s*\{', OCLPatternType.TUPLE_LITERAL),
        (r'Tuple\s*\{.*=.*,.*=.*\}', OCLPatternType.TUPLE_LITERAL),
        (r'->any.*Tuple\s*\{', OCLPatternType.TUPLE_LITERAL),
        
        # ═══════════════════════════════════════════════════════════════════
        # 38 & 39. LET_EXPRESSION & LET_NESTED (16 patterns) - MUST be before ARITHMETIC
        # ═══════════════════════════════════════════════════════════════════
        (r'let\s+\w+\s*:\s*\w+\s*=.*in\s+let', OCLPatternType.LET_NESTED),
        (r'let\s+\w+\s*=.*in\s+let\s+\w+\s*=', OCLPatternType.LET_NESTED),
        (r'let.*in.*let.*in.*let', OCLPatternType.LET_NESTED),
        # More specific let patterns first
        (r'let\s+\w+\s*=.*->size\(\)\s+in\s+\w+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*=.*->collect.*in\s+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*=.*->select.*in\s+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*=.*->sum.*in\s+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*=.*->forAll.*in\s+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*=.*if.*then.*in\s+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*=.*Tuple\s*\{.*in\s+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*=.*allInstances.*in\s+', OCLPatternType.LET_EXPRESSION),
        # General let expression
        (r'let\s+\w+\s*=.*\s+in\s+\w+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*:\s*\w+\s*=.*\s+in\s+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*=\s*self\..*\s+in\s+', OCLPatternType.LET_EXPRESSION),
        (r'let\s+\w+\s*:\s*Set\(\w+\)\s*=.*\s+in\s+', OCLPatternType.LET_EXPRESSION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 40. UNION_INTERSECTION (11 patterns) - Size-specific patterns moved earlier
        # ═══════════════════════════════════════════════════════════════════
        (r'->union\(', OCLPatternType.UNION_INTERSECTION),
        (r'->union\(\w+\)', OCLPatternType.UNION_INTERSECTION),
        (r'->asSet\(\)->union', OCLPatternType.UNION_INTERSECTION),
        (r'->select.*->union', OCLPatternType.UNION_INTERSECTION),
        (r'->collect.*->union', OCLPatternType.UNION_INTERSECTION),
        (r'allInstances\(\)->union', OCLPatternType.UNION_INTERSECTION),
        (r'->union.*->select', OCLPatternType.UNION_INTERSECTION),
        (r'->union.*->forAll', OCLPatternType.UNION_INTERSECTION),
        (r'->reject.*->union', OCLPatternType.UNION_INTERSECTION),
        (r'->union.*->union', OCLPatternType.UNION_INTERSECTION),
        (r'->flatten\(\)->union', OCLPatternType.UNION_INTERSECTION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 41. SYMMETRIC_DIFFERENCE (8 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->symmetricDifference\(', OCLPatternType.SYMMETRIC_DIFFERENCE),
        (r'->asSet\(\)->symmetricDifference', OCLPatternType.SYMMETRIC_DIFFERENCE),
        (r'->select.*->symmetricDifference', OCLPatternType.SYMMETRIC_DIFFERENCE),
        (r'allInstances\(\)->symmetricDifference', OCLPatternType.SYMMETRIC_DIFFERENCE),
        (r'->symmetricDifference.*->size', OCLPatternType.SYMMETRIC_DIFFERENCE),
        (r'->symmetricDifference.*->notEmpty', OCLPatternType.SYMMETRIC_DIFFERENCE),
        (r'->union.*->symmetricDifference', OCLPatternType.SYMMETRIC_DIFFERENCE),
        (r'->collect.*->symmetricDifference', OCLPatternType.SYMMETRIC_DIFFERENCE),
        
        # ═══════════════════════════════════════════════════════════════════
        # 42. INCLUDING_EXCLUDING (14 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->including\(', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->excluding\(', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->select.*->including', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->select.*->excluding', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->asSet\(\)->including', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->asSet\(\)->excluding', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->including\(self\)', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->excluding\(self\)', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->collect.*->including', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->collect.*->excluding', OCLPatternType.INCLUDING_EXCLUDING),
        (r'allInstances\(\)->excluding', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->including.*->including', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->excluding.*->excluding', OCLPatternType.INCLUDING_EXCLUDING),
        (r'->union.*->excluding', OCLPatternType.INCLUDING_EXCLUDING),
        
        # ═══════════════════════════════════════════════════════════════════
        # 43. FLATTEN_OPERATION (covered in #21, additional patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->flatten\(\)->collect', OCLPatternType.FLATTEN_OPERATION),
        (r'->flatten\(\)->reject', OCLPatternType.FLATTEN_OPERATION),
        (r'->flatten\(\)->any', OCLPatternType.FLATTEN_OPERATION),
        (r'->flatten\(\)->one', OCLPatternType.FLATTEN_OPERATION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 44. NAVIGATION_CHAIN (14 patterns) - MUST come before STRING_COMPARISON
        # ═══════════════════════════════════════════════════════════════════
        # Navigation with comparison - specific first
        (r'self\.\w+\.\w+\.\w+\.\w+\s*[<>=]', OCLPatternType.NAVIGATION_CHAIN),
        (r'self\.\w+\.\w+\.\w+\s*[<>=]', OCLPatternType.NAVIGATION_CHAIN),
        (r'\w+\.\w+\.\w+\.\w+\s*[<>=]', OCLPatternType.NAVIGATION_CHAIN),
        # General navigation
        (r'self\.\w+\.\w+\.\w+\.\w+', OCLPatternType.NAVIGATION_CHAIN),
        (r'self\.\w+\.\w+\.\w+', OCLPatternType.NAVIGATION_CHAIN),
        (r'\w+\.\w+\.\w+\.\w+', OCLPatternType.NAVIGATION_CHAIN),
        (r'\.\w+\.\w+\.\w+', OCLPatternType.NAVIGATION_CHAIN),
        (r'self\.\w+\.\w+->', OCLPatternType.NAVIGATION_CHAIN),
        (r'\.\w+\.\w+->', OCLPatternType.NAVIGATION_CHAIN),
        (r'self\.\w+\.\w+\.\w+->', OCLPatternType.NAVIGATION_CHAIN),
        (r'->collect\(\w+\s*\|\s*\w+\.\w+\.\w+\)', OCLPatternType.NAVIGATION_CHAIN),
        (r'->forAll\([^)]*\.\w+\.\w+', OCLPatternType.NAVIGATION_CHAIN),
        (r'->select\([^)]*\.\w+\.\w+', OCLPatternType.NAVIGATION_CHAIN),
        (r'if.*\.\w+\.\w+.*then', OCLPatternType.NAVIGATION_CHAIN),
        
        # ═══════════════════════════════════════════════════════════════════
        # 45. OPTIONAL_NAVIGATION (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->isEmpty\(\)\s+or\s+\w+', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'\w+\s*=\s*null\s+or\s+\w+', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'if.*->isEmpty\(\).*then.*else', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'if.*=\s*null.*then.*else', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'->notEmpty\(\)\s+implies.*\.', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'<>\s*null\s+implies.*\.', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'isDefined\(\)\s+implies.*\.', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'->isEmpty\(\)\s+or.*->', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'if.*isDefined\(\).*then', OCLPatternType.OPTIONAL_NAVIGATION),
        (r'->select\([^)]*<>\s*null\s*\)\.\w+', OCLPatternType.OPTIONAL_NAVIGATION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 46. COLLECTION_NAVIGATION (14 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'->first\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->last\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->at\(', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->at\(\d+\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->select.*->first\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->select.*->last\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->sortedBy.*->first\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->sortedBy.*->last\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'allInstances\(\)->first\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->asSequence\(\)->at\(', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->collect.*->first\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->reject.*->first\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->flatten\(\)->first\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        (r'->union.*->last\(\)', OCLPatternType.COLLECTION_NAVIGATION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 47. SHORTHAND_NOTATION (10 patterns) - ONLY when no comparison after size()
        # ═══════════════════════════════════════════════════════════════════
        (r'\w+\.\w+->sum\(\)', OCLPatternType.SHORTHAND_NOTATION),
        # Size only if NOT followed by comparison (handled by SIZE_CONSTRAINT earlier)
        (r'\w+\.\w+->size\(\)(?!\s*[<>=])', OCLPatternType.SHORTHAND_NOTATION),
        (r'\w+\.\w+->select', OCLPatternType.SHORTHAND_NOTATION),
        (r'\w+\.\w+->forAll', OCLPatternType.SHORTHAND_NOTATION),
        (r'\w+\.\w+->exists', OCLPatternType.SHORTHAND_NOTATION),
        (r'\w+\.\w+->collect', OCLPatternType.SHORTHAND_NOTATION),
        (r'\w+\.\w+->isEmpty\(\)', OCLPatternType.SHORTHAND_NOTATION),
        (r'\w+\.\w+->notEmpty\(\)', OCLPatternType.SHORTHAND_NOTATION),
        (r'\w+\.\w+->includes', OCLPatternType.SHORTHAND_NOTATION),
        (r'\w+\.\w+->isUnique', OCLPatternType.SHORTHAND_NOTATION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 48. OCL_IS_UNDEFINED (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\.oclIsUndefined\(\)', OCLPatternType.OCL_IS_UNDEFINED),
        (r'self\.\w+\.oclIsUndefined\(\)', OCLPatternType.OCL_IS_UNDEFINED),
        (r'not.*\.oclIsUndefined\(\)', OCLPatternType.OCL_IS_UNDEFINED),
        (r'->select\([^)]*oclIsUndefined', OCLPatternType.OCL_IS_UNDEFINED),
        (r'->forAll\([^)]*oclIsUndefined', OCLPatternType.OCL_IS_UNDEFINED),
        (r'->exists\([^)]*oclIsUndefined', OCLPatternType.OCL_IS_UNDEFINED),
        (r'if.*oclIsUndefined.*then', OCLPatternType.OCL_IS_UNDEFINED),
        (r'\.oclIsUndefined\(\)\s+or', OCLPatternType.OCL_IS_UNDEFINED),
        (r'\.oclIsUndefined\(\)\s+and', OCLPatternType.OCL_IS_UNDEFINED),
        (r'->reject\([^)]*oclIsUndefined', OCLPatternType.OCL_IS_UNDEFINED),
        
        # ═══════════════════════════════════════════════════════════════════
        # 49. OCL_IS_INVALID (10 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\.oclIsInvalid\(\)', OCLPatternType.OCL_IS_INVALID),
        (r'self\.\w+\.oclIsInvalid\(\)', OCLPatternType.OCL_IS_INVALID),
        (r'not.*\.oclIsInvalid\(\)', OCLPatternType.OCL_IS_INVALID),
        (r'->select\([^)]*oclIsInvalid', OCLPatternType.OCL_IS_INVALID),
        (r'->forAll\([^)]*oclIsInvalid', OCLPatternType.OCL_IS_INVALID),
        (r'->exists\([^)]*oclIsInvalid', OCLPatternType.OCL_IS_INVALID),
        (r'if.*oclIsInvalid.*then', OCLPatternType.OCL_IS_INVALID),
        (r'\.oclIsInvalid\(\)\s+or', OCLPatternType.OCL_IS_INVALID),
        (r'\.oclIsInvalid\(\)\s+and', OCLPatternType.OCL_IS_INVALID),
        (r'->reject\([^)]*oclIsInvalid', OCLPatternType.OCL_IS_INVALID),
        
        # ═══════════════════════════════════════════════════════════════════
        # 50. OCL_AS_TYPE (12 patterns)
        # ═══════════════════════════════════════════════════════════════════
        (r'\.oclAsType\(', OCLPatternType.OCL_AS_TYPE),
        (r'\.oclAsType\([A-Z]\w+\)', OCLPatternType.OCL_AS_TYPE),
        (r'->select\([^)]*oclAsType', OCLPatternType.OCL_AS_TYPE),
        (r'->collect\([^)]*oclAsType', OCLPatternType.OCL_AS_TYPE),
        (r'->forAll\([^)]*oclAsType', OCLPatternType.OCL_AS_TYPE),
        (r'if.*oclAsType.*then', OCLPatternType.OCL_AS_TYPE),
        (r'oclIsKindOf.*oclAsType', OCLPatternType.OCL_AS_TYPE),
        (r'->any\([^)]*oclAsType', OCLPatternType.OCL_AS_TYPE),
        (r'let.*oclAsType', OCLPatternType.OCL_AS_TYPE),
        (r'self\.oclAsType', OCLPatternType.OCL_AS_TYPE),
        (r'->reject\([^)]*oclAsType', OCLPatternType.OCL_AS_TYPE),
        (r'->exists\([^)]*oclAsType', OCLPatternType.OCL_AS_TYPE),
        
        # ═══════════════════════════════════════════════════════════════════
        # Generic patterns (LOWEST priority - must be last!)
        # ═══════════════════════════════════════════════════════════════════
        
        # ═══════════════════════════════════════════════════════════════════
        # 32. ARITHMETIC_EXPRESSION (16 patterns) - After all specific patterns
        # ═══════════════════════════════════════════════════════════════════
        (r'\w+\s*\+\s*\w+\s*\+', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'\w+\s*-\s*\w+\s*-', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'\w+\s*\*\s*\w+\s*\*', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'\w+\s*/\s*\w+\s*/', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'self\.\w+\s*\+\s*self\.', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'self\.\w+\s*-\s*self\.', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'self\.\w+\s*\*\s*self\.', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'self\.\w+\s*/\s*self\.', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'\(\s*\w+\s*\+\s*\w+\s*\)\s*\*', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'\(\s*\w+\s*-\s*\w+\s*\)\s*/', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'->sum\(\)\s*\+', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'->sum\(\)\s*-', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'->size\(\)\s*\+\s*\w+', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'->size\(\)\s*\*\s*\w+', OCLPatternType.ARITHMETIC_EXPRESSION),
        (r'\w+\.\w+\s*\+\s*\w+\.\w+\s*\+', OCLPatternType.ARITHMETIC_EXPRESSION),
        
        # ═══════════════════════════════════════════════════════════════════
        # 30. STRING_COMPARISON (10 patterns) - After navigation chain!
        # ═══════════════════════════════════════════════════════════════════
        (r'\.equalsIgnoreCase\(', OCLPatternType.STRING_COMPARISON),
        (r'\w+\s*=\s*["\']', OCLPatternType.STRING_COMPARISON),
        (r'\w+\s*<>\s*["\']', OCLPatternType.STRING_COMPARISON),
        (r'self\.\w+\s*=\s*["\']', OCLPatternType.STRING_COMPARISON),
        (r'\.toUpper\(\)\s*=.*\.toUpper\(\)', OCLPatternType.STRING_COMPARISON),
        (r'\.toLower\(\)\s*=.*\.toLower\(\)', OCLPatternType.STRING_COMPARISON),
        (r'\.compareTo\(', OCLPatternType.STRING_COMPARISON),
        (r'->forAll\([^)]*\w+\s*=\s*["\']', OCLPatternType.STRING_COMPARISON),
        (r'->exists\([^)]*\w+\s*=\s*["\']', OCLPatternType.STRING_COMPARISON),
        (r'->select\([^)]*\w+\s*=\s*["\']', OCLPatternType.STRING_COMPARISON),
        
        # Very generic - absolutely last
        (r'implies', OCLPatternType.IMPLIES),
        (r'\s+and\s+', OCLPatternType.BOOLEAN_OPERATIONS),
        (r'>=|<=|>|<', OCLPatternType.NUMERIC_COMPARISON),
    ]
    
    def __init__(self):
        """Initialize the comprehensive pattern detector with compiled regex patterns."""
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), pattern_type)
            for pattern, pattern_type in self.COMPREHENSIVE_PATTERNS
        ]
    
    def detect_pattern(self, constraint_text: str) -> Optional[OCLPatternType]:
        """
        Detect OCL pattern using comprehensive regex matching.
        
        Args:
            constraint_text: The OCL constraint text to analyze
            
        Returns:
            OCLPatternType if pattern detected, None otherwise
        """
        if not constraint_text:
            return None
        
        # Try each pattern in priority order (specific → general)
        for compiled_regex, pattern_type in self.compiled_patterns:
            if compiled_regex.search(constraint_text):
                return pattern_type
        
        return None
    
    def get_pattern_count(self) -> int:
        """Return the total number of regex patterns available."""
        return len(self.COMPREHENSIVE_PATTERNS)
    
    def get_patterns_by_type(self, pattern_type: OCLPatternType) -> list:
        """Get all regex patterns for a specific OCL pattern type."""
        return [
            pattern for pattern, ptype in self.COMPREHENSIVE_PATTERNS
            if ptype == pattern_type
        ]
