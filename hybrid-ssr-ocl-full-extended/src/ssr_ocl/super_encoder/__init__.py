# Super Encoder Module
# Generic, metadata-driven encoders for all 50 OCL patterns
# Works with ANY XMI domain model - NO HARDCODING

from .comprehensive_pattern_detector import ComprehensivePatternDetector
from .generic_global_consistency_checker import GenericGlobalConsistencyChecker

__all__ = [
    'ComprehensivePatternDetector',
    'GenericGlobalConsistencyChecker',
]
