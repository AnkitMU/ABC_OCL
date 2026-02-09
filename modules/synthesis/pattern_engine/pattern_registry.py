"""
Pattern Registry - Load and manage all constraint patterns
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from modules.core.models import Pattern, Parameter, ParameterType, PatternCategory

# Configure logging (suppress debug by default)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class PatternRegistry:
    """
    Registry of all available OCL constraint patterns
    
    Loads patterns from JSON and provides search, filter, and retrieval
    """
    
    def __init__(self, json_file: Optional[str] = None):
        """
        Initialize pattern registry
        
        Args:
            json_file: Path to pattern JSON file (default: templates/patterns_revised.json)
        """
        logger.info("Initializing PatternRegistry")
        if json_file is None:
            # Default to REVISED patterns (validated, OCL-compliant)
            base_path = Path(__file__).parent.parent.parent.parent
            revised_file = base_path / 'templates' / 'patterns_revised.json'
            unified_file = base_path / 'templates' / 'patterns_unified.json'
            
            # Use revised if available, fallback to unified, then legacy
            if revised_file.exists():
                json_file = str(revised_file)
                logger.debug(f"Using REVISED pattern file: {json_file}")
            elif unified_file.exists():
                json_file = str(unified_file)
                logger.debug(f"Using UNIFIED pattern file: {json_file}")
            else:
                # Legacy fallback
                json_file = str(base_path / 'templates' / 'pattern_registry.json')
                logger.debug(f"Using LEGACY pattern file: {json_file}")
        else:
            logger.debug(f"Using custom pattern file: {json_file}")
        
        self.json_file = json_file
        self.patterns: Dict[str, Pattern] = {}
        self._load_patterns()
        logger.info(f"PatternRegistry initialized with {len(self.patterns)} patterns")
    
    def _load_patterns(self):
        """Load patterns from JSON file"""
        logger.debug(f"Loading patterns from {self.json_file}")
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            pattern_count_before = len(self.patterns)
            for pattern_data in data.get('patterns', []):
                pattern = self._parse_pattern(pattern_data)
                self.patterns[pattern.id] = pattern
                logger.debug(f"Loaded pattern: {pattern.id} - {pattern.name}")
            
            pattern_count = len(self.patterns) - pattern_count_before
            # Show which file was loaded
            file_name = Path(self.json_file).name
            if 'patterns_revised' in file_name:
                logger.info(f"Loaded {pattern_count} patterns from REVISED pattern library (validated)")
                print(f"Loaded {pattern_count} patterns from REVISED pattern library (validated)")
            elif 'patterns_unified' in file_name:
                logger.info(f"Loaded {pattern_count} patterns from unified pattern library")
                print(f"Loaded {pattern_count} patterns from unified pattern library")
            else:
                logger.info(f"Loaded {pattern_count} patterns from {file_name}")
                print(f"Loaded {pattern_count} patterns from {file_name}")
        
        except FileNotFoundError:
            logger.error(f"Pattern file not found: {self.json_file}")
            print(f"Pattern file not found: {self.json_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in pattern file: {e}")
            print(f"Invalid JSON in pattern file: {e}")
            raise
    
    def _parse_pattern(self, data: Dict) -> Pattern:
        """Parse pattern from JSON data"""
        
        # Parse category
        category_str = data.get('category', 'basic')
        category = PatternCategory(category_str)
        
        # Parse parameters
        parameters = []
        for param_data in data.get('parameters', []):
            param_type_str = param_data.get('type', 'text')
            param_type = ParameterType(param_type_str)
            
            param = Parameter(
                name=param_data['name'],
                label=param_data['label'],
                type=param_type,
                options=param_data.get('options'),
                default=param_data.get('default'),
                required=param_data.get('required', True),
                help_text=param_data.get('help_text'),
                depends_on=param_data.get('depends_on')
            )
            parameters.append(param)
        
        # Create pattern
        pattern = Pattern(
            id=data['id'],
            name=data['name'],
            category=category,
            description=data['description'],
            template=data['template'],
            parameters=parameters,
            examples=data.get('examples', []),
            requires_context=data.get('requires_context', True),
            complexity=data.get('complexity', 1),
            tags=data.get('tags', [])
        )
        
        return pattern
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Get pattern by ID
        
        Args:
            pattern_id: Pattern identifier
            
        Returns:
            Pattern object or None if not found
        """
        logger.debug(f"Retrieving pattern: {pattern_id}")
        pattern = self.patterns.get(pattern_id)
        if pattern:
            logger.debug(f"Pattern found: {pattern.name}")
        else:
            logger.warning(f"Pattern not found: {pattern_id}")
        return pattern
    
    def get_all_patterns(self) -> List[Pattern]:
        """Get all patterns"""
        return list(self.patterns.values())
    
    def get_patterns_by_category(self, category: PatternCategory) -> List[Pattern]:
        """
        Get all patterns in a category
        
        Args:
            category: Pattern category
            
        Returns:
            List of patterns in category
        """
        return [p for p in self.patterns.values() if p.category == category]
    
    def get_all_categories(self) -> List[PatternCategory]:
        """Get list of all categories with patterns"""
        categories = set(p.category for p in self.patterns.values())
        return sorted(categories, key=lambda c: c.value)
    
    def search_patterns(self, query: str) -> List[Pattern]:
        """
        Search patterns by query string
        
        Searches in: name, description, tags, examples
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching patterns
        """
        query_lower = query.lower()
        results = []
        
        for pattern in self.patterns.values():
            # Search in name
            if query_lower in pattern.name.lower():
                results.append(pattern)
                continue
            
            # Search in description
            if query_lower in pattern.description.lower():
                results.append(pattern)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in pattern.tags):
                results.append(pattern)
                continue
            
            # Search in ID
            if query_lower in pattern.id.lower():
                results.append(pattern)
                continue
        
        return results
    
    def get_patterns_by_complexity(self, max_complexity: int) -> List[Pattern]:
        """
        Get patterns with complexity <= max_complexity
        
        Args:
            max_complexity: Maximum complexity (1-5)
            
        Returns:
            List of patterns
        """
        return [p for p in self.patterns.values() if p.complexity <= max_complexity]
    
    def get_patterns_by_tags(self, tags: List[str]) -> List[Pattern]:
        """
        Get patterns matching any of the given tags
        
        Args:
            tags: List of tags to match
            
        Returns:
            List of matching patterns
        """
        tags_lower = [t.lower() for t in tags]
        results = []
        
        for pattern in self.patterns.values():
            if any(tag.lower() in tags_lower for tag in pattern.tags):
                results.append(pattern)
        
        return results
    
    def get_pattern_count(self) -> int:
        """Get total number of patterns"""
        return len(self.patterns)
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get pattern count per category"""
        counts = {}
        for pattern in self.patterns.values():
            category_name = pattern.category.value
            counts[category_name] = counts.get(category_name, 0) + 1
        return counts
    
    def print_summary(self):
        """Print registry summary"""
        print("\n" + "="*60)
        print("PATTERN REGISTRY SUMMARY")
        print("="*60)
        print(f"Total Patterns: {self.get_pattern_count()}")
        print("\nPatterns by Category:")
        
        for category in self.get_all_categories():
            patterns = self.get_patterns_by_category(category)
            print(f"  {category.value.capitalize()}: {len(patterns)} patterns")
        
        print("\nComplexity Distribution:")
        for level in range(1, 6):
            count = len(self.get_patterns_by_complexity(level))
            if count > 0:
                print(f"  Level {level}: {count} patterns")
        
        print("="*60 + "\n")
    
    def list_patterns(self, category: Optional[PatternCategory] = None):
        """
        List patterns with details
        
        Args:
            category: Optional category filter
        """
        patterns = (self.get_patterns_by_category(category) if category 
                   else self.get_all_patterns())
        
        if category:
            print(f"\n{category.value.upper()} PATTERNS ({len(patterns)}):")
        else:
            print(f"\nALL PATTERNS ({len(patterns)}):")
        
        print("-" * 60)
        
        for pattern in sorted(patterns, key=lambda p: p.id):
            print(f"\n{pattern.id}")
            print(f"  Name: {pattern.name}")
            print(f"  Description: {pattern.description}")
            print(f"  Parameters: {len(pattern.parameters)}")
            print(f"  Complexity: {pattern.complexity}/5")
            if pattern.tags:
                print(f"  Tags: {', '.join(pattern.tags)}")


# Singleton instance
_registry_instance: Optional[PatternRegistry] = None


def get_registry(json_file: Optional[str] = None) -> PatternRegistry:
    """
    Get singleton pattern registry instance
    
    Args:
        json_file: Path to pattern JSON file (only used on first call)
        
    Returns:
        Pattern registry instance
    """
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance = PatternRegistry(json_file)
    
    return _registry_instance


# Main function for testing
def main():
    """Test pattern registry"""
    import sys
    
    try:
        registry = PatternRegistry()
        registry.print_summary()
        
        # Test search
        if len(sys.argv) > 1:
            query = sys.argv[1]
            print(f"\nSearching for '{query}':")
            results = registry.search_patterns(query)
            print(f"Found {len(results)} patterns:")
            for pattern in results:
                print(f"  - {pattern.id}: {pattern.name}")
        else:
            # List basic patterns
            registry.list_patterns(PatternCategory.BASIC)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
