#!/usr/bin/env python3
"""
Date Type Adapter for SMT Encoding
===================================

Converts date fields (EString in XMI) to Int for proper arithmetic comparison in Z3.

Handles:
- startDate, endDate
- dateFrom, dateTo
- expiry
- Any field matching date patterns

Strategies:
1. Symbolic ordering: date1, date2, date3 with axioms
2. Epoch days: parse ISO dates to Int (days since epoch)
3. Bounded symbolic: fixed set of dates with total order
"""

import re
from typing import Dict, Optional, Tuple
from datetime import datetime


class DateAdapter:
    """Adapter to convert date fields from EString to Int for Z3 encoding"""
    
    # Known date field names in common domains
    DATE_FIELDS = {
        'startDate', 'endDate', 'dateFrom', 'dateTo', 
        'expiry', 'expiryDate', 'birthDate', 'hireDate',
        'releaseDate', 'dueDate', 'timestamp'
    }
    
    def __init__(self, strategy: str = 'symbolic'):
        """
        Initialize date adapter.
        
        Args:
            strategy: 'symbolic' (default), 'epoch', or 'bounded'
        """
        self.strategy = strategy
        self.date_registry = {}  # Maps date_field -> symbolic int index
        self.date_counter = 0
    
    def is_date_field(self, field_name: str) -> bool:
        """Check if field name represents a date"""
        field_lower = field_name.lower()
        return (field_name in self.DATE_FIELDS or 
                'date' in field_lower or 
                'time' in field_lower or
                'expir' in field_lower)
    
    def extract_date_comparison(self, constraint_text: str) -> Optional[Tuple[str, str, str]]:
        """
        Extract date comparison from constraint text.
        
        Returns: (left_date, operator, right_date) or None
        Example: ('startDate', '>', 'endDate')
        """
        # Pattern: self.dateField op self.dateField
        pattern = r'self\.(\w+)\s*([<>=]+)\s*self\.(\w+)'
        match = re.search(pattern, constraint_text)
        if match:
            left, op, right = match.groups()
            if self.is_date_field(left) and self.is_date_field(right):
                return (left, op, right)
        
        # Pattern: self.obj.dateField op self.dateField
        pattern2 = r'self\.(\w+)\.(\w+)\s*([<>=]+)\s*self\.(\w+)'
        match = re.search(pattern2, constraint_text)
        if match:
            obj, date_field, op, other_date = match.groups()
            if self.is_date_field(date_field) or self.is_date_field(other_date):
                return (f"{obj}.{date_field}", op, other_date)
        
        return None
    
    def get_date_variable(self, date_field: str) -> int:
        """
        Get or create symbolic Int index for date field.
        
        For symbolic strategy: assigns unique indices (0, 1, 2, ...)
        For epoch strategy: would parse date string to days since epoch
        """
        if date_field not in self.date_registry:
            self.date_registry[date_field] = self.date_counter
            self.date_counter += 1
        return self.date_registry[date_field]
    
    def parse_iso_date_to_epoch_days(self, date_str: str) -> Optional[int]:
        """
        Parse ISO date string to days since epoch.
        
        Example: '2024-01-15' -> 19737
        """
        try:
            dt = datetime.fromisoformat(date_str)
            epoch = datetime(1970, 1, 1)
            return (dt - epoch).days
        except:
            return None
    
    def create_date_ordering_axioms(self, solver, date_vars: Dict):
        """
        Add axioms to enforce total ordering over dates.
        
        For bounded strategy with known dates: date1 < date2 < date3
        """
        if self.strategy == 'bounded':
            # Example: enforce ordering if we know the dates
            # This is a placeholder - would be populated with actual date order
            pass
    
    def adapt_constraint(self, constraint_text: str) -> Tuple[str, Dict]:
        """
        Adapt constraint with date fields to use Int variables.
        
        Returns:
            (adapted_text, metadata) where metadata contains date mappings
        """
        date_comp = self.extract_date_comparison(constraint_text)
        if not date_comp:
            return constraint_text, {}
        
        left_date, op, right_date = date_comp
        
        metadata = {
            'has_dates': True,
            'left_date': left_date,
            'right_date': right_date,
            'operator': op,
            'strategy': self.strategy,
            'left_index': self.get_date_variable(left_date),
            'right_index': self.get_date_variable(right_date)
        }
        
        return constraint_text, metadata


# Convenience functions
def adapt_date_constraint(constraint_text: str, strategy: str = 'symbolic') -> Tuple[str, Dict]:
    """
    Quick adapter for date constraints.
    
    Example:
        text, metadata = adapt_date_constraint("self.endDate > self.startDate")
        # metadata = {'has_dates': True, 'left_date': 'endDate', ...}
    """
    adapter = DateAdapter(strategy=strategy)
    return adapter.adapt_constraint(constraint_text)


if __name__ == "__main__":
    # Test the adapter
    print("Date Adapter Test Suite")
    print("=" * 80)
    
    adapter = DateAdapter(strategy='symbolic')
    
    test_cases = [
        "self.endDate > self.startDate",
        "self.dateTo > self.dateFrom",
        "self.license.expiry > self.startDate",
        "self.credits >= 1 and self.credits <= 10",  # Not a date
    ]
    
    for test in test_cases:
        print(f"\nConstraint: {test}")
        text, metadata = adapter.adapt_constraint(test)
        if metadata.get('has_dates'):
            print(f"   Date comparison detected")
            print(f"     Left:  {metadata['left_date']} (index {metadata['left_index']})")
            print(f"     Op:    {metadata['operator']}")
            print(f"     Right: {metadata['right_date']} (index {metadata['right_index']})")
        else:
            print(f"    No date comparison")
