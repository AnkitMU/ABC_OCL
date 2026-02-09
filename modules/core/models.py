"""
Core Data Models for Revolutionary OCL Generation Framework

This module defines the foundational data structures used throughout the framework.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
from datetime import datetime

# Configure logging for models (suppress debug by default)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ===== Metamodel Data Structures =====

@dataclass
class Attribute:
    """
    Represents a class attribute in the metamodel
    
    Attributes:
        name: Attribute name
        type: Data type (String, Integer, Boolean, Real, Date, etc.)
        lower: Multiplicity lower bound (0 or 1)
        upper: Multiplicity upper bound (1 or -1 for *)
        is_derived: Whether this is a derived attribute
        initial_value: Default/initial value if any
    """
    name: str
    type: str
    lower: int = 1
    upper: int = 1
    is_derived: bool = False
    initial_value: Optional[Any] = None
    
    @property
    def is_optional(self) -> bool:
        """Returns True if attribute is optional (lower bound = 0)"""
        return self.lower == 0
    
    @property
    def is_collection(self) -> bool:
        """Returns True if attribute has collection multiplicity"""
        return self.upper == -1 or self.upper > 1
    
    @property
    def multiplicity(self) -> str:
        """Returns multiplicity as string (e.g., '0..1', '1..*')"""
        upper_str = '*' if self.upper == -1 else str(self.upper)
        return f"{self.lower}..{upper_str}"
    
    def __str__(self) -> str:
        return f"{self.name}: {self.type} [{self.multiplicity}]"


@dataclass
class Association:
    """
    Represents an association/reference between classes
    
    Attributes:
        name: Association name
        ref_name: Role name (navigation property name)
        source_class: Source class name
        target_class: Target class name
        lower: Multiplicity lower bound
        upper: Multiplicity upper bound (-1 for *)
        is_composition: Whether this is a composition relationship
        is_bidirectional: Whether association is bidirectional
        opposite_ref: Name of opposite reference (if bidirectional)
    """
    name: str
    ref_name: str
    source_class: str
    target_class: str
    lower: int = 0
    upper: int = -1
    is_composition: bool = False
    is_bidirectional: bool = False
    opposite_ref: Optional[str] = None
    
    @property
    def is_optional(self) -> bool:
        """Returns True if association is optional (lower bound = 0)"""
        return self.lower == 0
    
    @property
    def is_collection(self) -> bool:
        """Returns True if association has collection multiplicity"""
        return self.upper == -1 or self.upper > 1
    
    @property
    def multiplicity(self) -> str:
        """Returns multiplicity as string"""
        upper_str = '*' if self.upper == -1 else str(self.upper)
        return f"{self.lower}..{upper_str}"
    
    def __str__(self) -> str:
        comp = " (composition)" if self.is_composition else ""
        return f"{self.source_class}.{self.ref_name} -> {self.target_class} [{self.multiplicity}]{comp}"


@dataclass
class Class:
    """
    Represents a class in the metamodel
    
    Attributes:
        name: Class name
        attributes: List of attributes
        associations: List of associations/references
        is_abstract: Whether class is abstract
        parent_class: Parent class name (if inheritance)
        operations: List of operation signatures
    """
    name: str
    attributes: List[Attribute] = field(default_factory=list)
    associations: List[Association] = field(default_factory=list)
    is_abstract: bool = False
    parent_class: Optional[str] = None
    operations: List[str] = field(default_factory=list)
    
    def get_attribute(self, name: str) -> Optional[Attribute]:
        """Get attribute by name"""
        return next((a for a in self.attributes if a.name == name), None)
    
    def get_association(self, ref_name: str) -> Optional[Association]:
        """Get association by reference name"""
        return next((a for a in self.associations if a.ref_name == ref_name), None)
    
    def __str__(self) -> str:
        abstract = "abstract " if self.is_abstract else ""
        parent = f" extends {self.parent_class}" if self.parent_class else ""
        return f"{abstract}class {self.name}{parent}"


@dataclass
class Metamodel:
    """
    Complete metamodel extracted from XMI or other sources
    
    Provides convenient access to classes, attributes, and associations
    """
    classes: Dict[str, Class] = field(default_factory=dict)
    _association_index: Dict[str, List[Association]] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Build indexes after initialization"""
        logger.info(f"Initializing metamodel with {len(self.classes)} classes")
        self._build_association_index()
        logger.info(f"Metamodel initialization complete. Association index built for {len(self._association_index)} classes")
    
    def _build_association_index(self):
        """Build index for fast association lookup"""
        logger.debug("Building association index...")
        self._association_index = {}
        assoc_count = 0
        for cls in self.classes.values():
            for assoc in cls.associations:
                if assoc.source_class not in self._association_index:
                    self._association_index[assoc.source_class] = []
                self._association_index[assoc.source_class].append(assoc)
                assoc_count += 1
        logger.debug(f"Association index built: {assoc_count} associations indexed")
    
    def add_class(self, cls: Class):
        """Add a class to the metamodel"""
        logger.debug(f"Adding class '{cls.name}' with {len(cls.attributes)} attributes and {len(cls.associations)} associations")
        self.classes[cls.name] = cls
        # Update association index
        for assoc in cls.associations:
            if assoc.source_class not in self._association_index:
                self._association_index[assoc.source_class] = []
            self._association_index[assoc.source_class].append(assoc)
        logger.debug(f"Class '{cls.name}' added successfully")
    
    def get_class(self, name: str) -> Optional[Class]:
        """Get class by name"""
        return self.classes.get(name)
    
    def get_class_names(self) -> List[str]:
        """Get list of all class names"""
        return sorted(self.classes.keys())
    
    def get_attributes_for(self, class_name: str) -> List[Attribute]:
        """Get all attributes for a class"""
        cls = self.get_class(class_name)
        return cls.attributes if cls else []
    
    def get_associations_for(self, class_name: str) -> List[Association]:
        """Get all associations originating from a class"""
        return self._association_index.get(class_name, [])
    
    def get_collection_associations(self, class_name: str) -> List[Association]:
        """Get associations with collection multiplicity"""
        return [a for a in self.get_associations_for(class_name) if a.is_collection]
    
    def get_single_associations(self, class_name: str) -> List[Association]:
        """Get associations with single multiplicity"""
        return [a for a in self.get_associations_for(class_name) if not a.is_collection]
    
    def get_all_associations(self) -> List[Association]:
        """Get all associations in the metamodel"""
        result = []
        for assocs in self._association_index.values():
            result.extend(assocs)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metamodel to dictionary representation"""
        return {
            'classes': [
                {
                    'name': cls.name,
                    'attributes': [{'name': a.name, 'type': a.type, 'multiplicity': a.multiplicity} 
                                  for a in cls.attributes],
                    'associations': [{'ref_name': a.ref_name, 'target': a.target_class, 
                                     'multiplicity': a.multiplicity} 
                                    for a in cls.associations]
                }
                for cls in self.classes.values()
            ]
        }


# ===== Pattern Data Structures =====

class ParameterType(Enum):
    """Types of parameters in constraint patterns"""
    SELECT = "select"              # Dropdown selection
    NUMBER = "number"              # Numeric input
    TEXT = "text"                  # Text input
    BOOLEAN = "boolean"            # Checkbox
    MULTI_SELECT = "multi_select"  # Multiple selections
    EXPRESSION = "expression"      # OCL expression input


class PatternCategory(Enum):
    """Categories of constraint patterns"""
    BASIC = "basic"                      # Patterns 1-9
    ADVANCED = "advanced"                # Patterns 10-19
    COLLECTION = "collection"            # Patterns 20-27
    STRING = "string"                    # Patterns 28-31
    ARITHMETIC = "arithmetic"            # Patterns 32-36
    TUPLE_LET = "tuple_let"             # Patterns 37-39
    SET_OPERATIONS = "set_operations"    # Patterns 40-43
    NAVIGATION = "navigation"            # Patterns 44-47
    OCL_LIBRARY = "ocl_library"         # Patterns 48-50


@dataclass
class Parameter:
    """
    Parameter definition for a constraint pattern
    
    Attributes:
        name: Parameter identifier
        label: Human-readable label
        type: Parameter type (select, number, text, etc.)
        options: Static list or dynamic option source
        default: Default value
        required: Whether parameter is required
        help_text: Help text for user
        depends_on: Other parameter this depends on
        validation: Validation function
    """
    name: str
    label: str
    type: ParameterType
    options: Optional[Union[List[str], str]] = None  # List or "attributes"/"associations"/etc.
    default: Any = None
    required: bool = True
    help_text: Optional[str] = None
    depends_on: Optional[str] = None
    validation: Optional[Callable] = None
    
    def get_options_for_context(self, metamodel: Metamodel, context: str, 
                                 dependency_values: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get dynamic options based on context class and dependencies
        
        Args:
            metamodel: The metamodel to query
            context: The context class name
            dependency_values: Values of parameters this depends on
            
        Returns:
            List of available options
        """
        if isinstance(self.options, list):
            # Static options
            return self.options
        
        if self.options == "attributes":
            # All attributes of context class
            return [a.name for a in metamodel.get_attributes_for(context)]
        
        elif self.options == "associations":
            # All associations
            return [a.ref_name for a in metamodel.get_associations_for(context)]
        
        elif self.options == "collection_associations":
            # Only collection associations
            return [a.ref_name for a in metamodel.get_collection_associations(context)]
        
        elif self.options == "single_associations":
            # Only single associations
            return [a.ref_name for a in metamodel.get_single_associations(context)]
        
        elif self.options == "classes":
            # All classes
            return metamodel.get_class_names()
        
        elif self.options == "numeric_attributes":
            # Only numeric attributes (support both UML and Ecore types)
            return [a.name for a in metamodel.get_attributes_for(context) 
                   if a.type in ['Integer', 'Real', 'Double', 'Float', 'EInt', 'EDouble', 'EFloat', 'ELong', 'EShort']]
        
        elif self.options == "string_attributes":
            # Only string attributes (support both UML and Ecore types)
            return [a.name for a in metamodel.get_attributes_for(context) 
                   if a.type in ['String', 'EString']]
        
        elif self.options == "boolean_attributes":
            # Only boolean attributes (support both UML and Ecore types)
            return [a.name for a in metamodel.get_attributes_for(context) 
                   if a.type in ['Boolean', 'EBoolean']]
        
        elif self.options == "target_attributes":
            # Attributes of target class (requires dependency)
            if dependency_values and 'collection' in dependency_values:
                assoc_name = dependency_values['collection']
                assoc = next((a for a in metamodel.get_associations_for(context) 
                            if a.ref_name == assoc_name), None)
                if assoc:
                    return [a.name for a in metamodel.get_attributes_for(assoc.target_class)]
            return []
        
        return []
    
    def validate_value(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate parameter value
        
        Returns:
            (is_valid, error_message)
        """
        if self.required and (value is None or value == ""):
            return False, f"{self.label} is required"
        
        if value is not None and self.validation:
            try:
                if not self.validation(value):
                    return False, f"Invalid value for {self.label}"
            except Exception as e:
                return False, str(e)
        
        if self.type == ParameterType.NUMBER and value is not None:
            try:
                float(value)
            except (ValueError, TypeError):
                return False, f"{self.label} must be a number"
        
        return True, None


@dataclass
class Pattern:
    """
    Constraint pattern definition
    
    Attributes:
        id: Unique pattern identifier
        name: Human-readable name
        category: Pattern category
        description: Detailed description
        template: OCL template with {param} placeholders
        parameters: List of parameters
        examples: Example instantiations
        requires_context: Whether pattern needs a context class
        complexity: Relative complexity (1-5)
        tags: Tags for searching/filtering
    """
    id: str
    name: str
    category: PatternCategory
    description: str
    template: str
    parameters: List[Parameter] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    requires_context: bool = True
    complexity: int = 1
    tags: List[str] = field(default_factory=list)
    
    def generate_ocl(self, context: str, params: Dict[str, Any]) -> str:
        """
        Generate OCL from template and parameters
        
        Args:
            context: Context class name
            params: Parameter values
            
        Returns:
            Generated OCL constraint
            
        Raises:
            ValueError: If required parameters are missing
        """
        try:
            # Fill template with parameters
            ocl_body = self.template.format(**params)
            
            # Wrap with context if needed
            if self.requires_context and context:
                return f"context {context}\ninv: {ocl_body}"
            else:
                return ocl_body
        
        except KeyError as e:
            raise ValueError(f"Missing parameter: {e}")
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate provided parameters
        
        Returns:
            (is_valid, error_message)
        """
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.label}"
            
            if param.name in params:
                is_valid, error = param.validate_value(params[param.name])
                if not is_valid:
                    return False, error
        
        return True, None
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """Get parameter by name"""
        return next((p for p in self.parameters if p.name == name), None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'template': self.template,
            'parameters': [
                {
                    'name': p.name,
                    'label': p.label,
                    'type': p.type.value,
                    'options': p.options,
                    'default': p.default,
                    'required': p.required
                }
                for p in self.parameters
            ],
            'examples': self.examples
        }


# ===== OCL Generation Results =====

@dataclass
class OCLConstraint:
    """
    Generated OCL constraint with metadata
    
    Attributes:
        ocl: The OCL constraint text
        pattern_id: ID of pattern used
        pattern_name: Name of pattern
        context: Context class
        parameters: Parameter values used
        confidence: Confidence score (0-1)
        timestamp: Generation timestamp
        metadata: Additional metadata
    """
    ocl: str
    pattern_id: str
    pattern_name: str
    context: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def __str__(self) -> str:
        return self.ocl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'ocl': self.ocl,
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'context': self.context,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCLConstraint':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ValidationResult:
    """
    Result of OCL validation
    
    Attributes:
        is_valid: Whether constraint is valid
        errors: List of error messages
        warnings: List of warning messages
        suggestions: List of improvement suggestions
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def add_error(self, message: str):
        """Add an error"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning"""
        self.warnings.append(message)
    
    def add_suggestion(self, message: str):
        """Add a suggestion"""
        self.suggestions.append(message)


@dataclass
class SynthesisResult:
    """
    Result from synthesis strategies
    
    Attributes:
        constraint: Generated constraint
        strategy: Strategy used (pattern/llm/synthesis)
        confidence: Confidence score
        alternatives: Alternative constraints
        explanation: Human-readable explanation
    """
    constraint: OCLConstraint
    strategy: str
    confidence: float
    alternatives: List[OCLConstraint] = field(default_factory=list)
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'constraint': self.constraint.to_dict(),
            'strategy': self.strategy,
            'confidence': self.confidence,
            'alternatives': [alt.to_dict() for alt in self.alternatives],
            'explanation': self.explanation
        }
