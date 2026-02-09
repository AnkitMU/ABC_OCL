"""
Benchmark Suite Configuration
Defines suite-level specs for automatic benchmark generation.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
from pathlib import Path


@dataclass
class VerificationSpec:
    """Verification configuration for the suite."""
    enable: bool = True
    objective: str = "both"  # sat | unsat | both
    per_constraint_timeout_ms: int = 8000
    batch_timeout_ms: int = 30000
    check_global_consistency: bool = True
    implication_use_z3: bool = False


@dataclass
class ProfileSpec:
    """Profile configuration for a single benchmark profile."""
    name: str
    seed: int = 42
    constraints: int = 50
    complexity_profile: str = "standard"  # basic | standard | advanced
    sat_ratio: float = 0.7
    unsat_ratio: float = 0.3
    include_unknown: bool = False
    
    # Override default profile settings
    families_pct: Optional[Dict[str, int]] = None
    per_class_min: Optional[int] = None
    per_class_max: Optional[int] = None
    similarity_threshold: Optional[float] = None
    novelty_boost: Optional[bool] = None
    
    # Difficulty mix
    difficulty_mix: Optional[Dict[str, float]] = None  # easy/medium/hard


@dataclass
class ModelSpec:
    """Model specification with associated profiles."""
    xmi: str
    name: Optional[str] = None  # Auto-inferred from filename if not provided
    profiles: List[ProfileSpec] = field(default_factory=list)
    
    def __post_init__(self):
        if self.name is None:
            self.name = Path(self.xmi).stem


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite specification."""
    suite_name: str
    version: str = "1.0"
    models: List[ModelSpec] = field(default_factory=list)
    verification: VerificationSpec = field(default_factory=VerificationSpec)
    output_root: str = "benchmarks/"
    description: Optional[str] = None
    
    # Reproducibility
    framework_version: str = "2.0"
    git_commit: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BenchmarkSuite':
        """Load suite from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkSuite':
        """Parse suite from dictionary."""
        # Parse verification
        verif_data = data.get('verification', {})
        verification = VerificationSpec(**verif_data)
        
        # Parse models
        models = []
        for model_data in data.get('models', []):
            profiles = []
            for prof_data in model_data.get('profiles', []):
                profile = ProfileSpec(**prof_data)
                profiles.append(profile)
            
            model = ModelSpec(
                xmi=model_data['xmi'],
                name=model_data.get('name'),
                profiles=profiles
            )
            models.append(model)
        
        suite = cls(
            suite_name=data['suite_name'],
            version=data.get('version', '1.0'),
            models=models,
            verification=verification,
            output_root=data.get('output_root', 'benchmarks/'),
            description=data.get('description'),
            git_commit=data.get('git_commit')
        )
        
        return suite
    
    def to_yaml(self, yaml_path: str):
        """Save suite to YAML file."""
        data = {
            'suite_name': self.suite_name,
            'version': self.version,
            'description': self.description,
            'models': [
                {
                    'xmi': m.xmi,
                    'name': m.name,
                    'profiles': [
                        {
                            'name': p.name,
                            'seed': p.seed,
                            'constraints': p.constraints,
                            'complexity_profile': p.complexity_profile,
                            'sat_ratio': p.sat_ratio,
                            'unsat_ratio': p.unsat_ratio,
                            'include_unknown': p.include_unknown,
                            'families_pct': p.families_pct,
                            'difficulty_mix': p.difficulty_mix
                        }
                        for p in m.profiles
                    ]
                }
                for m in self.models
            ],
            'verification': {
                'enable': self.verification.enable,
                'objective': self.verification.objective,
                'per_constraint_timeout_ms': self.verification.per_constraint_timeout_ms,
                'batch_timeout_ms': self.verification.batch_timeout_ms
            },
            'output_root': self.output_root,
            'framework_version': self.framework_version,
            'git_commit': self.git_commit
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Difficulty profile presets
DIFFICULTY_PROFILES = {
    "easy": {
        "max_depth": 2,
        "max_quantifier_depth": 0,
        "max_hops": 1,
        "allowed_operators": ["=", ">=", "<=", "size", "isEmpty", "notEmpty"],
        "difficulty_mix": {"easy": 1.0, "medium": 0.0, "hard": 0.0}
    },
    "standard": {
        "max_depth": 3,
        "max_quantifier_depth": 1,
        "max_hops": 2,
        "allowed_operators": ["forAll", "exists", "select", "collect", "size", "implies"],
        "difficulty_mix": {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    },
    "advanced": {
        "max_depth": 5,
        "max_quantifier_depth": 2,
        "max_hops": 4,
        "allowed_operators": None,  # All operators
        "difficulty_mix": {"easy": 0.2, "medium": 0.4, "hard": 0.4}
    }
}


def get_difficulty_profile(name: str) -> Dict:
    """Get difficulty profile by name."""
    return DIFFICULTY_PROFILES.get(name, DIFFICULTY_PROFILES["standard"])
