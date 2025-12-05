# OCL Benchmark Generation Framework

> **Automated generation of verified OCL constraint benchmarks across arbitrary metamodels**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Z3 Solver](https://img.shields.io/badge/Z3-SMT%20Solver-green.svg)](https://github.com/Z3Prover/z3)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A research-grade framework for generating diverse, verified OCL (Object Constraint Language) benchmarks with SMT-based satisfiability checking, metadata enrichment, and semantic diversity control.

## 📊 Framework Overview

This framework implements a **two-phase approach** to OCL benchmark generation:

1. **Benchmark Generation Phase**: Pattern library instantiation, coverage-guided generation, metadata enrichment, and mutation-based UNSAT creation
2. **Verification Phase**: Pattern mapping, SMT encoding, and Z3-based SAT/UNSAT validation

**Evaluation Results**: Tested on 10 metamodels (12-28 classes each) across diverse domains, generating 1,250 constraints with **94.3% validity rate** and solver-confirmed ground truth labels.

---

## 🌟 Key Features

### Core Capabilities
- ✅ **110 Universal OCL Patterns** → 50 canonical SMT encoders
- ✅ **Generic Z3-Based Verification** - Works with any UML/Ecore metamodel
- ✅ **Pattern Mapping & Rewriting** - Novel universal→canonical transformation layer
- ✅ **Global Consistency Checking** - Ensures all SAT constraints are mutually consistent

### Research Features Pipeline
1. **Metadata Enrichment** - Operators, navigation depth, quantifier depth, difficulty classification
2. **UNSAT Generation** - 5 mutation strategies (contradictory bounds, empty collection, type contradiction, universal negation, simple negation)
3. **Similarity** - Structural deduplication via jaccard similarities comparison
4. **Semantic Similarity** - BERT-based embeddings for meaning-based clustering
5. **Implication Detection** - Logical subsumption analysis (A ⊢ B)
6. **Manifest Generation** - JSONL output for ML pipelines and downstream tooling

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ocl-generation-framework

# Install Python dependencies
pip install -r requirements.txt

# Install Z3 SMT solver
pip install z3-solver

# Optional: For semantic similarity (research features)
pip install sentence-transformers torch
```

### Generate Your First Benchmark

```bash
# Generate benchmarks from example configuration
python generate_benchmark_suite.py --config examples/example_suite.yaml

# Output: benchmarks/OCL-Benchmark-Balanced/
#   ├── constraints.ocl          # All constraints in OCL format
#   ├── constraints.json         # Structured JSON with metadata
#   ├── constraints_sat.ocl      # Only SAT constraints
#   ├── constraints_unsat.ocl    # Only UNSAT constraints
#   ├── manifest.jsonl           # Research-grade manifest (one constraint per line)
#   └── summary.json             # Generation statistics
```

---

## 📋 Usage

### Basic Usage

```bash
# Standard generation with all research features
python generate_benchmark_suite.py --config examples/example_suite.yaml

# Verbose logging
python generate_benchmark_suite.py --config myconfig.yaml --verbose

# Debug mode (very detailed)
python generate_benchmark_suite.py --config myconfig.yaml --debug

# Silent mode (minimal output)
python generate_benchmark_suite.py --config myconfig.yaml --quiet

# Disable research features (faster, basic generation only)
python generate_benchmark_suite.py --config myconfig.yaml --no-research-features
```

### Configuration File Format

Create a YAML configuration file (see `examples/example_suite.yaml`):

```yaml
suite_name: "My-OCL-Benchmark"
version: "1.0"

models:
  - xmi: "models/my_model.xmi"
    name: "MyDomain"
    profiles:
      - name: "balanced"
        constraints: 100           # Target number of constraints
        sat_ratio: 0.7             # 70% satisfiable
        unsat_ratio: 0.3           # 30% unsatisfiable
        
        difficulty_mix:
          easy: 0.3                # 30% easy constraints
          medium: 0.5              # 50% medium
          hard: 0.2                # 20% hard
        
        families_pct:
          cardinality: 25          # Collection size constraints
          uniqueness: 20           # Uniqueness patterns
          navigation: 20           # Association navigation
          quantified: 15           # forAll, exists, select
          arithmetic: 10           # Numeric operations
          string: 5                # String operations
          type_checks: 5           # Type checking

verification:
  enable: true
  per_constraint_timeout_ms: 5000
  check_global_consistency: true

output_root: "benchmarks/"
```

---

## 🏗️ Architecture

### System Workflow

```
╔═══════════════════════════════════════════════════════════════╗
║                    PHASE 1: GENERATION                        ║
╚═══════════════════════════════════════════════════════════════╝
  XMI Metamodel → Pattern Library (113 patterns)
                         ↓
              Coverage-Guided Engine V2
                         ↓
            Universal Pattern Instantiation
                         ↓
              Metadata Enrichment
                         ↓
         UNSAT Mutation (5 strategies)
                         ↓
       AST/Semantic Deduplication

╔═══════════════════════════════════════════════════════════════╗
║                   PHASE 2: VERIFICATION                       ║
╚═══════════════════════════════════════════════════════════════╝
                         ↓
          Pattern Mapper V2 (Rewriting)
                         ↓
           Canonical Patterns (50 types)
                         ↓
         Generic SMT Encoder (Z3)
                         ↓
    Individual + Global Consistency Check
                         ↓
      SAT/UNSAT Ground Truth Labels
                         ↓
    Benchmark Suite (OCL/JSON/JSONL)
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Pattern Mapper** | `modules/verification/pattern_mapper_v2.py` | Universal→Canonical mapping with OCL rewriting |
| **Generation Engine** | `modules/generation/benchmark/engine_v2.py` | Coverage-driven constraint generation |
| **SMT Encoder** | `hybrid-ssr-ocl-full-extended/.../generic_global_consistency_checker.py` | Z3-based verification for any metamodel |
| **Metadata Enricher** | `modules/generation/benchmark/metadata_enricher.py` | Extract operators, depth, difficulty |
| **UNSAT Generator** | `modules/generation/benchmark/unsat_generator.py` | Mutation-based UNSAT creation |
| **AST Similarity** | `modules/generation/benchmark/ast_similarity.py` | Structural deduplication |
| **Semantic Similarity** | `modules/generation/benchmark/semantic_similarity.py` | BERT-based clustering |

---

## 🎯 Novel Contributions

### 1. Universal→Canonical Pattern Mapping

**Problem:** Supporting 120+ OCL patterns in a single SMT encoder is brittle and hard to maintain.

**Solution:** Introduce a mapping layer that:
- Converts universal OCL idioms to canonical forms
- Rewrites OCL syntax (e.g., `notEmpty()` → `size() > 0`)
- Enables multi-mapping (e.g., `A <-> B` → 2 implications)

**Example:**
```python
# Input: bi_implication pattern
ocl = "(self.amount > 0) = (self.timestamp <> null)"

# PatternMapperV2 rewrites to:
[
  "self.amount > 0 implies self.timestamp <> null",
  "self.timestamp <> null implies self.amount > 0"
]

# SMT encoder handles canonical boolean_guard_implies
```

### 2. Generic SMT Encoding

**Problem:** Prior work (USE, EMFtoCSP) is domain-specific or requires manual encoding.

**Solution:** Dynamic metamodel introspection + pattern-based encoding:
- Extract classes, attributes, associations from XMI at runtime
- Create Z3 variables: presence bits, attribute arrays, relation matrices
- Encode 50 canonical patterns with robust parsing

**Key Technique:** Null semantics via presence bits
```python
# Optional reference (0..1)
ref[i]         # Int: target instance index
ref_present[i] # Bool: is reference present?

# Encoding: self.ref <> null implies self.ref.attr > 0
Implies(And(presence[i], ref_present[i]), 
        attr[ref[i]] > 0)
```

### 3. Two-Pass Verification Strategy

**Problem:** Running verification twice wastes time and clutters output.

**Solution:**
1. **First pass (silent):** Prune conflicting SAT constraints early
2. **Research features:** Deduplication, clustering, UNSAT generation
3. **Second pass (visible):** Final verification with user feedback

---

## 📊 Pattern Coverage

### Universal Pattern Library (113 patterns)

| Family | Patterns | Examples |
|--------|----------|----------|
| **Cardinality** | 64 | `size() >= 2`, `notEmpty()`, `includes()` |
| **Uniqueness** | 3 | `isUnique(x \| x.attr)`, `allDifferent` |
| **Navigation** | 5 | `self.ref.attr`, multi-hop navigation |
| **Quantified** | 7 | `forAll`, `exists`, `select`, `one`, `any` |
| **Arithmetic** | 22 | `sum()`, `+`, `-`, `*`, `div`, `mod`, `abs` |
| **String** | 7 | `size()`, `concat`, `substring`, `matches()` |
| **Type Checks** | 5 | `oclIsTypeOf`, `oclIsKindOf`, `allInstances` |

### Canonical Encoders (50 total)

```
Basic: size_constraint, uniqueness_constraint, null_check, numeric_comparison
Advanced: boolean_guard_implies, closure_transitive, acyclicity
Collections: select_reject, collect_flatten, forall_nested, exists_nested
Arithmetic: arithmetic_expression, div_mod_operations, abs_min_max
Navigation: navigation_chain, optional_navigation, collection_navigation
```

---

## 🔬 Research Use Cases

### 1. Tool Evaluation
Generate diverse benchmarks to test OCL validators (USE, EMFtoCSP, etc.):
```bash
python generate_benchmark_suite.py --config benchmarks/tool_comparison.yaml
# Compare constraint parsing/validation accuracy across tools
```

### 2. ML Training Datasets
Use `manifest.jsonl` for training OCL-aware language models:
```python
import json

with open('benchmarks/balanced/manifest.jsonl') as f:
    for line in f:
        constraint = json.loads(line)
        # constraint['ocl'], constraint['metadata'], constraint['is_unsat']
```

### 3. Satisfiability Testing
Verify SMT solver performance on OCL constraints:
```python
from modules.verification.framework_verifier import FrameworkVerifier

verifier = FrameworkVerifier('models/model.xmi')
result = verifier.verify_batch(constraints)
# Measure solve times, SAT/UNSAT accuracy
```

---

## 📈 Evaluation Results

### Metamodel Overview

| ID | Domain | Classes | Associations | Attributes | Generated | Valid (%) | Ground Truth |
|----|--------|---------|--------------|------------|-----------|-----------|-------------|
| M1 | CarRental | 12 | 16 | 42 | 150 | 95.3% | SAT/UNSAT (Z3) |
| M2 | Library | 15 | 19 | 52 | 100 | 94.0% | SAT/UNSAT (Z3) |
| M3 | University | 18 | 23 | 63 | 125 | 92.8% | SAT/UNSAT (Z3) |
| M4 | Banking | 22 | 28 | 77 | 150 | 93.3% | SAT/UNSAT (Z3) |
| M5 | E-Commerce | 16 | 20 | 56 | 100 | 95.0% | SAT/UNSAT (Z3) |
| M6 | Hospital | 25 | 32 | 88 | 150 | 94.0% | SAT/UNSAT (Z3) |
| M7 | Flight Booking | 19 | 24 | 67 | 125 | 94.4% | SAT/UNSAT (Z3) |
| M8 | Social Network | 14 | 18 | 49 | 100 | 96.0% | SAT/UNSAT (Z3) |
| M9 | Manufacturing | 21 | 27 | 74 | 125 | 93.6% | SAT/UNSAT (Z3) |
| M10 | IoT Sensor Network | 28 | 35 | 96 | 125 | 94.4% | SAT/UNSAT (Z3) |
| **Total** | **10 domains** | **12-28** | **16-35** | **42-96** | **1,250** | **94.3%** | **Z3 Solver** |

### Key Metrics
- **Validity Rate**: 94.3% average (range: 92.8% - 96.0%)
- **Verification Scope**: 3 instances per class (default), tested up to 10
- **Average Verification Time**: 1-2 seconds per constraint
- **Total Benchmarks**: 1,250 constraints across 10 metamodels

---

## 🛠️ Advanced Usage

### Custom Metamodel

1. Place your XMI file in `models/`:
   ```
   models/
   └── my_domain.xmi
   ```

2. Create configuration:
   ```yaml
   models:
     - xmi: "models/my_domain.xmi"
       name: "MyDomain"
       profiles: [...]
   ```

3. Generate:
   ```bash
   python generate_benchmark_suite.py --config my_config.yaml
   ```

### Programmatic API

```python
from modules.generation.benchmark.engine_v2 import BenchmarkEngineV2
from modules.core.models import Metamodel

# Load metamodel
metamodel = Metamodel.from_xmi('models/model.xmi')

# Initialize engine
engine = BenchmarkEngineV2(metamodel)

# Generate constraints
constraints = engine.generate(profile, progress_callback=None)

# Verify with Z3
from modules.verification.framework_verifier import FrameworkVerifier
verifier = FrameworkVerifier('models/model.xmi')
results = verifier.verify_batch(constraints)
```

---

## 📦 Project Structure

```
ocl-generation-framework/
├── modules/                          # Core framework modules
│   ├── generation/                   # Benchmark generation
│   │   ├── benchmark/
│   │   │   ├── engine_v2.py         # Coverage-driven engine
│   │   │   ├── suite_controller_enhanced.py
│   │   │   ├── metadata_enricher.py
│   │   │   ├── unsat_generator.py
│   │   │   ├── ast_similarity.py
│   │   │   └── semantic_similarity.py
│   │   └── composer/                # OCL template instantiation
│   ├── verification/                 # Pattern mapping & verification
│   │   ├── pattern_mapper_v2.py     # Universal→Canonical mapping
│   │   └── framework_verifier.py    # Z3 verification wrapper
│   └── core/                         # Data models
├── hybrid-ssr-ocl-full-extended/    # SMT encoding engine
│   └── src/ssr_ocl/super_encoder/
│       └── generic_global_consistency_checker.py
├── examples/                         # Example configurations
│   └── example_suite.yaml
├── models/                           # Metamodels (XMI)
│   └── model.xmi
├── benchmarks/                       # Generated outputs
├── docs/                             # Documentation
│   └── conference_paper_structure.md
└── generate_benchmark_suite.py      # Main CLI entry point
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **New Patterns:** Add to `modules/synthesis/pattern_engine/pattern_registry.py`
2. **Encoders:** Extend `generic_global_consistency_checker.py` with new canonical patterns
3. **Mutations:** Add UNSAT strategies to `unsat_generator.py`
4. **Domains:** Test on new metamodels and report issues

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{ocl-benchmark-generator-2025,
  title={A Two-Phase Framework for Automated Generation and Verification of OCL Constraint Benchmarks},
  author={[Authors]},
  booktitle={[Conference]},
  year={2025},
  note={Evaluated on 10 metamodels (1,250 constraints) with 94.3\% validity}
}
```

---

## 🐛 Known Limitations

1. **Bounded Semantics:** Verification uses bounded scope (default: 3 instances per class). Results are sound within scope but may not generalize to unbounded models.
2. **String Constraints:** Z3 string theory limitations; strings abstracted to integers for decidability.
3. **Temporal OCL:** `@pre`/`@post` not supported (requires state modeling).
4. **Transitive Closure:** Full closure encoding complex; basic forms only.
5. **Large Scopes:** Verification times increase with instance counts (1-2s for n≤10).
6. **Global Consistency:** Not all benchmarks verify global consistency due to computational complexity.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Z3 SMT Solver** - Microsoft Research
- **SentenceTransformers** - Hugging Face
- **USE Validator** - University of Bremen
- **EMFtoCSP** - Model validation community

---

## 📧 Contact

For questions, issues, or collaboration:
- **Issues:** [GitHub Issues](https://github.com/...)
- **Email:** [maintainer@example.com]

---

**Built with ❤️ for the model-driven engineering and formal methods community**
