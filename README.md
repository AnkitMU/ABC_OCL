# Automatic Benchmark Generation Framework for OCL(Object Constraint Language)

Our framework generates OCL benchmarks from UML/Ecore metamodels. It generates feature driven diverse OCL constraints which are solver verified (Z3 SMT) through a verification pipeline.

## Feature of Framework
- Generate OCL constraints from a pattern library based on user Configuration
- Create SAT and UNSAT constraints
- Add metadata (operators, difficulty, depth, etc.)
-  Remove deduplicate and analyze constraints.
- verify entire benchmark using SMT solver (Z3)
- Export JSON/JSONL and OCL files

## Quick start

```bash
# install deps
pip install -r requirements.txt

# run example suite
python generate_benchmark_suite.py --config examples/example_suite.yaml
```

Outputs are written to `benchmarks/`.

## Configuration
Edit a YAML file like `examples/example_suite.yaml` to control:
- models (XMI paths)
- number of constraints
- SAT/UNSAT ratio
- pattern family mix
- verification options

## Output files
Typical run produces:
- `constraints.ocl`
- `constraints.json`
- `constraints_sat.ocl` / `constraints_unsat.ocl`
- `manifest.jsonl`
- summary JSON


- Solver verification is slower(affected by number of constraints to be verified) but gives verification labels (SAT/UNSAT).
- Adavance features (similarity, implication checks, metadata_label) are enabled by default in the config file (example_suite.yaml).

---
## GenAI Disclosure & Usage
This project acknowledges the use of Generative AI (GenAI) in its development and debugging:

Development: GenAI tools were utilized for code optimization, documentation assistance and refining the natural language patterns within the OCL generation engine.

Verification: No constraint is added to a benchmark without passing the SMT verification pipeline. Every SAT/UNSAT label is confirmed by the Z3 solver, ensuring the ground truth is mathematically sound and not subject to AI hallucinations.
