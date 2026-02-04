# OCL Benchmark Generation Framework

This project generates OCL (Object Constraint Language) benchmarks from UML/Ecore metamodels. It builds diverse SAT/UNSAT constraints, enriches them with metadata, and can verify them with Z3 through the included verification pipeline.

## What it does
- Generate OCL constraints from a pattern library
- Create SAT and UNSAT sets
- Add metadata (operators, difficulty, depth, etc.)
- Deduplicate and analyze constraints
- Optionally verify constraints with Z3
- Export JSON/JSONL and OCL files

## Quick start

```bash
# install deps
pip install -r requirements.txt

# run example suite
python generate_benchmark_suite.py --config examples/example_suite.yaml
```

Outputs are written to `benchmarks/` by default.

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

## Notes
- Solver verification is slower but gives ground truth labels.
- Research features (similarity, implication checks, manifest) are enabled by default in the example config.

---

If you want a more detailed write‑up, check the `docs/` folder.
