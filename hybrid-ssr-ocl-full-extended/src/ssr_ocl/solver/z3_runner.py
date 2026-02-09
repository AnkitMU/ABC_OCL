# src/ssr_ocl/solver/z3_runner.py
from time import perf_counter
from z3 import sat, unsat

def _z3_to_py(val):
    """Best-effort conversion of Z3 values to JSON-serializable Python types."""
    if val is None:
        return None
    s = str(val)

    # BoolRef -> bool
    if s == "True":
        return True
    if s == "False":
        return False

    # Int / Real numerals -> int/float when possible
    try:
        # IntNumRef has as_long(); Algebraic numbers may have as_decimal()
        if hasattr(val, "as_long"):
            return int(val.as_long())
    except Exception:
        pass
    try:
        if hasattr(val, "as_decimal"):
            dec = val.as_decimal(20)  # string like "3.14" or "1/3?"
            if dec.endswith("?"):
                dec = dec[:-1]
            if "/" in dec:
                num, den = dec.split("/", 1)
                return float(int(num) / int(den))
            return float(dec)
    except Exception:
        pass

    # Enums and everything else -> string
    return s

def run_solver(solver, model_vars):
    t0 = perf_counter()
    
    # Get solver statistics before solving
    try:
        num_assertions = len(solver.assertions()) if hasattr(solver, 'assertions') else 0
        assertions = [str(assertion) for assertion in solver.assertions()] if hasattr(solver, 'assertions') else []
        solver_info = {
            "num_assertions": num_assertions,
            "assertions": assertions
        }
    except Exception:
        solver_info = {
            "num_assertions": 0,
            "assertions": []
        }
    
    res = solver.check()
    ms = (perf_counter() - t0) * 1000.0

    if res == sat:
        m = solver.model()
        cx = {}
        # Use model_completion=True to get defaults for unassigned symbols
        for name, sym in model_vars.items():
            z = m.eval(sym, model_completion=True)
            cx[name] = _z3_to_py(z)
        
        # Add solver diagnostics to counterexample
        cx["_z3_info"] = {
            "result": "SAT",
            "num_assertions": solver_info["num_assertions"],
            "solver_time_ms": ms,
            "model_size": len(cx) - 1,  # Exclude _z3_info itself
        }
        
        return "SAT", cx, ms
    elif res == unsat:
        # For UNSAT, try to get an unsat core if possible
        unsat_info = {
            "result": "UNSAT", 
            "num_assertions": solver_info["num_assertions"],
            "solver_time_ms": ms,
            "assertions": solver_info["assertions"]
        }
        return "UNSAT", {"_z3_info": unsat_info}, ms
    
    unknown_info = {
        "result": "UNKNOWN",
        "num_assertions": solver_info["num_assertions"], 
        "solver_time_ms": ms,
        "reason": solver.reason_unknown() if hasattr(solver, 'reason_unknown') else "Unknown"
    }
    return "UNKNOWN", {"_z3_info": unknown_info}, ms
