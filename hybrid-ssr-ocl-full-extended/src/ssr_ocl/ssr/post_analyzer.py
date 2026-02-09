from ssr_ocl.utils.pretty_print import describe_counterexample
def explain_and_suggest(result, original_ocl: str) -> str:
    if result.status == "UNSAT": return "Verified (within current scope)."
    if result.status == "SAT":   return "Potential violation. " + describe_counterexample(result.counterexample)
    if result.status in ("TYPE_ERROR","SYNTAX_ERROR"): return "Parsing/Typing issue – check navigation or operators."
    return "Unknown status."
