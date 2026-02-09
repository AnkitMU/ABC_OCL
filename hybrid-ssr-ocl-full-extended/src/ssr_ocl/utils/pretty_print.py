def describe_counterexample(model: dict) -> str:
    if not model: return "No counterexample."
    parts = [f"{k}={v}" for k, v in model.items()]
    return "Counterexample: " + ", ".join(parts)
