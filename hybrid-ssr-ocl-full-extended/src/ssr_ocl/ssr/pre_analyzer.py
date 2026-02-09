from ssr_ocl.types import Candidate

def pre_analyze(ocl_text: str) -> Candidate:
    context = "Unknown"
    for line in ocl_text.splitlines():
        if line.strip().startswith("context "):
            parts = line.strip().split()
            if len(parts) >= 2: context = parts[1]
            break
    confidence = 0.7 if (">=" in ocl_text or "forAll" in ocl_text or "exists" in ocl_text) else 0.5
    return Candidate(context=context, ocl=ocl_text.strip(), confidence=confidence)
