def quick_sanity_check(ocl: str) -> bool:
    return "context" in ocl and "inv" in ocl
