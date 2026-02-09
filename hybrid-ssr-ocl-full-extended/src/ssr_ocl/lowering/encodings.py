from z3 import Int, Bool, EnumSort

def int_array(prefix: str, n: int):
    return [Int(f"{prefix}_{i}") for i in range(n)]

def enum_sort(name: str, literals: list):
    return EnumSort(name, literals)

def bool_symbol(name: str):
    return Bool(name)
