def resolve_count(name: str, cfg: dict, default: int = 3) -> int:
    classes = (cfg.get('__scopes') or {}).get('classes', {})
    return int(classes.get(name, default))

def resolve_enum(name: str, cfg: dict):
    enums = (cfg.get('__scopes') or {}).get('enums', {})
    vals = enums.get(name)
    return vals if isinstance(vals, list) else None
