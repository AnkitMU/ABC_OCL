import yaml, os
def load_cfg(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scopes_file = cfg.get('scopes_file', 'config/scopes.yaml')
    if os.path.exists(scopes_file):
        with open(scopes_file, 'r', encoding='utf-8') as sf:
            cfg['__scopes'] = yaml.safe_load(sf)
    else:
        cfg['__scopes'] = {"classes": {}, "enums": {}}
    return cfg
