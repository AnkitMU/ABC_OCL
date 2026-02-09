import logging, logging.config, yaml
def setup_logging(path: str = "config/logging.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logging.config.dictConfig(cfg)
