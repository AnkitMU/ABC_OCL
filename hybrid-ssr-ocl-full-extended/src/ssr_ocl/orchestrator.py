import time, os, pathlib
from ssr_ocl.verification.pipeline import verify_constraints
from ssr_ocl.config_loader import load_cfg
from ssr_ocl.utils.io_utils import read_text
from ssr_ocl.reports.artifacts import write_run_artifacts

def run_verification(model_path: str, ocl_path: str, cfg_path: str) -> bool:
    cfg = load_cfg(cfg_path)
    ocl_text = read_text(ocl_path)
    results = verify_constraints(model_path, ocl_text, cfg)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("reports", "runs", ts)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_run_artifacts(out_dir, results)
    return all(r.status in ("UNSAT", "VALID") for r in results)
