import typer
from ssr_ocl.orchestrator import run_verification
app = typer.Typer(help="Hybrid SSR + SMT (Z3) OCL Verification")

@app.command()
def verify(model: str, ocl: str, config: str = "config/framework.yaml"):
    ok = run_verification(model_path=model, ocl_path=ocl, cfg_path=config)
    raise typer.Exit(code=0 if ok else 1)

if __name__ == "__main__":
    app()
