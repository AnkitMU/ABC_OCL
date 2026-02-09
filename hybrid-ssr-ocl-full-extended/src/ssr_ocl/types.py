from pydantic import BaseModel
from typing import Optional, Dict, List

class Candidate(BaseModel):
    context: str
    ocl: str
    confidence: float = 0.5
    scope: Dict[str, int] = {}
    assumptions: List[str] = []
    hints: List[str] = []

class Result(BaseModel):
    status: str
    negationChecked: bool = True
    counterexample: Optional[dict] = None
    diagnostics: List[str] = []
    timeMs: float = 0.0
