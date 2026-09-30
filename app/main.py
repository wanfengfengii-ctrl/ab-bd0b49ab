"""FastAPI application exposing the delay-plan compiler."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .schemas import CompileRequest
from .solver import SolverResourceError, compile_plan

app = FastAPI(
    title="Ultrasound Delay Plan Compiler",
    version="1.0.0",
    description=(
        "Compiles per-element target delays into an integer delay sequence "
        "with a bounded number of ramps, honouring a global delay interval, "
        "a max adjacent-step limit and exact element anchors. Feasible plans "
        "minimise, in order: max absolute error, total absolute error, "
        "actual ramp count, and the lexicographic delay sequence."
    ),
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/api/delay-plans/compile")
def compile_delay_plan(request: CompileRequest) -> dict:
    """Compile one delay plan.

    Returns HTTP 200 with ``feasible: true`` and the optimal plan, or
    HTTP 200 with ``feasible: false`` plus a stable infeasibility reason
    (including the conflicting element interval for anchor/step clashes)
    and no partial delay table.  Malformed or out-of-envelope requests
    are rejected with HTTP 422.
    """
    try:
        return compile_plan(request.to_instance())
    except SolverResourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
