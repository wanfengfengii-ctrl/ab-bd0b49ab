"""One-shot verification entry point used by the docker-compose ``verify``
service.

Aggregates, and reports through the process exit code:

1. **code tests**       -- the pytest suite shipped inside the image;
2. **image artifacts**  -- the build artifacts expected inside the image
                           (application package, version metadata,
                           dependencies, generated OpenAPI schema);
3. **API smoke**        -- typical compile requests executed over HTTP
                           against the live ``api`` service.

Exit code 0 means every group passed; 1 means at least one group failed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
HTTP_TIMEOUT = 10
STARTUP_DEADLINE_S = 60


# ---------------------------------------------------------------------------
# 1. code tests
# ---------------------------------------------------------------------------

def run_code_tests() -> bool:
    import pytest

    print("[verify] running unit/integration test suite ...", flush=True)
    rc = pytest.main(["-q", "--tb=short", "tests"])
    ok = rc == 0
    print(f"[verify] test suite {'passed' if ok else 'FAILED'} (rc={rc})",
          flush=True)
    return ok


# ---------------------------------------------------------------------------
# 2. image build artifacts
# ---------------------------------------------------------------------------

def check_image_artifacts() -> bool:
    print("[verify] checking image build artifacts ...", flush=True)
    checks: list[tuple[str, bool]] = []

    import app

    checks.append(("app package exposes __version__",
                   isinstance(getattr(app, "__version__", None), str)))

    base = os.path.dirname(os.path.abspath(app.__file__))
    for rel in ("__init__.py", "main.py", "schemas.py", "solver.py",
                "verify.py"):
        checks.append((f"app/{rel} present in image",
                       os.path.isfile(os.path.join(base, rel))))

    tests_dir = os.path.join(os.path.dirname(base), "tests")
    checks.append(("tests/ directory present in image",
                   os.path.isdir(tests_dir)))

    for mod in ("fastapi", "uvicorn", "pydantic", "numpy"):
        try:
            __import__(mod)
            checks.append((f"dependency '{mod}' importable", True))
        except Exception:  # pragma: no cover - defensive
            checks.append((f"dependency '{mod}' importable", False))

    try:
        from app.main import app as fastapi_app

        schema = fastapi_app.openapi()
        checks.append(("OpenAPI schema generates",
                       "/api/delay-plans/compile" in schema.get("paths", {})))
    except Exception:  # pragma: no cover - defensive
        checks.append(("OpenAPI schema generates", False))

    ok = True
    for name, passed in checks:
        ok = ok and passed
        print(f"[verify]   {'ok  ' if passed else 'FAIL'} {name}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# 3. API smoke tests
# ---------------------------------------------------------------------------

def _http(method: str, path: str, payload: dict | None = None):
    url = API_BASE_URL.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        body = resp.read().decode()
        return resp.status, json.loads(body) if body else None


def _wait_for_api() -> bool:
    deadline = time.monotonic() + STARTUP_DEADLINE_S
    while time.monotonic() < deadline:
        try:
            status, body = _http("GET", "/healthz")
            if status == 200 and body.get("status") == "ok":
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1.0)
    return False


def _check_plan_invariants(body: dict, request: dict) -> list[str]:
    """Re-verify every contract a feasible response must satisfy."""
    problems: list[str] = []
    delays = body["delays"]
    targets = request["targets"]
    n = len(targets)
    if len(delays) != n:
        return [f"delays length {len(delays)} != {n}"]
    if len(body["errors"]) != n:
        problems.append("errors length mismatch")
    if [d - t for d, t in zip(delays, targets)] != body["errors"]:
        problems.append("errors are not delays - targets")
    if not all(request["minDelay"] <= d <= request["maxDelay"] for d in delays):
        problems.append("delay outside global interval")
    for a in request["anchors"]:
        if delays[a["element"]] != a["delay"]:
            problems.append(f"anchor at element {a['element']} not hit")
    for i in range(1, n):
        if abs(delays[i] - delays[i - 1]) > request["maxStep"]:
            problems.append(f"step limit violated at element {i}")
    diffs = [delays[i + 1] - delays[i] for i in range(n - 1)]
    runs = 1 + sum(1 for i in range(1, len(diffs)) if diffs[i] != diffs[i - 1])
    if runs != body["rampCount"]:
        problems.append("rampCount does not match recomputed runs")
    if runs > request["maxRamps"]:
        problems.append("ramp limit exceeded")
    if len(body["ramps"]) != runs:
        problems.append("ramps list length mismatch")
    pos = 0
    for ramp in body["ramps"]:
        if ramp["startElement"] != pos:
            problems.append("ramp boundaries are not contiguous")
        if ramp["startValue"] != delays[ramp["startElement"]] or \
                ramp["endValue"] != delays[ramp["endElement"]]:
            problems.append("ramp boundary values inconsistent with delays")
        span = ramp["endElement"] - ramp["startElement"]
        if span < 1 or delays[ramp["endElement"]] - delays[ramp["startElement"]] != ramp["slope"] * span:
            problems.append("ramp slope inconsistent with boundary values")
        pos = ramp["endElement"]
    if pos != n - 1:
        problems.append("ramps do not cover all elements")
    abs_errors = [abs(e) for e in body["errors"]]
    if body["maxAbsError"] != max(abs_errors):
        problems.append("maxAbsError inconsistent")
    if body["totalAbsError"] != sum(abs_errors):
        problems.append("totalAbsError inconsistent")
    return problems


def run_api_smoke() -> bool:
    print(f"[verify] running API smoke tests against {API_BASE_URL} ...",
          flush=True)
    if not _wait_for_api():
        print("[verify]   FAIL api did not become healthy in time", flush=True)
        return False
    print("[verify]   ok   /healthz reports ok", flush=True)

    # typical feasible compile request: 24 elements, two-slope target
    feasible_req = {
        "targets": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9],
        "minDelay": 0,
        "maxDelay": 40,
        "maxStep": 2,
        "maxRamps": 3,
        "anchors": [
            {"element": 0, "delay": 10},
            {"element": 12, "delay": 20},
            {"element": 23, "delay": 9},
        ],
    }
    # anchor/step conflict: |20 - 0| = 20 > 2 * (5 - 1) = 8
    conflict_req = {
        "targets": [0] * 12,
        "minDelay": 0,
        "maxDelay": 40,
        "maxStep": 2,
        "maxRamps": 4,
        "anchors": [
            {"element": 1, "delay": 0},
            {"element": 5, "delay": 20},
        ],
    }
    # ramp limit too tight for the forced zig-zag between anchors
    ramp_limit_req = {
        "targets": [0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        "minDelay": 0,
        "maxDelay": 10,
        "maxStep": 2,
        "maxRamps": 2,
        "anchors": [
            {"element": 0, "delay": 0},
            {"element": 2, "delay": 2},
            {"element": 4, "delay": 0},
            {"element": 6, "delay": 2},
            {"element": 11, "delay": 7},
        ],
    }

    ok = True
    try:
        status, body = _http("POST", "/api/delay-plans/compile", feasible_req)
        if status != 200 or not body.get("feasible"):
            print(f"[verify]   FAIL feasible request -> {status} {body}",
                  flush=True)
            ok = False
        else:
            problems = _check_plan_invariants(body, feasible_req)
            if problems:
                for p in problems:
                    print(f"[verify]   FAIL feasible plan: {p}", flush=True)
                ok = False
            else:
                print(f"[verify]   ok   feasible plan "
                      f"(maxAbsError={body['maxAbsError']}, "
                      f"totalAbsError={body['totalAbsError']}, "
                      f"ramps={body['rampCount']})", flush=True)

        status, body = _http("POST", "/api/delay-plans/compile", conflict_req)
        conflict = (body or {}).get("conflict", {})
        good = (status == 200 and body.get("feasible") is False
                and body.get("reason") == "anchor-step-conflict"
                and body.get("delays") is None
                and conflict.get("startElement") == 1
                and conflict.get("endElement") == 5
                and conflict.get("requiredChange") == 20
                and conflict.get("allowedChange") == 8)
        print(f"[verify]   {'ok  ' if good else 'FAIL'} conflict request -> "
              f"{status} reason={body.get('reason')!r}", flush=True)
        ok = ok and good

        status, body = _http("POST", "/api/delay-plans/compile", ramp_limit_req)
        good = (status == 200 and body.get("feasible") is False
                and body.get("reason") == "ramp-limit-exceeded"
                and body.get("delays") is None
                and body.get("minRampsRequired") == 3)
        print(f"[verify]   {'ok  ' if good else 'FAIL'} ramp-limit request -> "
              f"{status} reason={body.get('reason')!r}", flush=True)
        ok = ok and good
    except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
        print(f"[verify]   FAIL smoke request raised {exc!r}", flush=True)
        ok = False
    return ok


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> int:
    results: list[tuple[str, bool]] = []
    results.append(("code tests", run_code_tests()))
    results.append(("image build artifacts", check_image_artifacts()))
    results.append(("API smoke", run_api_smoke()))

    print("[verify] ================= summary =================", flush=True)
    for name, ok in results:
        print(f"[verify]   {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    all_ok = all(ok for _, ok in results)
    print(f"[verify] overall: {'PASS' if all_ok else 'FAIL'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
