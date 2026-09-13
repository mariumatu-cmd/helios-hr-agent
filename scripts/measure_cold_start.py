"""Measure a real Render free-tier cold start.

Free instances spin down after ~15 minutes idle. This waits out the idle
window, then times the first request, so deployed.md can quote a measured
number rather than an estimate. The wait is the point -- do not shorten it, or
the measurement is just a warm request.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.request

BASE = "https://helios-hr-assistant-wz3c.onrender.com"
IDLE_SECONDS = 17 * 60


def timed(path: str, timeout: int = 300) -> tuple[int, float]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as resp:
            resp.read()
            code = resp.status
    except Exception as exc:  # noqa: BLE001 - any failure is a data point
        print(f"  {path} raised {type(exc).__name__}: {exc}")
        code = 0
    return code, (time.perf_counter() - started) * 1000


print(f"idling {IDLE_SECONDS // 60} minutes so the instance spins down...")
time.sleep(IDLE_SECONDS)

print("timing the cold wake on /healthz")
cold_code, cold_ms = timed("/healthz")
print(f"  cold  /healthz : {cold_code} in {cold_ms / 1000:.1f}s")

warm_code, warm_ms = timed("/healthz")
health_code, health_ms = timed("/health")
print(f"  warm  /healthz : {warm_code} in {warm_ms:.0f}ms")
print(f"  warm  /health  : {health_code} in {health_ms:.0f}ms")

out = pathlib.Path("evidence")
out.mkdir(exist_ok=True)
(out / "cold-start.json").write_text(
    json.dumps(
        {
            "base_url": BASE,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "idle_seconds_before": IDLE_SECONDS,
            "cold_healthz_ms": round(cold_ms),
            "warm_healthz_ms": round(warm_ms),
            "warm_health_ms": round(health_ms),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("wrote evidence/cold-start.json")
