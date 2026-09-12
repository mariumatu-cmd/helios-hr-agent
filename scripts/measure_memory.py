"""Measure the resident memory footprint of a fully warmed service.

    python scripts/measure_memory.py

Render's free web-service tier caps a container at 512 MB RSS. The embedding
model is the only component large enough to threaten that, and it is loaded
lazily on the first retrieval call -- so a naive measurement taken at boot is
misleading. This script measures three points:

  1. after import      -- the floor
  2. after app startup -- MCP session + index metadata, still no embedder
  3. after a query     -- the embedder is resident; this is the real ceiling

The number printed at step 3 is what has to fit in 512 MB.
"""
from __future__ import annotations

import gc
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUDGET_MB = 512.0


def rss_mb() -> float:
    """Resident set size in MB, without taking a psutil dependency."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wt

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # argtypes must be declared: without them the 64-bit process HANDLE is
        # truncated to a C int and the call silently reports zero.
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wt.DWORD]
        get_info.restype = wt.BOOL

        handle = ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess())
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not get_info(handle, ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.WorkingSetSize / (1024 * 1024)

    # Linux (the platform that actually matters for the deploy target).
    status = pathlib.Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024
    raise RuntimeError("could not read VmRSS")


def child_rss_mb() -> float:
    """RSS of the stdio MCP server process.

    Sampling the live child from the parent is unreliable on Windows -- an idle
    process blocked on a stdin read gets its working set trimmed, which reports
    ~4 MB for a process that actually holds ~73 MB of imports. Instead the child
    measures and reports itself immediately after importing the server module,
    which is the honest steady-state cost.
    """
    import subprocess

    probe = (
        "import sys; sys.path.insert(0, 'scripts')\n"
        "from measure_memory import rss_mb\n"
        "import mcp_server.hr_server\n"
        "print(rss_mb())\n"
    )
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", probe],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"child probe failed: {result.stderr[-500:]}")
    return float(result.stdout.strip().splitlines()[-1])


def main() -> int:
    marks: list[tuple[str, float]] = [("interpreter + stdlib", rss_mb())]

    from fastapi.testclient import TestClient

    from app.main import app

    marks.append(("after imports", rss_mb()))

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["rag_index"]["ok"], health["rag_index"]
        marks.append(("after app startup (MCP + index metadata)", rss_mb()))

        from rag.retrieve import search

        # Two distinct queries: the first pays the model-load cost, the second
        # shows whether steady-state query traffic keeps growing the heap.
        search("how many PTO days do I accrue")
        marks.append(("after first query (embedder resident)", rss_mb()))

        for query in (
            "what is the international remote work day limit",
            "who approves an exception to a blackout period",
            "how do I expense a hotel room",
        ):
            search(query)
        gc.collect()
        marks.append(("after four queries (steady state)", rss_mb()))

    peak = max(value for _, value in marks)
    child = child_rss_mb()
    width = max(len(label) for label, _ in marks)
    print(f"{'stage':<{width}}  {'RSS (MB)':>9}  {'delta':>8}")
    previous = 0.0
    for label, value in marks:
        print(f"{label:<{width}}  {value:>9.1f}  {value - previous:>+8.1f}")
        previous = value

    total = peak + child
    print()
    print(f"web process peak    : {peak:.1f} MB")
    print(f"MCP stdio child     : {child:.1f} MB")
    print(f"container total     : {total:.1f} MB")
    print(f"free-tier budget    : {BUDGET_MB:.0f} MB")
    print(f"headroom            : {BUDGET_MB - total:.1f} MB ({100 * total / BUDGET_MB:.0f}% used)")
    print(f"platform            : {sys.platform} (python {sys.version.split()[0]})")

    if total > BUDGET_MB:
        print("\nOVER BUDGET -- the service will be OOM-killed on Render free tier.")
        return 1
    if total > BUDGET_MB * 0.85:
        print("\nTIGHT -- under budget but with little headroom.")
        return 0
    print("\nWITHIN BUDGET")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("FASTEMBED_CACHE_PATH", ".fastembed_cache")
    raise SystemExit(main())
