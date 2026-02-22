"""SAC Demo Companion Server - connects the browser demo to a real sac2c compiler."""

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SAC Demo Companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROGRAMS_DIR = Path("/app/programs")
PRECOMPILED_DIR = Path("/app/precompiled")
COMPILE_TIMEOUT = 30
RUN_TIMEOUT = 10


def get_sac2c_version() -> str:
    try:
        result = subprocess.run(
            ["sac2c", "-V"], capture_output=True, text=True, timeout=5
        )
        # sac2c -V outputs version info on stderr or stdout
        output = result.stdout.strip() or result.stderr.strip()
        for line in output.splitlines():
            if "sac2c" in line.lower() or "version" in line.lower():
                return line.strip()
        return output.splitlines()[0].strip() if output else "sac2c (version unknown)"
    except Exception:
        return "sac2c (version unavailable)"


SAC2C_VERSION = get_sac2c_version()


# ---------- Models ----------

class CompileRunRequest(BaseModel):
    code: str
    target: str = "seq"  # "seq" or "mt_pth"
    threads: int = 4


class CompileRunResponse(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    compile_time_ms: float = 0
    run_time_ms: float = 0
    error: str = ""


class BenchmarkResult(BaseModel):
    threads: int
    time_ms: float
    speedup: float


class BenchmarkResponse(BaseModel):
    success: bool
    results: list[BenchmarkResult] = []
    error: str = ""


# ---------- Helpers ----------

async def compile_and_run(
    code: str, target: str = "seq", threads: int = 4
) -> CompileRunResponse:
    """Compile SAC code and run the resulting binary."""
    with tempfile.TemporaryDirectory(prefix="sac_") as tmpdir:
        src_path = os.path.join(tmpdir, "program.sac")
        bin_path = os.path.join(tmpdir, "program")

        with open(src_path, "w") as f:
            f.write(code)

        # Compile
        compile_start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                "sac2c", "-t", target, "-o", bin_path, src_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=COMPILE_TIMEOUT
            )
        except asyncio.TimeoutError:
            return CompileRunResponse(
                success=False,
                error=f"Compilation timed out after {COMPILE_TIMEOUT}s",
            )
        except Exception as e:
            return CompileRunResponse(success=False, error=f"Compile error: {e}")

        compile_time = (time.monotonic() - compile_start) * 1000

        if proc.returncode != 0:
            return CompileRunResponse(
                success=False,
                stderr=stderr.decode(errors="replace"),
                compile_time_ms=compile_time,
                error="Compilation failed",
            )

        # Run
        env = os.environ.copy()
        if target == "mt_pth":
            env["SAC_PARALLEL"] = str(threads)

        run_start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                bin_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=RUN_TIMEOUT
            )
        except asyncio.TimeoutError:
            return CompileRunResponse(
                success=False,
                compile_time_ms=compile_time,
                error=f"Execution timed out after {RUN_TIMEOUT}s",
            )
        except Exception as e:
            return CompileRunResponse(
                success=False,
                compile_time_ms=compile_time,
                error=f"Runtime error: {e}",
            )

        run_time = (time.monotonic() - run_start) * 1000

        return CompileRunResponse(
            success=proc.returncode == 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            compile_time_ms=compile_time,
            run_time_ms=run_time,
            error="" if proc.returncode == 0 else "Program exited with non-zero status",
        )


async def run_precompiled(name: str, target: str, threads: int) -> CompileRunResponse:
    """Run a pre-compiled binary if available, fall back to compiling from source."""
    bin_path = PRECOMPILED_DIR / f"{name}_{target}"
    if bin_path.exists():
        env = os.environ.copy()
        if target == "mt_pth":
            env["SAC_PARALLEL"] = str(threads)

        run_start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                str(bin_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=RUN_TIMEOUT
            )
        except asyncio.TimeoutError:
            return CompileRunResponse(
                success=False,
                error=f"Execution timed out after {RUN_TIMEOUT}s",
            )
        except Exception as e:
            return CompileRunResponse(success=False, error=f"Runtime error: {e}")

        run_time = (time.monotonic() - run_start) * 1000
        return CompileRunResponse(
            success=proc.returncode == 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            compile_time_ms=0,
            run_time_ms=run_time,
        )

    # Fall back to compiling from source
    src_path = PROGRAMS_DIR / f"{name}.sac"
    if src_path.exists():
        code = src_path.read_text()
        return await compile_and_run(code, target, threads)

    return CompileRunResponse(success=False, error=f"Program '{name}' not found")


# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sac2c_version": SAC2C_VERSION,
        "precompiled": [
            f.stem for f in PRECOMPILED_DIR.glob("*") if f.is_file()
        ] if PRECOMPILED_DIR.exists() else [],
    }


@app.post("/compile-run", response_model=CompileRunResponse)
async def compile_run(req: CompileRunRequest):
    if req.target not in ("seq", "mt_pth"):
        return CompileRunResponse(success=False, error="Target must be 'seq' or 'mt_pth'")

    # Handle pre-compiled program sentinel
    if req.code.startswith("USE_PRECOMPILED:"):
        name = req.code.split(":", 1)[1].strip()
        return await run_precompiled(name, req.target, req.threads)

    if len(req.code) > 50_000:
        return CompileRunResponse(success=False, error="Code too large (max 50KB)")
    return await compile_and_run(req.code, req.target, req.threads)


@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark():
    """Run the parallel benchmark: compile mandelbrot for seq and mt_pth,
    then run mt_pth with 1, 2, 4, 8 threads."""
    thread_counts = [1, 2, 4, 8]
    results: list[BenchmarkResult] = []

    # First, get the sequential baseline
    seq_result = await run_precompiled("parallel_bench", "seq", 1)
    if not seq_result.success:
        # Try compiling from source
        src_path = PROGRAMS_DIR / "parallel_bench.sac"
        if src_path.exists():
            seq_result = await compile_and_run(src_path.read_text(), "seq", 1)
        if not seq_result.success:
            return BenchmarkResponse(
                success=False,
                error=f"Sequential compilation/run failed: {seq_result.error or seq_result.stderr}",
            )

    # Parse timing from sequential output (expects "TIME_MS: <number>")
    seq_time = _parse_time(seq_result.stdout)
    if seq_time is None:
        # Use wall-clock run_time_ms as fallback
        seq_time = seq_result.run_time_ms

    results.append(BenchmarkResult(threads=1, time_ms=seq_time, speedup=1.0))

    # Now run mt_pth with varying thread counts
    for tc in thread_counts[1:]:
        mt_result = await run_precompiled("parallel_bench", "mt_pth", tc)
        if not mt_result.success:
            src_path = PROGRAMS_DIR / "parallel_bench.sac"
            if src_path.exists():
                mt_result = await compile_and_run(src_path.read_text(), "mt_pth", tc)

        if mt_result.success:
            mt_time = _parse_time(mt_result.stdout)
            if mt_time is None:
                mt_time = mt_result.run_time_ms
            speedup = seq_time / mt_time if mt_time > 0 else 0
            results.append(BenchmarkResult(threads=tc, time_ms=mt_time, speedup=speedup))
        else:
            results.append(BenchmarkResult(threads=tc, time_ms=0, speedup=0))

    return BenchmarkResponse(success=True, results=results)


def _parse_time(output: str) -> float | None:
    """Parse 'TIME_MS: <number>' from program output."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("TIME_MS:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return None
