"""SaC Demo Companion Server - connects the browser demo to a real sac2c compiler."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SaC Demo Companion")

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


class ComputeRequest(BaseModel):
    operation: str      # "array_op", "tensor", "stencil", "nbody"
    params: dict = {}


class ComputeResponse(BaseModel):
    success: bool
    result: dict = {}
    compute_time_ms: float = 0
    error: str = ""


# ---------- Helpers ----------

async def compile_and_run(
    code: str, target: str = "seq", threads: int = 4
) -> CompileRunResponse:
    """Compile SaC code and run the resulting binary."""
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


# ---------- SaC Source Generators for /compute ----------

def generate_array_op_sac(op: str, shape: list[int], data: list[int]) -> str:
    """Generate SaC code that reshapes data, applies an operation, and outputs JSON."""
    ndim = len(shape)
    shape_str = ",".join(str(s) for s in shape)
    data_str = ",".join(str(d) for d in data)

    if ndim == 1:
        decl = f"  arr = [{data_str}];"
        arr_type = "int[.]"
    else:
        decl = f"  arr = reshape([{shape_str}], [{data_str}]);"
        arr_type = f"int[{','.join('.' for _ in shape)}]"

    # Build the operation
    if op == "transpose":
        if ndim == 1:
            op_code = "  result = reverse(arr);"
        else:
            op_code = "  result = transpose(arr);"
    elif op == "reverse":
        op_code = "  result = reverse(arr);"
    elif op == "rotate":
        op_code = "  result = rotate([1], arr);"
    elif op == "take":
        if ndim == 1:
            op_code = "  result = take([3], arr);"
        else:
            take_dims = ",".join(["3"] * ndim)
            op_code = f"  result = take([{take_dims}], arr);"
    elif op == "drop":
        if ndim == 1:
            op_code = "  result = drop([1], arr);"
        else:
            drop_dims = ",".join(["1"] * ndim)
            op_code = f"  result = drop([{drop_dims}], arr);"
    else:
        op_code = f'  result = arr; /* unknown op: {op} */'

    return f"""use Array: all;
use StdIO: all;

int main()
{{
{decl}
{op_code}

  shp = shape(result);
  flat = reshape([prod(shp)], result);
  ndims = shape(shp)[0];

  printf("{{\\"shape\\":[");
  for (i = 0; i < ndims; i++) {{
    if (i > 0) printf(",");
    printf("%d", shp[[i]]);
  }}
  printf("],\\"data\\":[");
  for (i = 0; i < prod(shp); i++) {{
    if (i > 0) printf(",");
    printf("%d", flat[[i]]);
  }}
  printf("]}}\\n");

  return 0;
}}
"""


def generate_tensor_sac(preset: str, matrix: list[list[int]]) -> str:
    """Generate SaC code that applies a tensor comprehension and outputs JSON 2D array."""
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    flat = [v for row in matrix for v in row]
    data_str = ",".join(str(v) for v in flat)

    if preset == "transpose":
        comp = f"""  result = with {{
    ([0,0] <= [i,j] < [{rows},{cols}]): A[j,i];
  }} : genarray([{cols},{rows}], 0);"""
        out_rows, out_cols = cols, rows
    elif preset == "diagonal":
        n = min(rows, cols)
        comp = f"""  diag = with {{
    ([0] <= [i] < [{n}]): A[i,i];
  }} : genarray([{n}], 0);"""
        # Output as 1-row matrix
        out_rows, out_cols = 1, n
    elif preset == "rowsum":
        # Sum each row by explicit addition
        sum_expr = " + ".join(f"A[i,{j}]" for j in range(cols))
        comp = f"""  sums = with {{
    ([0] <= [i] < [{rows}]): {sum_expr};
  }} : genarray([{rows}], 0);"""
        out_rows, out_cols = 1, rows
    elif preset == "scale":
        comp = f"""  result = with {{
    ([0,0] <= [i,j] < [{rows},{cols}]): 2 * A[i,j];
  }} : genarray([{rows},{cols}], 0);"""
        out_rows, out_cols = rows, cols
    elif preset == "upper":
        comp = f"""  result = with {{
    ([0,0] <= [i,j] < [{rows},{cols}]): (j >= i ? A[i,j] : 0);
  }} : genarray([{rows},{cols}], 0);"""
        out_rows, out_cols = rows, cols
    else:
        comp = "  result = A;"
        out_rows, out_cols = rows, cols

    # For 1D results (diagonal/rowsum), output differently
    if preset in ("diagonal", "rowsum"):
        var_name = "diag" if preset == "diagonal" else "sums"
        return f"""use Array: all;
use StdIO: all;

int main()
{{
  A = reshape([{rows},{cols}], [{data_str}]);

{comp}

  printf("{{\\\"matrix\\\":[[");
  for (i = 0; i < {out_cols}; i++) {{
    if (i > 0) printf(",");
    printf("%d", {var_name}[[i]]);
  }}
  printf("]]}}\\n");

  return 0;
}}
"""

    return f"""use Array: all;
use StdIO: all;

int main()
{{
  A = reshape([{rows},{cols}], [{data_str}]);

{comp}

  printf("{{\\\"matrix\\\":[");
  for (i = 0; i < {out_rows}; i++) {{
    if (i > 0) printf(",");
    printf("[");
    for (j = 0; j < {out_cols}; j++) {{
      if (j > 0) printf(",");
      printf("%d", result[i,j]);
    }}
    printf("]");
  }}
  printf("]}}\\n");

  return 0;
}}
"""


def generate_stencil_sac(kernel_name: str, width: int, height: int) -> str:
    """Generate SaC code that creates a procedural image, applies a kernel, outputs RGB JSON."""
    kernels = {
        "blur":    ([1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0], 16.0),
        "sharpen": ([0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0], 1.0),
        "edge":    ([-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0], 1.0),
        "emboss":  ([-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 1.0),
    }

    k_vals, k_div = kernels.get(kernel_name, kernels["blur"])
    k_str = ",".join(f"{v:.1f}" for v in k_vals)

    return f"""use Array: all;
use StdIO: all;
use Math: all;

int main()
{{
  W = {width};
  H = {height};

  /* Generate procedural image: gradient circles (same as JS demo) */
  src_r = with {{
    ([0,0] <= [y,x] < [H,W]) {{
      cx1 = tod(W) * 0.3;
      cy1 = tod(H) * 0.3;
      cx2 = tod(W) * 0.7;
      cy2 = tod(H) * 0.6;
      dx1 = tod(x) - cx1;
      dy1 = tod(y) - cy1;
      d1 = sqrt(dx1*dx1 + dy1*dy1);
      dx2 = tod(x) - cx2;
      dy2 = tod(y) - cy2;
      d2 = sqrt(dx2*dx2 + dy2*dy2);
      r = 128.0 + 127.0 * sin(d1 * 0.05);
      inC1 = d1 < 60.0;
      inC2 = d2 < 50.0;
      inRect = x > 100 && x < 200 && y > 120 && y < 180;
    }} : (inC1 ? 220.0 : (inC2 ? 80.0 : (inRect ? 255.0 : r)));
  }} : genarray([H,W], 0.0);

  src_g = with {{
    ([0,0] <= [y,x] < [H,W]) {{
      cx2 = tod(W) * 0.7;
      cy2 = tod(H) * 0.6;
      cx1 = tod(W) * 0.3;
      cy1 = tod(H) * 0.3;
      dx1 = tod(x) - cx1;
      dy1 = tod(y) - cy1;
      d1 = sqrt(dx1*dx1 + dy1*dy1);
      dx2 = tod(x) - cx2;
      dy2 = tod(y) - cy2;
      d2 = sqrt(dx2*dx2 + dy2*dy2);
      g = 128.0 + 127.0 * sin(d2 * 0.06 + 2.0);
      inC1 = d1 < 60.0;
      inC2 = d2 < 50.0;
      inRect = x > 100 && x < 200 && y > 120 && y < 180;
    }} : (inC1 ? 80.0 : (inC2 ? 200.0 : (inRect ? 220.0 : g)));
  }} : genarray([H,W], 0.0);

  src_b = with {{
    ([0,0] <= [y,x] < [H,W]) {{
      cx1 = tod(W) * 0.3;
      cy1 = tod(H) * 0.3;
      cx2 = tod(W) * 0.7;
      cy2 = tod(H) * 0.6;
      dx1 = tod(x) - cx1;
      dy1 = tod(y) - cy1;
      d1 = sqrt(dx1*dx1 + dy1*dy1);
      dx2 = tod(x) - cx2;
      dy2 = tod(y) - cy2;
      d2 = sqrt(dx2*dx2 + dy2*dy2);
      b = 128.0 + 127.0 * sin((d1 + d2) * 0.03 + 4.0);
      inC1 = d1 < 60.0;
      inC2 = d2 < 50.0;
      inRect = x > 100 && x < 200 && y > 120 && y < 180;
    }} : (inC1 ? 160.0 : (inC2 ? 220.0 : (inRect ? 100.0 : b)));
  }} : genarray([H,W], 0.0);

  /* Define kernel */
  kernel = reshape([3,3], [{k_str}]) / {k_div:.1f};

  /* Apply convolution */
  dst_r = with {{
    ([1,1] <= [y,x] < [H-1, W-1]) {{
      s = 0.0;
      for (dy = 0; dy < 3; dy++) {{
        for (dx = 0; dx < 3; dx++) {{
          s = s + src_r[y+dy-1, x+dx-1] * kernel[dy,dx];
        }}
      }}
    }} : max(0.0, min(255.0, s));
  }} : genarray([H,W], 0.0);

  dst_g = with {{
    ([1,1] <= [y,x] < [H-1, W-1]) {{
      s = 0.0;
      for (dy = 0; dy < 3; dy++) {{
        for (dx = 0; dx < 3; dx++) {{
          s = s + src_g[y+dy-1, x+dx-1] * kernel[dy,dx];
        }}
      }}
    }} : max(0.0, min(255.0, s));
  }} : genarray([H,W], 0.0);

  dst_b = with {{
    ([1,1] <= [y,x] < [H-1, W-1]) {{
      s = 0.0;
      for (dy = 0; dy < 3; dy++) {{
        for (dx = 0; dx < 3; dx++) {{
          s = s + src_b[y+dy-1, x+dx-1] * kernel[dy,dx];
        }}
      }}
    }} : max(0.0, min(255.0, s));
  }} : genarray([H,W], 0.0);

  /* Output JSON: source and filtered RGB interleaved */
  printf("{{\\\"width\\\":%d,\\\"height\\\":%d,\\\"source_rgb\\\":[", W, H);
  for (y = 0; y < H; y++) {{
    for (x = 0; x < W; x++) {{
      idx = y * W + x;
      if (idx > 0) printf(",");
      printf("%d,%d,%d", toi(max(0.0, min(255.0, src_r[y,x]))),
                         toi(max(0.0, min(255.0, src_g[y,x]))),
                         toi(max(0.0, min(255.0, src_b[y,x]))));
    }}
  }}
  printf("],\\\"filtered_rgb\\\":[");
  for (y = 0; y < H; y++) {{
    for (x = 0; x < W; x++) {{
      idx = y * W + x;
      if (idx > 0) printf(",");
      printf("%d,%d,%d", toi(dst_r[y,x]), toi(dst_g[y,x]), toi(dst_b[y,x]));
    }}
  }}
  printf("]}}\\n");

  return 0;
}}
"""


def generate_nbody_sac(
    n_particles: int, n_steps: int,
    positions: list[list[float]], velocities: list[list[float]],
    masses: list[float], dt: float, g: float, softening: float,
) -> str:
    """Generate SaC code for N-body simulation that outputs JSON frames."""
    n = n_particles

    # Build position array literal
    pos_vals = []
    for p in positions:
        pos_vals.extend([p[0], p[1]])
    pos_str = ",".join(f"{v:.6f}" for v in pos_vals)

    vel_vals = []
    for v in velocities:
        vel_vals.extend([v[0], v[1]])
    vel_str = ",".join(f"{v:.6f}" for v in vel_vals)

    mass_str = ",".join(f"{m:.6f}" for m in masses)

    return f"""use Array: all;
use StdIO: all;
use Math: all;

int main()
{{
  N = {n};
  G = {g:.6f};
  DT = {dt:.6f};
  SOFTENING = {softening:.6f};
  NSTEPS = {n_steps};

  pos = reshape([N, 2], [{pos_str}]);
  vel = reshape([N, 2], [{vel_str}]);
  mass = [{mass_str}];

  printf("{{\\\"frames\\\":[");

  for (t = 0; t < NSTEPS; t++) {{
    /* Output current positions */
    if (t > 0) printf(",");
    printf("{{\\\"positions\\\":[");
    for (i = 0; i < N; i++) {{
      if (i > 0) printf(",");
      printf("[%.4f,%.4f]", pos[i,0], pos[i,1]);
    }}
    printf("]}}");

    /* Compute accelerations */
    acc = with {{
      ([0,0] <= [i,k] < [N, 2]) {{
        a = 0.0;
        for (j = 0; j < N; j++) {{
          if (j != i) {{
            ddx = pos[j,0] - pos[i,0];
            ddy = pos[j,1] - pos[i,1];
            distSq = ddx*ddx + ddy*ddy + SOFTENING;
            dist = sqrt(distSq);
            force = G * mass[[j]] / (distSq * dist);
            if (k == 0) {{
              a = a + ddx * force;
            }} else {{
              a = a + ddy * force;
            }}
          }}
        }}
      }} : a;
    }} : genarray([N, 2], 0.0);

    vel = with {{
      ([0,0] <= [i,j] < [N, 2]):
        vel[i,j] + acc[i,j] * DT;
    }} : genarray([N, 2], 0.0);

    /* Damping */
    vel = with {{
      ([0,0] <= [i,j] < [N, 2]):
        vel[i,j] * 0.999;
    }} : genarray([N, 2], 0.0);

    pos = with {{
      ([0,0] <= [i,j] < [N, 2]):
        pos[i,j] + vel[i,j];
    }} : genarray([N, 2], 0.0);

    /* Soft boundary */
    pos = with {{
      ([0,0] <= [i,j] < [N, 2]) {{
        cx = 300.0;
        cy = 300.0;
        maxR = 280.0;
        ddx = pos[i,0] - cx;
        ddy = pos[i,1] - cy;
        d = sqrt(ddx*ddx + ddy*ddy);
        newval = pos[i,j];
        if (d > maxR) {{
          if (j == 0) {{
            newval = cx + ddx / d * maxR;
          }} else {{
            newval = cy + ddy / d * maxR;
          }}
        }}
      }} : newval;
    }} : genarray([N, 2], 0.0);
  }}

  printf("]}}\\n");

  return 0;
}}
"""


# ---------- Compute Handlers ----------

async def handle_array_op(params: dict) -> ComputeResponse:
    op = params.get("op", "transpose")
    shape = params.get("shape", [4, 4])
    data = params.get("data", list(range(1, 17)))
    sac_code = generate_array_op_sac(op, shape, data)
    result = await compile_and_run(sac_code, "seq", 1)
    total_ms = result.compile_time_ms + result.run_time_ms
    if not result.success:
        return ComputeResponse(success=False, error=result.error or result.stderr, compute_time_ms=total_ms)
    try:
        parsed = json.loads(result.stdout.strip())
        return ComputeResponse(success=True, result=parsed, compute_time_ms=total_ms)
    except json.JSONDecodeError as e:
        return ComputeResponse(success=False, error=f"Parse error: {e}\nOutput: {result.stdout[:500]}", compute_time_ms=total_ms)


async def handle_tensor(params: dict) -> ComputeResponse:
    preset = params.get("preset", "transpose")
    matrix = params.get("matrix", [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    sac_code = generate_tensor_sac(preset, matrix)
    result = await compile_and_run(sac_code, "seq", 1)
    total_ms = result.compile_time_ms + result.run_time_ms
    if not result.success:
        return ComputeResponse(success=False, error=result.error or result.stderr, compute_time_ms=total_ms)
    try:
        parsed = json.loads(result.stdout.strip())
        return ComputeResponse(success=True, result=parsed, compute_time_ms=total_ms)
    except json.JSONDecodeError as e:
        return ComputeResponse(success=False, error=f"Parse error: {e}\nOutput: {result.stdout[:500]}", compute_time_ms=total_ms)


async def handle_stencil(params: dict) -> ComputeResponse:
    kernel_name = params.get("kernel", "blur")
    width = params.get("width", 280)
    height = params.get("height", 280)
    sac_code = generate_stencil_sac(kernel_name, width, height)
    result = await compile_and_run(sac_code, "seq", 1)
    total_ms = result.compile_time_ms + result.run_time_ms
    if not result.success:
        return ComputeResponse(success=False, error=result.error or result.stderr, compute_time_ms=total_ms)
    try:
        parsed = json.loads(result.stdout.strip())
        return ComputeResponse(success=True, result=parsed, compute_time_ms=total_ms)
    except json.JSONDecodeError as e:
        return ComputeResponse(success=False, error=f"Parse error: {e}\nOutput: {result.stdout[:200]}", compute_time_ms=total_ms)


async def handle_nbody(params: dict) -> ComputeResponse:
    n_particles = params.get("n_particles", 200)
    n_steps = params.get("n_steps", 120)
    positions = params.get("positions", [])
    velocities = params.get("velocities", [])
    masses = params.get("masses", [])
    dt = params.get("dt", 0.016)
    g = params.get("g", 0.5)
    softening = params.get("softening", 4.0)

    if not positions or not velocities or not masses:
        return ComputeResponse(success=False, error="Missing particle data (positions, velocities, masses)")
    if len(positions) != n_particles or len(velocities) != n_particles or len(masses) != n_particles:
        return ComputeResponse(success=False, error="Particle data length mismatch")

    # Limit to avoid huge compilations
    if n_particles > 300:
        return ComputeResponse(success=False, error="Too many particles (max 300)")
    if n_steps > 200:
        return ComputeResponse(success=False, error="Too many steps (max 200)")

    sac_code = generate_nbody_sac(n_particles, n_steps, positions, velocities, masses, dt, g, softening)
    result = await compile_and_run(sac_code, "seq", 1)
    total_ms = result.compile_time_ms + result.run_time_ms
    if not result.success:
        return ComputeResponse(success=False, error=result.error or result.stderr, compute_time_ms=total_ms)
    try:
        parsed = json.loads(result.stdout.strip())
        return ComputeResponse(success=True, result=parsed, compute_time_ms=total_ms)
    except json.JSONDecodeError as e:
        return ComputeResponse(success=False, error=f"Parse error: {e}\nOutput: {result.stdout[:200]}", compute_time_ms=total_ms)


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


@app.post("/compute", response_model=ComputeResponse)
async def compute(req: ComputeRequest):
    handlers = {
        "array_op": handle_array_op,
        "tensor": handle_tensor,
        "stencil": handle_stencil,
        "nbody": handle_nbody,
    }
    handler = handlers.get(req.operation)
    if not handler:
        return ComputeResponse(success=False, error=f"Unknown operation: {req.operation}")
    return await handler(req.params)


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
