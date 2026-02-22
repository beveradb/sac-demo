# SAC Interactive Demo

An interactive demo of [SAC (Single-Assignment C)](https://www.sac-home.org), a functional array programming language for high-performance computing.

**Live demo:** [sacdemo.beveradb.com](https://sacdemo.beveradb.com)

## What's in the demo

The page walks through SAC's key features with interactive visualizations:

- **Array Operations** — rank-polymorphic transpose, reverse, rotate, take, drop on 1D/2D/3D arrays
- **Tensor Comprehensions** — with-loop expressions for matrix transforms (transpose, diagonal, scaling, triangular extraction, row sums)
- **Stencil Convolution** — image filters (blur, sharpen, edge detect, emboss) using 3x3 kernels
- **Parallelization** — Mandelbrot benchmark comparing sequential vs multi-threaded execution
- **N-Body Simulation** — 200-particle gravity simulation with sequential vs parallel views

## Quick start

### Just the demo page (no compiler)

Open `index.html` in a browser. Everything works standalone with JavaScript simulations — no server needed.

```bash
# or serve it locally:
python3 -m http.server 8080
# then open http://localhost:8080
```

### With the real SAC compiler (Docker)

Running the companion server connects the demo to a real `sac2c` compiler. The page auto-detects it and adds "Run in SAC" buttons to every code block.

**Option A — Docker Compose (recommended):**

```bash
docker compose up
```

**Option B — Docker run:**

```bash
docker run --rm -p 7227:7227 --memory=512m --cpus=4 beveradb/sac-demo-companion
```

**Option C — Build locally:**

```bash
docker build -t beveradb/sac-demo-companion ./companion
docker run --rm -p 7227:7227 --memory=512m --cpus=4 beveradb/sac-demo-companion
```

Then open `index.html` (or serve it on any port). When the companion is running:

1. A green **"SAC Runtime Connected"** indicator appears in the bottom-right corner
2. **"Run in SAC"** buttons appear on every code block
3. Code blocks become editable — modify the SAC code and run your changes
4. The parallelization section gets a **"Run Real SAC Benchmark"** button that shows actual seq vs mt_pth timing

When you stop the companion, the page gracefully falls back to JS-only mode. No errors, no broken state.

## How it works

```
Browser (index.html)                    Docker (localhost:7227)
                                         Python FastAPI server
  JS visualizations always work          sac2c compiler + stdlib
                                         Pre-compiled demo binaries
  On load: fetch /health ──────────────>
  If OK: show indicator, add buttons
  "Run in SAC" ── POST /compile-run ───>  Compile + run, return output
  Benchmark ───── POST /benchmark ─────>  seq vs mt_pth real timing
```

The page is a single self-contained `index.html` with no build step. The companion server is optional — it provides real compilation but the demo is fully functional without it.

### Companion server API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{ status, sac2c_version, precompiled[] }` |
| `/compile-run` | POST | Accepts `{ code, target, threads }`, compiles and runs SAC code |
| `/benchmark` | POST | Runs Mandelbrot with seq and mt_pth at 1/2/4/8 threads |

### Demo programs

Pre-written SAC programs in `companion/programs/`:

| File | Description |
|------|-------------|
| `array_ops_demo.sac` | stdlib transpose, reverse, rotate, take, drop |
| `tensor_demo.sac` | with-loop comprehensions: transpose, diagonal, scale, upper triangle, row sums |
| `stencil_demo.sac` | 3x3 Gaussian blur and Laplacian edge detection |
| `mandelbrot_demo.sac` | ASCII Mandelbrot set visualization |
| `nbody_demo.sac` | All-pairs gravitational N-body simulation |
| `parallel_bench.sac` | Mandelbrot for sequential vs multi-threaded benchmarking |

These are pre-compiled at Docker build time for instant execution. User-edited code compiles on demand (~1-5 seconds).

## Project structure

```
.
├── index.html              # The entire demo (HTML + CSS + JS, self-contained)
├── docker-compose.yml      # One-command local setup
├── CNAME                   # GitHub Pages custom domain
├── companion/
│   ├── Dockerfile          # Based on sacbase/sac-compiler:main
│   ├── server.py           # FastAPI server (3 endpoints)
│   ├── entrypoint.sh       # Container startup script
│   ├── requirements.txt    # Python dependencies (fastapi, uvicorn)
│   └── programs/           # Pre-written SAC demo programs
│       ├── array_ops_demo.sac
│       ├── tensor_demo.sac
│       ├── stencil_demo.sac
│       ├── mandelbrot_demo.sac
│       ├── nbody_demo.sac
│       └── parallel_bench.sac
└── architecture.html       # Separate architecture overview page
```

## Requirements

- **Demo only:** Any modern browser
- **With compiler:** Docker (supports both ARM64/Apple Silicon and x86_64)

## Notes

- The companion uses port **7227** (unlikely to conflict with common services)
- Browsers allow `http://localhost` from HTTPS pages as a "potentially trustworthy origin", so the live site at `sacdemo.beveradb.com` can connect to a local companion
- Compile timeout: 30s, run timeout: 10s
- The Docker image is ~2.5GB because it includes the full sac2c compiler and standard library
