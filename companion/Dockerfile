FROM sacbase/sac-compiler:main

USER root

# Install Python and pip (sac-compiler is Debian-based, minimal image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Install FastAPI and uvicorn
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Copy server and programs
WORKDIR /app
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
COPY programs/ /app/programs/
RUN chmod +x /app/entrypoint.sh

# Create precompiled directory and attempt to pre-compile demo programs.
# These may fail if sac2c or stdlib aren't fully available at build time,
# so we use || true to make failures non-fatal - the server will compile
# on demand as a fallback.
RUN mkdir -p /app/precompiled && \
    for prog in mandelbrot_demo array_ops_demo tensor_demo stencil_demo nbody_demo; do \
        sac2c -t seq -o /app/precompiled/${prog}_seq /app/programs/${prog}.sac 2>/dev/null || true; \
    done && \
    sac2c -t seq -o /app/precompiled/parallel_bench_seq /app/programs/parallel_bench.sac 2>/dev/null || true && \
    sac2c -t mt_pth -o /app/precompiled/parallel_bench_mt_pth /app/programs/parallel_bench.sac 2>/dev/null || true

EXPOSE 7227

ENTRYPOINT ["/app/entrypoint.sh"]
