# Docker Usage

## Build

```bash
docker build -t cj-block:latest .
```

Build a new versioned image (and keep `latest` in sync):

```bash
export IMAGE_TAG=v2026.04.01
docker build -t cj-block:${IMAGE_TAG} -t cj-block:latest .
```

## Run (CPU)

```bash
docker run --rm -it -v "$PWD":/app -w /app cj-block:latest
```

Then run your scripts inside the container, for example:

```bash
python train.py
python test.py
```

## Run with Docker Compose

Compose uses `IMAGE_TAG` (defaults to `latest`), so you can choose the version:

```bash
export IMAGE_TAG=v2026.04.01
```

CPU shell:

```bash
docker compose run --rm app
```

GPU shell (requires NVIDIA Container Toolkit):

```bash
docker compose --profile gpu run --rm app-gpu
```

Quick GPU check:

```bash
docker compose --profile gpu run --rm app-gpu \
  python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

## Demo Compose File

`docker-compose.demo.yml` is a CPU-first compose file for quick demonstrations.
It builds the same Docker image, bind-mounts the repo into `/app`, and runs
portable train/test/notebook workflows without requiring NVIDIA Docker.
The demo train/test commands read their smoke settings from
`configs/train_demo.yaml` and `configs/test_demo.yaml`.

Build the demo image:

```bash
docker compose -f docker-compose.demo.yml config
docker compose -f docker-compose.demo.yml build
```

Run the deterministic MCTS validation demo:

```bash
docker compose -f docker-compose.demo.yml run --rm test
```

Generated interactive replays are written under `_plotly_live/`, especially
`_plotly_live/demo_compose/` for this service.

Run a short training smoke test:

```bash
docker compose -f docker-compose.demo.yml run --rm train
```

Start JupyterLab for `tutorials/packing_demo.ipynb`:

```bash
docker compose -f docker-compose.demo.yml up notebook
```

Open:

```text
http://localhost:8888
```

The notebook service starts without a token for local demo convenience. Do not
expose it on an untrusted network.

## Notes

- The image uses Python 3.11 and CUDA-capable PyTorch wheels (`cu121`).
- The image installs Debian `sqlite3`/`libsqlite3` packages and checks
  `import sqlite3` during build, so notebook/kernel tooling does not fail on a
  missing or mismatched SQLite runtime.
- To use CUDA at runtime, run the container with NVIDIA runtime (`--gpus all` or the compose GPU profile).
- `app-gpu` also sets `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
- General dependencies are installed from `requirements_docker.txt` (ROS-free).
- Tianshou is installed from a pinned Git commit during image build for reproducibility.
- `docker-compose.yml` includes a GPU profile (`app-gpu`) if your host supports it.
- Project checkpoints/datasets/log artifacts are excluded from build context by `.dockerignore` to keep builds fast.

- You can override the pinned Tianshou revision at build time with `--build-arg TIANSHOU_REF=<commit>`.

## Monitoring Stack (Grafana + Prometheus + cAdvisor + GPU)

Start container and host metrics (CPU/RAM/network/disk):

```bash
docker compose --profile monitoring up -d prometheus cadvisor grafana
```

If your host has NVIDIA GPUs, also start GPU metrics exporter:

```bash
docker compose --profile monitoring-gpu up -d dcgm-exporter
```

Or start all monitoring services together:

```bash
docker compose --profile monitoring --profile monitoring-gpu up -d
```

Open dashboards/services:

- Grafana: `http://localhost:3000` (default `admin` / `admin`)
- Prometheus: `http://localhost:9090`
- cAdvisor: `http://localhost:8080`
- GPU metrics endpoint: `http://localhost:9400/metrics`

In Grafana, add Prometheus datasource with URL:

```text
http://prometheus:9090
```

Stop monitoring services:

```bash
docker compose --profile monitoring --profile monitoring-gpu down
```
