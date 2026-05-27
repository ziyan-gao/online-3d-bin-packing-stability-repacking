# Docker Usage

Docker is optional. Use it when you want a reproducible project environment for
the demo, notebook, validation runs, or GPU training without tuning local Python,
PyTorch, SQLite, Jupyter, and Tianshou installs by hand.

## What Docker Provides

- Python 3.11 with the project dependency stack.
- CUDA-capable PyTorch wheels; GPU access is enabled only for the `train-gpu`
  compose service.
- A pinned Tianshou source revision for reproducible behavior.
- System packages needed by notebook and visualization workflows.
- Bind-mounted source code, so outputs and edited files stay in this checkout.

## Build

Build the default image:

```bash
docker compose build
```

Build a versioned image:

```bash
export IMAGE_TAG=v2026.05.27
docker compose build
```

You can override the pinned Tianshou revision at build time:

```bash
docker compose build --build-arg TIANSHOU_REF=<commit>
```

## Shell

Open an interactive container:

```bash
docker compose run --rm shell
```

From there, run normal project commands such as:

```bash
python test.py --config configs/test_default.yaml
python train.py --config configs/train_default.yaml
```

## Validation

Run the default validation command:

```bash
docker compose run --rm test
```

## Demo Replay

Generate a replay/demo artifact under `_three_live/demo/`:

```bash
docker compose run --rm demo
```

## Notebook

Start JupyterLab on the packing tutorial:

```bash
docker compose up notebook
```

Open:

```text
http://localhost:8890
```

The notebook service starts without a token for local demo convenience. Do not
expose it on an untrusted network.

## GPU Training

GPU training requires NVIDIA Container Toolkit on the host.

Start the GPU training service:

```bash
docker compose --profile gpu up train-gpu
```

TensorBoard is exposed at:

```text
http://localhost:16006
```

Quick GPU check:

```bash
docker compose --profile gpu run --rm train-gpu \
  python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

## Notes

- Compose uses `IMAGE_TAG`; default is `latest`.
- Live visualization ports default to `8765` through `8769`, depending on the
  service.
- Project checkpoints, datasets, logs, and generated artifacts are excluded from
  build context by `.dockerignore` to keep builds fast.
