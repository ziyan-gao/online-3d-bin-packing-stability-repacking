# Docker Usage

Docker is optional. Use it when you want a reproducible project environment for
the demo, notebook, validation runs, or GPU training without tuning local Python,
PyTorch, SQLite, Jupyter, and Tianshou installs by hand.

## Files

- `docker-compose.yml`: Compose services for shell, validation, demo, notebook,
  and GPU training.
- `Dockerfile`: Project image definition.
- `Dockerfile.dockerignore`: Build-context exclusions for the Dockerfile.
- `requirements_docker.txt`: Python dependencies used only by the Docker image.
- `setup_nvidia_docker_and_compose.sh`: Ubuntu/Debian helper for NVIDIA
  Container Toolkit setup and the project GPU smoke check.

## What Docker Provides

- Python 3.11 with the project dependency stack.
- CUDA-capable PyTorch wheels; GPU access is enabled only for the `train-gpu`
  compose service.
- A pinned Tianshou source revision for reproducible behavior.
- System packages needed by notebook and visualization workflows.
- Bind-mounted source code, so outputs and edited files stay in this checkout.
- A read-only bind mount for the CJ policy checkpoint directory. By default this
  is `/home/gao/CJ_block_dockerize/outputs/model_cj/CJ`; override it with
  `CJ_POLICY_CHECKPOINT_DIR` if your checkpoint lives elsewhere.

## Build

Build the default image:

```bash
docker compose -f docker/docker-compose.yml build
```

Build a versioned image:

```bash
export IMAGE_TAG=v2026.05.27
docker compose -f docker/docker-compose.yml build
```

You can override the pinned Tianshou revision at build time:

```bash
docker compose -f docker/docker-compose.yml build --build-arg TIANSHOU_REF=<commit>
```

## Shell

Open an interactive container:

```bash
docker compose -f docker/docker-compose.yml run --rm shell
```

From there, run normal project commands such as:

```bash
python test.py --config configs/test_default.yaml
python train.py --config configs/train_default.yaml
```

## Validation

Run the default validation command:

```bash
docker compose -f docker/docker-compose.yml run --rm test
```

## Demo Replay

Generate a replay/demo artifact under `outputs/three_live/demo/`:

```bash
docker compose -f docker/docker-compose.yml run --rm demo
```

The demo uses `configs/test_cj_default.yaml` and loads the policy checkpoint
from the read-only CJ checkpoint mount.

## Replay Viewer

Serve generated replay HTML files from the repository root:

```bash
docker compose -f docker/docker-compose.yml up replay
```

Open the generated demo replay at:

```text
http://127.0.0.1:8090/outputs/three_live/demo/run_seed_101_mcts_false_optimize_false.html
```

Set `REPLAY_HOST_PORT` if port `8090` is already in use.

## Notebook

Start JupyterLab on the packing tutorial:

```bash
docker compose -f docker/docker-compose.yml up notebook
```

Open:

```text
http://localhost:8890
```

The notebook service starts without a token for local demo convenience. Do not
expose it on an untrusted network.

## GPU Training

GPU training requires NVIDIA Container Toolkit on the host.

For Ubuntu/Debian hosts, the helper script can install and configure the
toolkit:

```bash
sudo bash docker/setup_nvidia_docker_and_compose.sh
```

To install/configure and then run the project GPU smoke check:

```bash
sudo bash docker/setup_nvidia_docker_and_compose.sh --gpu-check
```

Train the block baseline policy:

```bash
docker compose -f docker/docker-compose.yml --profile gpu up train-baseline-gpu
```

This writes checkpoints under:

```text
outputs/train_outputs/baseline-blocks/
```

Train the cascaded block selector policy:

```bash
docker compose -f docker/docker-compose.yml --profile gpu up train-cascaded-gpu
```

This writes checkpoints under:

```text
outputs/train_outputs/cascaded-block-selector/
```

Both services use vertical SimpleBlock candidates with the same buffer settings
from `configs/train_default.yaml`. The baseline service keeps the largest usable
block policy, while the cascaded service trains the new block selector policy.

TensorBoard is exposed at `http://localhost:16006` for baseline training and
`http://localhost:16007` for cascaded training. The older `train-gpu` service is
kept as a baseline-compatible alias.

Quick GPU check:

```bash
docker compose -f docker/docker-compose.yml --profile gpu run --rm train-gpu \
  python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

## Notes

- Compose uses `IMAGE_TAG`; default is `latest`.
- Live visualization ports default to `8765` through `8769`, depending on the
  service.
- Project checkpoints, datasets, logs, and generated artifacts are excluded from
  build context by `docker/Dockerfile.dockerignore` to keep builds fast.
