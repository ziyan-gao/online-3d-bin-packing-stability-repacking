# Docker Entrypoint Cleanup Design

## Goal

Make Docker easy to understand and useful as a project entrypoint. Docker should answer one question clearly: how do I run the project without hand-configuring Python, CUDA, Tianshou, Jupyter, and supporting system packages?

## Scope

Keep one Dockerfile and one Compose file. Remove the separate demo compose file and the stale Portainer/experiment-service sprawl. Keep Docker focused on common workflows:

- `shell`: interactive development shell.
- `test`: validation/test command.
- `demo`: replay/demo generation command.
- `notebook`: JupyterLab tutorial entrypoint.
- `train-gpu`: GPU training entrypoint.

## Compose Shape

The compose file will use shared anchors for common build, image, working directory, bind mount, and cache environment settings. CPU services will not request GPUs. The GPU service will be isolated behind the `gpu` profile and set NVIDIA runtime environment variables.

## Documentation

`docker/README.md` explains why Docker exists, then shows the smallest command set for build, shell, test, demo, notebook, and GPU training. It does not mention monitoring, Portainer, deleted demo configs, or old CJ experiment service variants.

## Out Of Scope

Do not change Python training/testing behavior. Do not rebuild or validate the image in this cleanup. Verification is limited to static compose validation because Docker builds may require network access and take significant time.
