# Docker Folder Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move operational Docker files into `docker/` and document the new command surface.

**Architecture:** Keep the project root as Docker build context so the image can still copy the repository. Move Docker definitions under `docker/`, use `docker/Dockerfile.dockerignore` so ignore rules remain active, and invoke compose with `-f docker/docker-compose.yml`.

**Tech Stack:** Docker, Docker Compose, Bash, Markdown.

---

### Task 1: Move Docker Assets

**Files:**
- Move: `Dockerfile` to `docker/Dockerfile`
- Move: `docker-compose.yml` to `docker/docker-compose.yml`
- Move: `.dockerignore` to `docker/Dockerfile.dockerignore`
- Move: `requirements_docker.txt` to `docker/requirements_docker.txt`
- Move: `DOCKER.md` to `docker/README.md`

- [x] Move the files under `docker/`.
- [x] Update `docker/Dockerfile` to copy `docker/requirements_docker.txt`.
- [x] Update `docker/docker-compose.yml` so `build.context` points at `..` and `build.dockerfile` points at `docker/Dockerfile`.

### Task 2: Update References

**Files:**
- Modify: `README.md`
- Modify: `docker/README.md`
- Modify: `docker/setup_nvidia_docker_and_compose.sh`

- [x] Replace root-level Docker file references with the new `docker/` paths.
- [x] Update compose examples to use `docker compose -f docker/docker-compose.yml`.
- [x] Update the NVIDIA setup script to find `docker/docker-compose.yml` and run the GPU smoke check through that file.

### Task 3: Verify

**Files:**
- Validate: `docker/docker-compose.yml`
- Validate: `docker/setup_nvidia_docker_and_compose.sh`

- [x] Run `bash -n docker/setup_nvidia_docker_and_compose.sh`.
- [x] Run `docker compose -f docker/docker-compose.yml config`.
- [x] Run `docker compose -f docker/docker-compose.yml --profile gpu config`.
- [x] Search for stale root-level Docker references.
