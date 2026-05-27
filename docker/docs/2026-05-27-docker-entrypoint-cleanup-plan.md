# Docker Entrypoint Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scattered Docker workflows with one clear compose entrypoint and matching docs.

**Architecture:** Keep the existing Dockerfile as the image definition. Rewrite the Compose file around shared anchors and five user-facing services. Delete the separate demo compose file and document the command surface in `docker/README.md`.

**Tech Stack:** Docker, Docker Compose, Markdown.

---

### Task 1: Compose File

**Files:**
- Modify: `docker/docker-compose.yml`
- Delete: `docker-compose.demo.yml`

- [ ] Replace the Compose file with shared `x-app-common`, CPU services `shell`, `test`, `demo`, `notebook`, and GPU service `train-gpu`.
- [ ] Remove Portainer and CJ experiment service variants.
- [ ] Delete `docker-compose.demo.yml`.

### Task 2: Docker Documentation

**Files:**
- Modify: `docker/README.md`

- [ ] Rewrite `docker/README.md` around why Docker exists and the five compose services.
- [ ] Remove stale references to `app`, `app-gpu`, `docker-compose.demo.yml`, monitoring, Portainer, and deleted demo configs.

### Task 3: Verification

**Files:**
- No source changes.

- [ ] Run `docker compose -f docker/docker-compose.yml config`.
- [ ] Run `git status --short`.
