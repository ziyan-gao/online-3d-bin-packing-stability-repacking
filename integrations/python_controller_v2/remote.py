from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable


@dataclass
class RemoteContext:
    client: object
    sim: object
    simIK: object

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self.client.step()


def connect(port: int = 23000) -> RemoteContext:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient(port=port)
    sim = client.require("sim")
    simIK = client.require("simIK")
    client.setStepping(True)
    return RemoteContext(client=client, sim=sim, simIK=simIK)


def first_object(sim: object, aliases: Iterable[str]) -> int:
    tried = []
    for alias in aliases:
        tried.append(alias)
        try:
            handle = sim.getObject(alias, {"noError": True})
        except Exception:
            handle = -1
        if handle is not None and handle >= 0:
            return int(handle)
    raise RuntimeError("Could not find object. Tried: " + ", ".join(tried))


def ensure_started(ctx: RemoteContext, scene_path: str | None = None) -> None:
    sim = ctx.sim
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        start = time.time()
        while sim.getSimulationState() != sim.simulation_stopped:
            if time.time() - start > 10:
                raise TimeoutError("Timed out waiting for simulation to stop")
            time.sleep(0.05)
    if scene_path:
        sim.loadScene(scene_path)
    sim.startSimulation()

