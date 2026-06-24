# Python CoppeliaSim Controller

This controller drives the CoppeliaSim scene through the ZeroMQ remote API.

## Scene Contract

Keep only tiny loader scripts embedded in the scene. The real Lua scripts live
beside the `.ttt` files under `coppeliaSim/` and are loaded when simulation
starts.

Current external script files:

- `coppeliaSim/belt.lua`
- `coppeliaSim/external_api.lua`
- `coppeliaSim/loading_orchastrator.lua`
- `coppeliaSim/collision_free_trajectory.lua`
- `coppeliaSim/ur10_joint.lua`
- `coppeliaSim/ur10_linear.lua`

The scene script publishes tracked belt objects through `trackedItemsData`.
The controller waits until the belt is full, sets `robotPicking=1`, randomly
picks one tracked Cuboid, parents it to the robot tip while carrying it, places
it at `(0, 0, 1)`, releases it back to the world, then sets `robotPicking=0`.

## Run

Start CoppeliaSim in true headless mode to avoid the Qt GUI crash path:

```bash
cd /home/gao/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04
./coppeliaSim.sh -H -GzmqRemoteApi.rpcPort=23000
```

The `coppeliaSim/belt.lua` in this workspace treats `simUI` as optional, so it can
run headless without the item table UI. If headless startup still reports
`simUI` errors, update the script attached to the scene from this file and
remove any remaining UI/customization scripts; embedded scene scripts should be
tiny loaders that call the matching external Lua files beside the scene.

In a second terminal, run the controller from the integrations workspace.

Normal controller run:

```bash
cd /home/gao/online-3d-bin-packing-stability-repacking/integrations
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m python_controller.main --port 23000 --place 0 0 1
```

Bounded smoke run:

```bash
cd /home/gao/online-3d-bin-packing-stability-repacking/integrations
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m python_controller.main --port 23000 --place 0 0 1 --max-cycles 1
```
