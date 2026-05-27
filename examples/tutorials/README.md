# Tutorials

This folder contains notebook-style examples for understanding and demonstrating
the repository.

## Proposed Framework Walkthrough

Open `packing_demo.ipynb` in JupyterLab. It walks through environment creation,
policy loading, policy packing, MCTS-based rearrangement, execution-order
optimization, and interactive Plotly replays.

## Clearance / Buffer Space Demo

Open `clearance_demo.ipynb` in JupyterLab. It compares packing with and without
`buffer_space`, shows true versus virtual item dimensions, and uses interactive
replays to explain how clearance changes the packed sequence.

## Interactive Simulator

Open `interactive_simulator_demo.ipynb` in JupyterLab, or run the simulator
directly from the project root:

```bash
python -m interactive_simulator_app
```

Then open:

```text
http://127.0.0.1:8765
```

The simulator shows a Three.js 3D container view with a Plotly buffer panel.
The current item is the head of the buffer. In `Anchors` mode, click a green
anchor marker to place it. In `Grid` mode, hover over 30 mm candidate positions
to preview the item with height-map-resolved `z`; solid previews are
stable/placeable, transparent red previews are rejected. Use the `Support`
button to show or hide the bearable support patches recorded by the stability
checker.

Implementation files live in `../../interactive_simulator_app/`.
