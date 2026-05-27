# Cascaded Block Selector Experiment

## Baseline

Train the largest-block baseline under the same vertical-stack/simple-block
conditions used by the cascaded selector:

```bash
python train.py --policy-mode largest_block_baseline --stack-only --use-simple-blocks --output-name largest-block-baseline
```

## Cascaded Selector

Train the cascaded selector with matching environment settings:

```bash
python train.py --policy-mode cascaded_block_selector --stack-only --use-simple-blocks --output-name cascaded-block-selector
```

## Evaluation

Evaluate a cascaded checkpoint with the matching cascaded policy mode:

```bash
python test.py --policy-mode cascaded_block_selector --checkpoint outputs/train_outputs/cascaded-block-selector/policy_step.pth --stack-only --use-simple-blocks --no-use-mcts
```

When evaluating a checkpoint, use the same policy mode that produced it. A
baseline checkpoint should be evaluated with `largest_block_baseline`, and a
cascaded checkpoint should be evaluated with `cascaded_block_selector`.

Compare the runs with matched item distributions, seeds, container settings,
buffer size, stack-only mode, and simple-block settings where practical. Track:

- Final utilization.
- Blocked step.
- Packed source boxes.
- Selected stack height distribution.
- Inference time per packing decision.

The first cascaded implementation intentionally rejects cascaded evaluation
with MCTS. Use `--no-use-mcts` for cascaded evaluation until
cascaded-compatible MCTS is designed.
