from __future__ import annotations

import torch


class CascadedCategoricalMasked(torch.distributions.Categorical):
    def __init__(self, actor_output):
        self.device = actor_output.logits.device
        self.masks = actor_output.action_mask.to(self.device).bool()
        self.batch_size = actor_output.logits.shape[0]
        flat_mask = self.masks.reshape(self.batch_size, -1)

        logits = actor_output.logits.clone().float()
        logits = logits.masked_fill(~flat_mask, -torch.inf)
        all_masked = ~flat_mask.any(dim=1)
        if all_masked.any():
            logits[all_masked, 0] = 0.0
            flat_mask = flat_mask.clone()
            flat_mask[all_masked, 0] = True

        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs * flat_mask.float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-10)

        super().__init__(probs=probs, validate_args=False)
        self.reshaped_masks = flat_mask

    def entropy(self):
        p_log_p = self.probs * torch.log(self.probs.clamp(min=1e-10))
        p_log_p = torch.where(
            self.reshaped_masks,
            p_log_p,
            torch.tensor(0.0, device=self.device),
        )
        return -p_log_p.sum(-1)
