import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        reduction="mean",
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:  [N, C]
            targets: [N, C]
                      Values should be 0 or 1.

        Returns:
            Scalar focal loss.
        """

        targets = targets.float()

        prob = torch.sigmoid(logits)

        # Binary cross entropy without reduction.
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # p_t = p when target == 1
        # p_t = 1-p when target == 0
        p_t = (
            prob * targets
            + (1.0 - prob) * (1.0 - targets)
        )

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        focal_weight = (
            alpha_t
            * (1.0 - p_t).pow(self.gamma)
        )

        loss = focal_weight * bce

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "none":
            return loss

        return loss.mean()