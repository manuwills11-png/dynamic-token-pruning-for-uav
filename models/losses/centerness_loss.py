import torch.nn as nn


class CenternessLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss(
            reduction=reduction,
        )

    def forward(self, logits, targets):
        """
        Args:
            logits:  model centerness logits
            targets: centerness targets in [0, 1]

        Returns:
            BCE loss, reduced according to `reduction`.
        """

        return self.loss(
            logits,
            targets,
        )