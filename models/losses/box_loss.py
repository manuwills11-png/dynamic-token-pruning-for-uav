import torch
import torch.nn as nn


class DIoULoss(nn.Module):
    """
    Distance-IoU loss.

    Boxes are expected in:
        [x1, y1, x2, y2]
    """

    def __init__(self, eps=1e-7, reduction="mean"):
        super().__init__()

        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes:
                [N, 4]

            target_boxes:
                [N, 4]

        Returns:
            Scalar DIoU loss.
        """

        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]

        target_x1 = target_boxes[:, 0]
        target_y1 = target_boxes[:, 1]
        target_x2 = target_boxes[:, 2]
        target_y2 = target_boxes[:, 3]

        # Intersection.
        inter_x1 = torch.maximum(
            pred_x1,
            target_x1,
        )

        inter_y1 = torch.maximum(
            pred_y1,
            target_y1,
        )

        inter_x2 = torch.minimum(
            pred_x2,
            target_x2,
        )

        inter_y2 = torch.minimum(
            pred_y2,
            target_y2,
        )

        inter_w = (
            inter_x2 - inter_x1
        ).clamp(min=0)

        inter_h = (
            inter_y2 - inter_y1
        ).clamp(min=0)

        intersection = inter_w * inter_h

        # Areas.
        pred_w = (
            pred_x2 - pred_x1
        ).clamp(min=0)

        pred_h = (
            pred_y2 - pred_y1
        ).clamp(min=0)

        target_w = (
            target_x2 - target_x1
        ).clamp(min=0)

        target_h = (
            target_y2 - target_y1
        ).clamp(min=0)

        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        union = (
            pred_area
            + target_area
            - intersection
        ).clamp(min=self.eps)

        iou = intersection / union

        # Centers.
        pred_cx = (
            pred_x1 + pred_x2
        ) / 2.0

        pred_cy = (
            pred_y1 + pred_y2
        ) / 2.0

        target_cx = (
            target_x1 + target_x2
        ) / 2.0

        target_cy = (
            target_y1 + target_y2
        ) / 2.0

        center_distance = (
            (pred_cx - target_cx).pow(2)
            + (pred_cy - target_cy).pow(2)
        )

        # Smallest enclosing box.
        enclosing_x1 = torch.minimum(
            pred_x1,
            target_x1,
        )

        enclosing_y1 = torch.minimum(
            pred_y1,
            target_y1,
        )

        enclosing_x2 = torch.maximum(
            pred_x2,
            target_x2,
        )

        enclosing_y2 = torch.maximum(
            pred_y2,
            target_y2,
        )

        enclosing_diagonal = (
            (enclosing_x2 - enclosing_x1).pow(2)
            + (enclosing_y2 - enclosing_y1).pow(2)
        ).clamp(min=self.eps)

        diou = (
            1.0
            - iou
            + center_distance / enclosing_diagonal
        )

        if self.reduction == "sum":
            return diou.sum()

        if self.reduction == "none":
            return diou

        return diou.mean()