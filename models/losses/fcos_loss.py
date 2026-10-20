import torch
import torch.nn as nn

from models.losses.focal_loss import SigmoidFocalLoss
from models.losses.box_loss import DIoULoss
from models.losses.centerness_loss import CenternessLoss


class FCOSLoss(nn.Module):
    def __init__(
        self,
        lambda_cls=1.0,
        lambda_loc=2.0,
        lambda_ctr=1.0,
    ):
        super().__init__()

        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.lambda_ctr = lambda_ctr

        self.cls_loss = SigmoidFocalLoss(
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        )

        self.box_loss = DIoULoss(
            reduction="none",
        )

        self.centerness_loss = CenternessLoss(
            reduction="sum",
        )

    def forward(
        self,
        predictions,
        targets,
    ):
        """
        predictions:
            List of 4 FPN-level prediction dictionaries.

            predictions[level]["cls_logits"]:
                [B, C, H, W]

            predictions[level]["bbox_reg"]:
                [B, 4, H, W]

            predictions[level]["centerness"]:
                [B, 1, H, W]

        targets:
            List with one entry per image.

            targets[image][level] is:

                (
                    locations,
                    cls_targets,
                    bbox_targets,
                    centerness_targets,
                )

        Returns:
            total_loss, loss_info
        """

        batch_size = predictions[0]["cls_logits"].shape[0]

        if len(targets) != batch_size:
            raise ValueError(
                f"Expected {batch_size} image targets, "
                f"got {len(targets)}"
            )

        # ---------------------------------------------------------
        # Flatten every level and every image into one flat set of
        # locations, so all three terms can be normalized by the
        # batch-wide positive count the way FCOS defines them.
        # ---------------------------------------------------------

        cls_logits_all = []
        bbox_reg_all = []
        centerness_all = []

        locations_all = []
        cls_targets_all = []
        bbox_targets_all = []
        ctr_targets_all = []

        for level, prediction in enumerate(predictions):

            cls_logits = prediction["cls_logits"]
            bbox_reg = prediction["bbox_reg"]
            centerness = prediction["centerness"]

            B, num_classes, H, W = cls_logits.shape

            cls_logits = cls_logits.permute(
                0, 2, 3, 1
            ).reshape(
                B,
                H * W,
                num_classes,
            )

            bbox_reg = bbox_reg.permute(
                0, 2, 3, 1
            ).reshape(
                B,
                H * W,
                4,
            )

            centerness = centerness.permute(
                0, 2, 3, 1
            ).reshape(
                B,
                H * W,
            )

            for image_idx in range(batch_size):

                (
                    locations,
                    cls_targets,
                    bbox_targets,
                    ctr_targets,
                ) = targets[image_idx][level]

                cls_logits_all.append(
                    cls_logits[image_idx]
                )

                bbox_reg_all.append(
                    bbox_reg[image_idx]
                )

                centerness_all.append(
                    centerness[image_idx]
                )

                locations_all.append(locations)
                cls_targets_all.append(cls_targets)
                bbox_targets_all.append(bbox_targets)
                ctr_targets_all.append(ctr_targets)

        cls_logits_all = torch.cat(cls_logits_all, dim=0)
        bbox_reg_all = torch.cat(bbox_reg_all, dim=0)
        centerness_all = torch.cat(centerness_all, dim=0)

        locations_all = torch.cat(locations_all, dim=0)
        cls_targets_all = torch.cat(cls_targets_all, dim=0)
        bbox_targets_all = torch.cat(bbox_targets_all, dim=0)
        ctr_targets_all = torch.cat(ctr_targets_all, dim=0)

        positive = cls_targets_all >= 0

        num_positive = int(positive.sum())

        positive_indices = positive.nonzero(
            as_tuple=False
        ).squeeze(1)

        # Normalizing by the batch-wide positive count keeps the loss
        # magnitude independent of batch size and FPN level count.
        normalizer = max(num_positive, 1)

        # ---------------------------------------------------------
        # Classification
        #
        # -1 = background, which becomes an all-zero one-hot target.
        # ---------------------------------------------------------

        cls_target_one_hot = torch.zeros_like(
            cls_logits_all
        )

        if num_positive > 0:
            cls_target_one_hot[
                positive_indices,
                cls_targets_all[positive_indices],
            ] = 1.0

        total_cls_loss = self.cls_loss(
            cls_logits_all,
            cls_target_one_hot,
        ) / normalizer

        # ---------------------------------------------------------
        # Regression + centerness
        # ---------------------------------------------------------

        if num_positive > 0:

            positive_locations = locations_all[
                positive_indices
            ]

            pred_boxes = self._decode_boxes(
                positive_locations,
                bbox_reg_all[positive_indices],
            )

            target_boxes = self._decode_boxes(
                positive_locations,
                bbox_targets_all[positive_indices],
            )

            positive_ctr_targets = ctr_targets_all[
                positive_indices
            ]

            diou = self.box_loss(
                pred_boxes,
                target_boxes,
            )

            # FCOS weights regression by the centerness target so
            # locations near a box centre dominate the signal.
            centerness_sum = positive_ctr_targets.sum().clamp(
                min=1e-6
            )

            total_box_loss = (
                diou * positive_ctr_targets
            ).sum() / centerness_sum

            total_ctr_loss = self.centerness_loss(
                centerness_all[positive_indices],
                positive_ctr_targets,
            ) / normalizer

        else:
            total_box_loss = bbox_reg_all.sum() * 0.0
            total_ctr_loss = centerness_all.sum() * 0.0

        total_loss = (
            self.lambda_cls * total_cls_loss
            + self.lambda_loc * total_box_loss
            + self.lambda_ctr * total_ctr_loss
        )

        return total_loss, {
            "cls_loss": float(
                total_cls_loss.detach()
            ),
            "box_loss": float(
                total_box_loss.detach()
            ),
            "centerness_loss": float(
                total_ctr_loss.detach()
            ),
            "num_positive": num_positive,
        }

    def centerness_floor(self, targets):
        """
        Smallest centerness term any model can reach.

        Centerness targets are continuous in (0, 1), and BCE against a
        soft target t is minimized at p == t where it equals the binary
        entropy H(t) = -t*log(t) - (1-t)*log(1-t), which is strictly
        positive. The centerness term therefore never reaches zero, so
        convergence has to be judged against this floor and not against
        zero.

        Args:
            targets:
                Same structure the forward pass takes.

        Returns:
            Float contribution of a perfectly fitted centerness branch
            to the total loss, lambda_ctr included.
        """

        positive_targets = []

        for image_targets in targets:
            for level_target in image_targets:

                cls_targets = level_target[1]
                ctr_targets = level_target[3]

                positive = cls_targets >= 0

                if positive.any():
                    positive_targets.append(
                        ctr_targets[positive]
                    )

        if not positive_targets:
            return 0.0

        t = torch.cat(
            positive_targets
        ).clamp(1e-6, 1.0 - 1e-6)

        entropy = -(
            t * torch.log(t)
            + (1.0 - t) * torch.log1p(-t)
        )

        return float(
            self.lambda_ctr * entropy.mean()
        )

    @staticmethod
    def _decode_boxes(
        locations,
        ltrb,
    ):
        """
        Convert:

            left, top, right, bottom

        into:

            x1, y1, x2, y2
        """

        x = locations[:, 0]
        y = locations[:, 1]

        left = ltrb[:, 0]
        top = ltrb[:, 1]
        right = ltrb[:, 2]
        bottom = ltrb[:, 3]

        x1 = x - left
        y1 = y - top
        x2 = x + right
        y2 = y + bottom

        return torch.stack(
            [x1, y1, x2, y2],
            dim=1,
        )
