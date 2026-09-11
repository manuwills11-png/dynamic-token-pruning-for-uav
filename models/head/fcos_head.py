import torch
import torch.nn as nn
import torch.nn.functional as F


class FCOSHead(nn.Module):
    
    def __init__(
        self,
        in_channels=256,
        num_classes=10,
        num_convs=4,
        prior_prob=0.01,
        strides=(4, 8, 16, 32),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.strides = strides

        # Shared convolutional tower.
        cls_tower = []

        for i in range(num_convs):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            cls_tower.append(nn.ReLU(inplace=True))

        self.cls_tower = nn.Sequential(*cls_tower)

        # Shared regression tower.
        bbox_tower = []

        for i in range(num_convs):
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            bbox_tower.append(nn.ReLU(inplace=True))

        self.bbox_tower = nn.Sequential(*bbox_tower)

        # Prediction layers.
        self.cls_logits = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bbox_reg = nn.Conv2d(
            in_channels,
            4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.centerness = nn.Conv2d(
            in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._initialize_weights(prior_prob)

    def _initialize_weights(self, prior_prob):
        """
        FCOS-style initialization.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight,
                    std=0.01,
                )

                if module.bias is not None:
                    nn.init.constant_(
                        module.bias,
                        0,
                    )

        # Initialize classification bias so that the initial
        # probability of a positive class is small.
        bias_value = -torch.log(
            torch.tensor(
                (1.0 - prior_prob) / prior_prob
            )
        )

        nn.init.constant_(
            self.cls_logits.bias,
            bias_value.item(),
        )

    def forward(self, features):
        """
        Args:
            features:
                List of four FPN feature maps.

        Returns:
            List of four dictionaries containing:
                cls_logits
                bbox_reg
                centerness
        """

        outputs = []

        for feature, stride in zip(features, self.strides):
            cls_features = self.cls_tower(feature)
            bbox_features = self.bbox_tower(feature)

            cls_logits = self.cls_logits(cls_features)

            bbox_reg = self.bbox_reg(bbox_features)

            # Regression output is in raw activation units;
            # scale by the level's stride to bring it into
            # input-image pixel units, matching how
            # FCOSTargetAssigner encodes bbox_targets.
            bbox_reg = F.softplus(bbox_reg) * stride

            centerness = self.centerness(bbox_features)

            outputs.append(
                {
                    "cls_logits": cls_logits,
                    "bbox_reg": bbox_reg,
                    "centerness": centerness,
                }
            )

        return outputs