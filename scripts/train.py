import torch
from torch.utils.data import DataLoader, Subset

from datasets.visdrone import VisDroneDataset
from datasets.transforms import Letterbox

from models.detector import UAVDetector
from models.head.fcos_target import FCOSTargetAssigner
from models.losses.fcos_loss import FCOSLoss


DATA_ROOT = "data/raw/VisDrone2019-DET"

BATCH_SIZE = 2
NUM_WORKERS = 0
LR = 1e-3


class TrainTransform:
    """
    Letterbox only.

    This is intentionally minimal for the initial
    training smoke test.
    """

    def __init__(self, size=640):
        self.letterbox = Letterbox(size=size)

    def __call__(self, image, boxes, labels):
        image, boxes = self.letterbox(
            image,
            boxes,
        )

        return image, boxes, labels


def collate_fn(batch):
    images = torch.stack(
        [item[0] for item in batch]
    )

    targets = [
        item[1]
        for item in batch
    ]

    return images, targets


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = VisDroneDataset(
        root=DATA_ROOT,
        split="train",
        transform=TrainTransform(size=640),
    )

    # Small subset for smoke test.
    dataset = Subset(
        dataset,
        range(8),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = UAVDetector(
        num_classes=10,
        pretrained=False,
        fpn_channels=256,
    )

    model = model.to(device)
    model.train()

    # ---------------------------------------------------------
    # FCOS components
    # ---------------------------------------------------------

    assigner = FCOSTargetAssigner()

    loss_fn = FCOSLoss(
        lambda_cls=1.0,
        lambda_loc=2.0,
        lambda_ctr=1.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    # ---------------------------------------------------------
    # One training step
    # ---------------------------------------------------------

    images, raw_targets = next(iter(loader))

    images = images.to(device)

    print(
        f"Batch images: {images.shape}"
    )

    # Forward through backbone + FPN once to get shapes.
    with torch.no_grad():

        features = model.backbone(images)

        fpn_features = model.fpn(
            features
        )

    feature_shapes = [
        feature.shape[-2:]
        for feature in fpn_features
    ]

    # ---------------------------------------------------------
    # Build FCOS targets
    # ---------------------------------------------------------

    fcos_targets = []

    for target in raw_targets:

        boxes = torch.as_tensor(
            target["boxes"],
            dtype=torch.float32,
            device=device,
        )

        labels = torch.as_tensor(
            target["labels"],
            dtype=torch.long,
            device=device,
        )

        image_targets = assigner(
            boxes,
            labels,
            feature_shapes,
        )

        fcos_targets.append(
            image_targets
        )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    optimizer.zero_grad()

    predictions = model(images)

    loss, info = loss_fn(
        predictions,
        fcos_targets,
    )

    print(
        f"Loss: {loss.item():.6f}"
    )

    print(
        f"Positive locations: "
        f"{info['num_positive']}"
    )

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    loss.backward()

    # ---------------------------------------------------------
    # Check gradients
    # ---------------------------------------------------------

    grad = model.head.cls_logits.weight.grad

    if grad is None:
        raise RuntimeError(
            "Classification head received no gradient."
        )

    if not torch.isfinite(grad).all():
        raise RuntimeError(
            "Non-finite gradient detected."
        )

    print(
        f"Mean head gradient: "
        f"{grad.abs().mean().item():.6f}"
    )

    # ---------------------------------------------------------
    # Optimizer step
    # ---------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=10.0,
    )

    optimizer.step()

    print()
    print(
        "PASS: training smoke test completed"
    )


if __name__ == "__main__":
    main()