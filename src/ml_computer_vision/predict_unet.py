import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import to_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        x = self.pool1(c1)

        c2 = self.down2(x)
        x = self.pool2(c2)

        x = self.bridge(x)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, c1], dim=1)
        x = self.conv1(x)

        return self.out(x)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    0 = background    -> black
    1 = InnerShadow   -> green
    2 = OuterShadow   -> blue
    """
    colors = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )
    return colors[mask]


def overlay_mask(
    image: Image.Image, mask: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    img = np.array(image.convert("RGB"), dtype=np.float32)
    color_mask = mask_to_rgb(mask).astype(np.float32)

    active = mask > 0
    out = img.copy()
    out[active] = (1.0 - alpha) * img[active] + alpha * color_mask[active]
    return Image.fromarray(out.astype(np.uint8))


def predict_one(
    model: nn.Module,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int = 512,
) -> None:
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((image_size, image_size))

    x = to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    color_path = output_dir / f"{stem}_mask_color.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    Image.fromarray(pred).save(mask_path)
    Image.fromarray(mask_to_rgb(pred)).save(color_path)
    overlay_mask(resized, pred).save(overlay_path)

    inner_area = int((pred == 1).sum())
    outer_area = int((pred == 2).sum())
    ratio = None if outer_area == 0 else 1.0 - (inner_area / outer_area)

    print(f"image: {image_path}")
    print(f"saved mask: {mask_path}")
    print(f"saved color mask: {color_path}")
    print(f"saved overlay: {overlay_path}")
    print(f"InnerShadow pixels: {inner_area}")
    print(f"OuterShadow pixels: {outer_area}")
    if ratio is None:
        print("fill_ratio: undefined, OuterShadow area is zero")
    else:
        print(f"fill_ratio = 1 - InnerShadow/OuterShadow = {ratio:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict InnerShadow/OuterShadow masks with a trained U-Net."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/unet/unet_tank_shadow.pt"),
        help="Path to trained U-Net weights.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset_seg/images"),
        help="Image file or directory with crop images.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/unet/predict"),
        help="Output directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, mps, cuda, cuda:0. Default chooses mps/cuda/cpu automatically.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size used by the U-Net.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print("device:", device)

    model = UNet(num_classes=3).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if args.source.is_dir():
        image_paths = sorted(
            list(args.source.glob("*.png"))
            + list(args.source.glob("*.jpg"))
            + list(args.source.glob("*.jpeg"))
        )
    else:
        image_paths = [args.source]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.source}")

    for image_path in image_paths:
        predict_one(model, image_path, args.out, device, image_size=args.image_size)


if __name__ == "__main__":
    main()
