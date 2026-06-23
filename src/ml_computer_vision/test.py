import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


class TankSegDataset(Dataset):
    def __init__(self, image_paths, augment=False):
        self.image_paths = image_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = Path("test/masks") / img_path.name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.augment:
            # Horizontal Flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical Flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Rotação aleatória
            angle = random.uniform(-30, 30)

            image = TF.rotate(
                image,
                angle,
                interpolation=TF.InterpolationMode.BILINEAR,
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=TF.InterpolationMode.NEAREST,
            )

        image = TF.to_tensor(image)

        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()

        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
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
    def __init__(self, num_classes=3):
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


def main():
    image_paths = sorted(Path("test/crops").glob("*.png"))
    random.seed(42)
    random.shuffle(image_paths)

    n = len(image_paths)
    print(n)
    train_n = int(n * 0.75)

    train_paths = image_paths[:train_n]
    val_paths = image_paths[train_n:]

    train_loader = DataLoader(
        TankSegDataset(train_paths, augment=False),
        batch_size=4,
        shuffle=True,
    )

    val_loader = DataLoader(
        TankSegDataset(val_paths, augment=False),
        batch_size=4,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    model = UNet(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

        print(
            f"epoch={epoch + 1:03d} "
            f"train_loss={train_loss / len(train_loader):.4f} "
            f"val_loss={val_loss / max(1, len(val_loader)):.4f}"
        )

    Path("runs/unet").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "runs/unet/unet_tank_shadow.pt")


if __name__ == "__main__":
    main()
