import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two convolution layers with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """U-Net encoder."""

    def __init__(self, in_channels, base_channels=16):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            base_channels * 8,
            base_channels * 16
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))

        return x1, x2, x3, x4, x5


class Decoder(nn.Module):
    """U-Net decoder."""

    def __init__(self, base_channels=16):
        super().__init__()

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, 2, stride=2
        )
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, 2, stride=2
        )
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 2, stride=2
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 2, stride=2
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        x = self.up4(x5)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.output(x)


class UNet(nn.Module):
    """Single-input U-Net for RGB, MS, or early fusion."""

    def __init__(self, in_channels, base_channels=16):
        super().__init__()

        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(base_channels)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class LateFusionUNet(nn.Module):
    """Separate RGB and multispectral encoders with feature fusion."""

    def __init__(self, base_channels=16):
        super().__init__()

        self.rgb_encoder = Encoder(3, base_channels)
        self.ms_encoder = Encoder(4, base_channels)

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.fusion = nn.ModuleList([
            ConvBlock(c * 2, c) for c in channels
        ])

        self.decoder = Decoder(base_channels)

    def forward(self, rgb, multispectral):
        rgb_features = self.rgb_encoder(rgb)
        ms_features = self.ms_encoder(multispectral)

        fused_features = []

        for rgb_feature, ms_feature, fusion_layer in zip(
            rgb_features,
            ms_features,
            self.fusion
        ):
            combined = torch.cat(
                [rgb_feature, ms_feature],
                dim=1
            )
            fused_features.append(fusion_layer(combined))

        return self.decoder(tuple(fused_features))


if __name__ == "__main__":
    print("Testing all four U-Net architectures...")

    device = torch.device("cpu")

    models = {
        "RGB U-Net": UNet(in_channels=3),
        "Multispectral U-Net": UNet(in_channels=4),
        "Early Fusion U-Net": UNet(in_channels=7),
        "Late Fusion U-Net": LateFusionUNet(),
    }

    batch_size = 2
    height = 192
    width = 256

    rgb = torch.randn(batch_size, 3, height, width)
    ms = torch.randn(batch_size, 4, height, width)
    early = torch.randn(batch_size, 7, height, width)

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            if name == "RGB U-Net":
                output = model(rgb)
            elif name == "Multispectral U-Net":
                output = model(ms)
            elif name == "Early Fusion U-Net":
                output = model(early)
            else:
                output = model(rgb, ms)

        print(f"{name}: {tuple(output.shape)}")

    print("\nALL MODEL TESTS PASSED")
