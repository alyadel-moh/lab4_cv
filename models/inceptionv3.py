
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Basic building block: Conv → BN → ReLU
# ─────────────────────────────────────────────────────────────────────────────
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Inception Module A  (used in the 35×35 grid)
# Factorizes 5×5 convolutions into two stacked 3×3 convolutions.
# ─────────────────────────────────────────────────────────────────────────────
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_proj):
        super().__init__()

        # Branch 1: simple 1×1
        self.branch1 = ConvBnRelu(in_channels, 64, kernel_size=1)

        # Branch 2: 1×1 then 5×5 (kept as-is for Module A)
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, 48, kernel_size=1),
            ConvBnRelu(48, 64, kernel_size=5, padding=2)
        )

        # Branch 3: 1×1 then two 3×3
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_channels, 64, kernel_size=1),
            ConvBnRelu(64, 96, kernel_size=3, padding=1),
            ConvBnRelu(96, 96, kernel_size=3, padding=1)
        )

        # Branch 4: avg pool then 1×1 projection
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # Concatenate along channel dimension
        return torch.cat([b1, b2, b3, b4], dim=1)  # out: 64+64+96+pool_proj


# ─────────────────────────────────────────────────────────────────────────────
# Grid Reduction A: 35×35 → 17×17
# ─────────────────────────────────────────────────────────────────────────────
class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Branch 1: 3×3 strided conv (no padding → halves spatial size)
        self.branch1 = ConvBnRelu(in_channels, 384, kernel_size=3, stride=2)

        # Branch 2: 1×1 → 3×3 → 3×3 strided
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, 64, kernel_size=1),
            ConvBnRelu(64, 96, kernel_size=3, padding=1),
            ConvBnRelu(96, 96, kernel_size=3, stride=2)
        )

        # Branch 3: plain max pool
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)  # out: 384+96+in_channels


# ─────────────────────────────────────────────────────────────────────────────
# Inception Module B  (used in the 17×17 grid)
# Factorizes n×n conv into 1×n followed by n×1 (asymmetric factorization).
# ─────────────────────────────────────────────────────────────────────────────
class InceptionB(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super().__init__()
        c7 = channels_7x7

        # Branch 1: plain 1×1
        self.branch1 = ConvBnRelu(in_channels, 192, kernel_size=1)

        # Branch 2: 1×1 → 1×7 → 7×1
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, c7, kernel_size=1),
            ConvBnRelu(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        # Branch 3: 1×1 → 7×1 → 1×7 → 7×1 → 1×7 (double factorization)
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_channels, c7, kernel_size=1),
            ConvBnRelu(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        # Branch 4: avg pool → 1×1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # out: 192×4 = 768


# ─────────────────────────────────────────────────────────────────────────────
# Grid Reduction B: 17×17 → 8×8
# ─────────────────────────────────────────────────────────────────────────────
class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Branch 1: 1×1 → 3×3 strided
        self.branch1 = nn.Sequential(
            ConvBnRelu(in_channels, 192, kernel_size=1),
            ConvBnRelu(192, 320, kernel_size=3, stride=2)
        )

        # Branch 2: 1×1 → 1×7 → 7×1 → 3×3 strided
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, 192, kernel_size=1),
            ConvBnRelu(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(192, 192, kernel_size=3, stride=2)
        )

        # Branch 3: max pool
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)  # out: 320+192+in_channels


# ─────────────────────────────────────────────────────────────────────────────
# Inception Module C  (used in the 8×8 grid)
# Expands filters by splitting 3×3 into parallel 1×3 and 3×1 branches.
# ─────────────────────────────────────────────────────────────────────────────
class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Branch 1: 1×1
        self.branch1 = ConvBnRelu(in_channels, 320, kernel_size=1)

        # Branch 2: 1×1 → [1×3 , 3×1]  (parallel split)
        self.branch2_reduce = ConvBnRelu(in_channels, 384, kernel_size=1)
        self.branch2a = ConvBnRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2b = ConvBnRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 3: 1×1 → 3×3 → [1×3 , 3×1]
        self.branch3_reduce = nn.Sequential(
            ConvBnRelu(in_channels, 448, kernel_size=1),
            ConvBnRelu(448, 384, kernel_size=3, padding=1)
        )
        self.branch3a = ConvBnRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3b = ConvBnRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 4: avg pool → 1×1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)

        b2 = self.branch2_reduce(x)
        b2 = torch.cat([self.branch2a(b2), self.branch2b(b2)], dim=1)  # 768

        b3 = self.branch3_reduce(x)
        b3 = torch.cat([self.branch3a(b3), self.branch3b(b3)], dim=1)  # 768

        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)  # 320+768+768+192 = 2048


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary Classifier
# Injected after the 3rd InceptionA block to fight vanishing gradients.
# Only used during training; disabled at inference.
# ─────────────────────────────────────────────────────────────────────────────
class AuxClassifier(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)        # → 5×5
        self.conv1 = ConvBnRelu(in_channels, 128, kernel_size=1) # → 128 ch
        self.conv2 = ConvBnRelu(128, 768, kernel_size=5)          # → 1×1
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full InceptionV3
# ─────────────────────────────────────────────────────────────────────────────
class InceptionV3(nn.Module):

    def __init__(self, num_classes=4, aux_logits=True, dropout=0.5):
        super().__init__()
        self.aux_logits = aux_logits

        # ── Stem ────────────────────────────────────────────────────────────
        # Input: 3×299×299  →  32×149×149
        self.stem = nn.Sequential(
            ConvBnRelu(3, 32, kernel_size=3, stride=2),          # 149×149
            ConvBnRelu(32, 32, kernel_size=3),                    # 147×147
            ConvBnRelu(32, 64, kernel_size=3, padding=1),         # 147×147
            nn.MaxPool2d(kernel_size=3, stride=2),                # 73×73
            ConvBnRelu(64, 80, kernel_size=1),                    # 73×73
            ConvBnRelu(80, 192, kernel_size=3),                   # 71×71
            nn.MaxPool2d(kernel_size=3, stride=2)                 # 35×35
        )

        # ── Inception A blocks (35×35 grid) ─────────────────────────────────
        # Three InceptionA modules.  pool_proj changes the 4th branch width.
        # Channel counts: 64+64+96+32=256, 64+64+96+64=288, 64+64+96+64=288
        self.inceptionA1 = InceptionA(192, pool_proj=32)   # out: 256
        self.inceptionA2 = InceptionA(256, pool_proj=64)   # out: 288
        self.inceptionA3 = InceptionA(288, pool_proj=64)   # out: 288

        # ── Reduction A: 35×35 → 17×17 ──────────────────────────────────────
        # out channels: 384+96+288 = 768
        self.reductionA = ReductionA(288)

        # ── Inception B blocks (17×17 grid) ─────────────────────────────────
        # channels_7x7 grows from 128 → 160 → 160 → 192 across four B blocks.
        self.inceptionB1 = InceptionB(768, channels_7x7=128)   # out: 768
        self.inceptionB2 = InceptionB(768, channels_7x7=160)   # out: 768
        self.inceptionB3 = InceptionB(768, channels_7x7=160)   # out: 768
        self.inceptionB4 = InceptionB(768, channels_7x7=192)   # out: 768

        # ── Auxiliary Classifier (branches off after B1) ─────────────────────
        if aux_logits:
            self.aux_classifier = AuxClassifier(768, num_classes)

        # ── Reduction B: 17×17 → 8×8 ────────────────────────────────────────
        # out: 320+192+768 = 1280
        self.reductionB = ReductionB(768)

        # ── Inception C blocks (8×8 grid) ───────────────────────────────────
        # Two InceptionC modules — each outputs 2048 channels
        self.inceptionC1 = InceptionC(1280)   # out: 2048
        self.inceptionC2 = InceptionC(2048)   # out: 2048

        # ── Classifier head ──────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # 2048×1×1
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)

        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Inception A
        x = self.inceptionA1(x)
        x = self.inceptionA2(x)
        x = self.inceptionA3(x)

        # Reduction A
        x = self.reductionA(x)

        # Inception B — aux branches off after the first B block
        x = self.inceptionB1(x)

        aux = None
        if self.aux_logits and self.training:
            aux = self.aux_classifier(x)

        x = self.inceptionB2(x)
        x = self.inceptionB3(x)
        x = self.inceptionB4(x)

        # Reduction B
        x = self.reductionB(x)

        # Inception C
        x = self.inceptionC1(x)
        x = self.inceptionC2(x)

        # Head
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux  # tuple during training
        return x           # single tensor during eval

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper (matches the pattern in your other model files)
# ─────────────────────────────────────────────────────────────────────────────
def get_inception_v3(num_classes: int, aux_logits: bool = True,
                     dropout: float = 0.5) -> InceptionV3:
    """Return an InceptionV3 instance ready for training."""
    return InceptionV3(num_classes=num_classes,
                       aux_logits=aux_logits,
                       dropout=dropout)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = get_inception_v3(num_classes=8)
    model.eval()

    dummy = torch.randn(2, 3, 299, 299)
    out = model(dummy)

    print(f"Output shape : {out.shape}")          # (2, 8)
    print(f"Total params : {sum(p.numel() for p in model.parameters()):,}")