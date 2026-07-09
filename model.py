"""
model.py
--------
Simple CNN architecture shared by all four pipelines.

Supports both greyscale (Fashion-MNIST / MNIST) and RGB (CIFAR-10) inputs
via the in_channels and image_size constructor arguments.

Architecture (greyscale, image_size=28)
----------------------------------------
  Conv1   : 1  → 16 ch, 3×3, padding=1, ReLU → MaxPool 2×2  [28×28 → 14×14]
  Conv2   : 16 → 32 ch, 3×3, padding=1, ReLU → MaxPool 2×2  [14×14 →  7×7]
  Flatten : 32 × 7 × 7 = 1,568 features
  FC1     : 1,568 → 128, ReLU, Dropout(p=0.3)
  FC2     : 128 → num_classes

Architecture (RGB, image_size=32, i.e. CIFAR-10)
-------------------------------------------------
  Conv1   :  3 → 32 ch, 3×3, padding=1, ReLU → MaxPool 2×2  [32×32 → 16×16]
  Conv2   : 32 → 64 ch, 3×3, padding=1, ReLU → MaxPool 2×2  [16×16 →  8×8]
  Flatten : 64 × 8 × 8 = 4,096 features
  FC1     : 4,096 → 256, ReLU, Dropout(p=0.4)
  FC2     : 256 → num_classes

Scaling rule
------------
  For RGB inputs (in_channels == 3), filter counts are doubled and FC1
  is widened proportionally to handle the richer colour feature space.
  The architecture is kept intentionally simple in both cases.

Monte Carlo Dropout
-------------------
  The Dropout layer in FC1 is intentionally kept active during inference
  by calling enable_dropout(model) before each stochastic forward pass.
  Averaging over multiple such passes gives the mean probability vector
  from which predictive entropy is derived.

  This follows the standard MC Dropout formulation (Gal & Ghahramani, 2016).
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Minimal two-conv-block CNN with a single dropout layer.

    Layer names (conv1, conv2, pool, fc1, fc2) are kept stable across
    configurations so that the AKRM embedding extractor, which accesses
    layers by name, requires no modifications when switching datasets.
    """

    def __init__(
        self,
        num_classes: int   = 10,
        dropout_p:   float = 0.3,
        in_channels: int   = 1,
        image_size:  int   = 28,
    ):
        """
        Parameters
        ----------
        num_classes : Output dimension (10 for MNIST / Fashion-MNIST / CIFAR-10)
        dropout_p   : Dropout probability for FC1
        in_channels : 1 for greyscale (MNIST / Fashion-MNIST),
                      3 for RGB (CIFAR-10)
        image_size  : Spatial size of input images (assumed square).
                      28 for MNIST / Fashion-MNIST, 32 for CIFAR-10.
        """
        super().__init__()

        # Scale filter counts and FC1 width for RGB inputs
        c1         = 32  if in_channels == 3 else 16
        c2         = 64  if in_channels == 3 else 32
        fc1_hidden = 256 if in_channels == 3 else 128

        # Two MaxPool2d(2,2) operations each halve spatial dimensions
        after_pool = image_size // 4          # 7 for 28px, 8 for 32px
        fc1_in     = c2 * after_pool * after_pool   # 1,568 or 4,096

        self.conv1   = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(c1, c2,         kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1     = nn.Linear(fc1_in,    fc1_hidden)
        self.fc2     = nn.Linear(fc1_hidden, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # → (B, c1, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))   # → (B, c2, H/4, W/4)
        x = x.view(x.size(0), -1)             # → (B, fc1_in)
        x = self.dropout(F.relu(self.fc1(x))) # → (B, fc1_hidden)
        x = self.fc2(x)                        # → (B, num_classes)
        return x

    def extract_features(self, x):
        """Extracts the dense representation before the final classifier."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)) # Note: No dropout during feature extraction
        return x


def enable_dropout(model: nn.Module) -> None:
    """
    Switch only Dropout layers to train mode while keeping everything else
    in eval mode.

    This is the standard approach for Monte Carlo Dropout inference:
    BatchNorm statistics stay frozen (eval) while dropout remains stochastic
    (train), enabling T stochastic forward passes for uncertainty estimation.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def get_model(
    num_classes: int   = 10,
    dropout_p:   float = 0.3,
    in_channels: int   = 1,
    image_size:  int   = 28,
) -> SimpleCNN:
    """Factory — returns a freshly initialised SimpleCNN."""
    return SimpleCNN(
        num_classes = num_classes,
        dropout_p   = dropout_p,
        in_channels = in_channels,
        image_size  = image_size,
    )
