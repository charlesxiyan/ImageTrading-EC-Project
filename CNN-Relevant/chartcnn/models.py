"""Chart CNN in the style of Jiang, Kelly & Xiu (2023), adapted to 234x234 / 78-bar images."""
import numpy as np
import pandas as pd
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """ResNet basic block on chart features, followed by vertical-only pooling.

        x ─► Conv5x3 ─► BN ─► LeakyReLU ─► Conv5x3 ─► BN ─┐
        │                                                 (+) ─► LeakyReLU ─► MaxPool(2x1)
        └──────────── 1x1 Conv ─► BN (shortcut) ──────────┘

    The last BN of the main branch starts at zero, so at initialisation the block passes
    the shortcut through and the network behaves like a shallower plain CNN.
    """

    def __init__(self, c_in, c_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(c_in, c_out, (5, 3), padding=(2, 1), bias=False), nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c_out, c_out, (5, 3), padding=(2, 1), bias=False), nn.BatchNorm2d(c_out))
        self.shortcut = (nn.Identity() if c_in == c_out else
                         nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out)))
        self.act = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool2d((2, 1))
        nn.init.zeros_(self.main[-1].weight)

    def forward(self, x):
        return self.pool(self.act(self.main(x) + self.shortcut(x)))


class ChartCNN(nn.Module):
    """
    arch="plain"    (Jiang-Kelly-Xiu style)
        Block i : Conv(5x3) -> BatchNorm -> LeakyReLU -> MaxPool(2x1)   (pooling is vertical only)
    arch="residual" (ResNet upgrade)
        Block 1 as above; blocks 2.. are ResidualBlocks (two convs + skip connection)

    Block 1 : stride (3, 3), dilation (2, 1). A bar is exactly 3 pixels wide, so the
              horizontal stride maps every bar to one feature column (78 columns).
    Head    : AvgPool over height only (-> head_height rows) -> Flatten -> Dropout -> Linear(2)

    Keeping the 78 time columns preserves "which bar did what"; pooling only the height
    keeps the linear layer's fan-in near the paper's 20-day model (~80k vs ~46k).
    """

    def __init__(self, channels=(64, 128, 256), dropout=0.5, head_height=4, arch="plain",
                 input_hw=(234, 234)):
        super().__init__()
        if arch not in ("plain", "residual"):
            raise ValueError("arch must be 'plain' or 'residual'")
        self.arch = arch
        layers, c_in = [], 1
        for i, c_out in enumerate(channels):
            if i == 0:
                conv = nn.Conv2d(c_in, c_out, (5, 3), stride=(3, 3), dilation=(2, 1), padding=(12, 0))
                layers += [conv, nn.BatchNorm2d(c_out), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
            elif arch == "residual":
                layers.append(ResidualBlock(c_in, c_out))
            else:
                conv = nn.Conv2d(c_in, c_out, (5, 3), padding=(2, 1))
                layers += [conv, nn.BatchNorm2d(c_out), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
            c_in = c_out
        layers.append(nn.AdaptiveAvgPool2d((head_height, None)))
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            self.feature_shape = tuple(self.features(torch.zeros(1, 1, *input_hw)).shape[1:])
        self.flat_features = int(np.prod(self.feature_shape))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(self.flat_features, 2))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():                    # re-apply after the generic init above
            if isinstance(m, ResidualBlock):
                nn.init.zeros_(m.main[-1].weight)
        nn.init.zeros_(self.head[-1].weight)       # start at P = 0.5 exactly (loss = ln 2)

    def _on_my_device(self, x):
        """Move inputs to wherever the weights live, so CPU/GPU mix-ups can never crash."""
        return x.to(self.head[-1].weight.device, non_blocking=True)

    def embed(self, x):
        """Flattened feature map fed to the classifier."""
        return self.features(self._on_my_device(x)).flatten(1)

    def forward(self, x):
        return self.head(self.features(self._on_my_device(x)))


def build_model(settings):
    return ChartCNN(channels=tuple(settings["channels"]), dropout=settings["dropout"],
                    head_height=settings["head_height"], arch=settings.get("arch", "plain"))


def parameter_table(model, input_hw=(234, 234)):
    """Per-layer output shape and parameter count, plus the total (advisor rule: always report)."""
    device = next(model.parameters()).device            # works for CPU and GPU models alike
    rows, x, was_training = [], torch.zeros(1, 1, *input_hw, device=device), model.training
    model.eval()
    with torch.no_grad():
        for name, layer in list(model.features.named_children()) + [("head." + n, l) for n, l in model.head.named_children()]:
            x = layer(x)
            rows.append(dict(layer=f"{name}: {layer.__class__.__name__}",
                             output_shape=tuple(x.shape[1:]),
                             parameters=sum(p.numel() for p in layer.parameters())))
    model.train(was_training)                           # leave the model in the mode it came in
    table = pd.DataFrame(rows)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == table.parameters.sum()
    return table, total, trainable


def print_parameter_count(model, name="model"):
    _, total, trainable = parameter_table(model)
    print(f"{name}: {total:,} parameters ({trainable:,} trainable)")
    return total
