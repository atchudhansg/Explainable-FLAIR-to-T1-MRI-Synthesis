"""
Model architectures for FLAIR-to-T1 MRI Synthesis
Paper-compliant: ResNet-9 Generator, PatchGAN Discriminator
Comparative: U-Net Generator (vanilla Pix2Pix), CycleGAN Generator
"""
import torch
import torch.nn as nn
import functools

# ============================================================
# BUILDING BLOCKS
# ============================================================

class ResidualBlock(nn.Module):
    """Residual block with InstanceNorm for the ResNet-9 generator."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ============================================================
# PROPOSED METHOD: ResNet-9 Generator (~3.1M params)
# ============================================================

class ResNet9Generator(nn.Module):
    """
    ResNet-based encoder-decoder generator with 9 residual blocks.
    Paper spec: 7x7 init conv -> 2 downsamples -> 9 ResBlocks -> 2 upsamples -> 7x7 out conv
    Input: (B, 3, 256, 256) -> Output: (B, 3, 256, 256)
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=9):
        super().__init__()
        # Initial 7x7 convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # Downsampling: 64->128->256
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
        # 9 Residual blocks at 256 channels
        mult = 4  # ngf * 4 = 256
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        # Upsampling: 256->128->64
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
        # Output 7x7 convolution with Tanh
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ============================================================
# COMPARATIVE: U-Net Generator (Vanilla Pix2Pix)
# ============================================================

class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """U-Net generator for vanilla Pix2Pix comparison."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2)
        d4 = self.down4(d3); d5 = self.down5(d4); d6 = self.down6(d5)
        d7 = self.down7(d6); d8 = self.down8(d7)
        u1 = self.up1(d8, d7); u2 = self.up2(u1, d6); u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4); u5 = self.up5(u4, d3); u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


# ============================================================
# PatchGAN Discriminator (~2.7M params)
# ============================================================

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator producing 31x31 activation map.
    Paper spec: 4 conv layers with 4x4 kernels, LeakyReLU(0.2), InstanceNorm
    Input: (B, 6, 256, 256) [concatenated FLAIR + T1]
    Output: (B, 1, 31, 31)
    """
    def __init__(self, in_channels=6):
        super().__init__()
        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),   # 256->128
            *block(64, 128),                             # 128->64
            *block(128, 256),                            # 64->32
            *block(256, 512, stride=1),                  # 32->32
            nn.Conv2d(512, 1, 4, stride=1, padding=1),  # 32->31
        )

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], dim=1)
        return self.model(x)

    def forward_features(self, img_a, img_b):
        """Forward pass returning (prediction, intermediate_features).
        Features are captured after each LeakyReLU block for feature matching loss."""
        x = torch.cat([img_a, img_b], dim=1)
        features = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        return x, features


# ============================================================
# CycleGAN: Standalone Discriminator (single-image input)
# Per Zhu et al. 2017 (arXiv:1703.10593)
# ============================================================

class CycleGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator for CycleGAN.
    Unlike Pix2Pix's PatchGAN, this takes a SINGLE image (not a pair).
    Input: (B, 3, 256, 256) -> Output: (B, 1, 31, 31)
    """
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),  # 256->128
            *block(64, 128),                            # 128->64
            *block(128, 256),                           # 64->32
            *block(256, 512, stride=1),                 # 32->32
            nn.Conv2d(512, 1, 4, stride=1, padding=1),  # 32->31
        )

    def forward(self, x):
        return self.model(x)


class ImageBuffer:
    """Replay buffer of 50 previously generated images (CycleGAN paper, Sec 4).
    Stabilizes discriminator training by mixing old and new fakes."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images):
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                result.append(img)
            elif torch.rand(1).item() > 0.5:
                idx = torch.randint(0, self.max_size, (1,)).item()
                result.append(self.data[idx].clone())
                self.data[idx] = img
            else:
                result.append(img)
        return torch.cat(result, dim=0)


def get_cyclegan_models():
    """Create CycleGAN model set: 2 generators + 2 discriminators."""
    g_ab = ResNet9Generator()   # FLAIR -> T1
    g_ba = ResNet9Generator()   # T1 -> FLAIR
    d_a = CycleGANDiscriminator()  # discriminates FLAIR domain
    d_b = CycleGANDiscriminator()  # discriminates T1 domain
    return g_ab, g_ba, d_a, d_b


# ============================================================
# UTILITY
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_pair(model_name='resnet9'):
    """Factory to create generator + discriminator pairs."""
    if model_name == 'resnet9':
        gen = ResNet9Generator()
    elif model_name == 'unet':
        gen = UNetGenerator()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    disc = PatchGANDiscriminator()
    return gen, disc

if __name__ == '__main__':
    # Quick sanity check
    for name in ['resnet9', 'unet']:
        g, d = get_model_pair(name)
        x = torch.randn(1, 3, 256, 256)
        y = g(x)
        pred = d(x, y)
        print(f"{name}: G={count_parameters(g)/1e6:.2f}M, D={count_parameters(d)/1e6:.2f}M, "
              f"out={y.shape}, disc={pred.shape}")
    # CycleGAN
    g_ab, g_ba, d_a, d_b = get_cyclegan_models()
    x = torch.randn(1, 3, 256, 256)
    y = g_ab(x)
    print(f"CycleGAN: G_AB={count_parameters(g_ab)/1e6:.2f}M, G_BA={count_parameters(g_ba)/1e6:.2f}M, "
          f"D_A={count_parameters(d_a)/1e6:.2f}M, D_B={count_parameters(d_b)/1e6:.2f}M")
