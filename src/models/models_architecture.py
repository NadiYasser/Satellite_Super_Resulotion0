from torch import nn
import torch.nn.functional as F

# SRCNN
class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        
        # Convolution 1 — extraction de features
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        
        # Convolution 2 — mapping non-linéaire
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
        
        # Convolution 3 — reconstruction (sortie SR)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    

# EDSR

class ResidualBlock(nn.Module):
    def __init__(self, channels=64, scale=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.scale = scale

    def forward(self, x):
        return x + self.block(x) * self.scale

class EDSR(nn.Module):
    def __init__(self, num_blocks=32, scale_factor=4):
        super().__init__()

        # 1) Initial Feature Extractor
        self.conv_head = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # 2) Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        # 3) Global skip
        self.conv_tail = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 4) Upsampling
        up_layers = []
        if scale_factor == 2 or scale_factor == 4:
            for _ in range(scale_factor // 2):
                up_layers += [
                    nn.Conv2d(64, 256, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(True)
                ]
        self.upsample = nn.Sequential(*up_layers)

        # 5) Reconstruction
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x_head = self.conv_head(x)
        x_res = self.res_blocks(x_head)
        x = self.conv_tail(x_res) + x_head   
        x = self.upsample(x)
        x = self.conv_last(x)
        return x


