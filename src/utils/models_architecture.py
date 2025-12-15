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


#SRRESNET

class ResidualBlockWithBATCH(nn.Module):
    def __init__(self,channels):
        super(ResidualBlockWithBATCH,self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_features=channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_features=channels)
        ) 
    def forward(self, X):
        features = self.Block(X)
        return X + features

class SubPixelConvBlock(nn.Module):
    def __init__(self,channels=64,scal=2):
        super(SubPixelConvBlock,self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels*4,kernel_size=3,padding=1,stride=1),
            nn.PixelShuffle(scal),
            nn.PReLU()
        )
    def forward(self,X):
        return self.Block(X)


class SRResNet(nn.Module):
    """
    n_residual_Blocks is the number of of block inside the reisdual fase 
    upscale_factor = the ratio between the input and the output image
    channels is standard 64 used in the paper we are using 
    """
    
    
    
    
    def __init__(self,n_residual_blocks:int=16,upscal_factor:int=4,channels=64,):
        super(SRResNet,self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=channels,kernel_size=9,padding=4,stride=1),
            nn.PReLU()
        )
        
        
        residual_layers = []
        for _ in range(n_residual_blocks):
            residual_layers.append(ResidualBlockWithBATCH(channels=channels))
        self.residuals = nn.Sequential(*residual_layers)
        
        
        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=channels)
        )
        
        
        upsampling_layers = []
        for _ in range(upscal_factor // 2):
            upsampling_layers.append(SubPixelConvBlock(channels=channels,scal=2))
        self.SubPixelConv = nn.Sequential(*upsampling_layers)
        
        
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=3,kernel_size=9,stride=1,padding=4),
        )
    def forward(self, X):
        result = self.initial(X)
        saved = result
        result = self.residuals(result)
        result = self.mid_conv(result)
        result += saved
        result = self.SubPixelConv(result)
        result = self.final(result)
        return result