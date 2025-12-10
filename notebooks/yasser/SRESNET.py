import torch
import torch.nn as nn





class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
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
    def __init__(self,n_residual_blocks:int=16,upscal_factor:int=4,channels=64,):
        super(SRResNet,self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=channels,kernel_size=9,padding=4,stride=1),
            nn.PReLU()
        )
        
        
        residual_layers = []
        for _ in range(n_residual_blocks):
            residual_layers.append(ResidualBlock(channels=channels))
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