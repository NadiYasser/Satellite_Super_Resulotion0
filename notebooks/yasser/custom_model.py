from turtle import forward
from numpy import concat
import torch
from torch import nn
import torch.nn.functional as F

class MultiScale_ResidualBlock(nn.Module):
    def __init__(self,channels,scale=0.1):
        super(MultiScale_ResidualBlock,self).__init__()
        self.small_conv1 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1,stride=1)
        self.large_conv1 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=5,padding=2,stride=1)
        self.small_conv2 = nn.Conv2d(in_channels=2*channels,out_channels=channels,kernel_size=3,padding=1,stride=1)
        self.large_conv2 = nn.Conv2d(in_channels=2*channels,out_channels=channels,kernel_size=5,padding=2,stride=1)
        self.fusion = nn.Conv2d(in_channels=2*channels,out_channels=channels,kernel_size=1,stride=1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
    def forward(self,X):
        skip = X
        output1 = self.relu(self.small_conv1(X))
        output2 = self.relu(self.large_conv1(X))
        shared = torch.cat([output1,output2],dim=1) # C --> 2C
        output1 = self.relu(self.small_conv2(shared)) # 2c -->c
        output2 = self.relu(self.large_conv2(shared)) # 2c -->c
        shared = torch.cat([output1,output2],dim=1) # C --> 2c
        shared = self.fusion(shared) # 2c --> C
        return self.scale*shared + X

class ResidualBlock(nn.Module):
    def __init__(self, channels=64, scale=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.scale = scale

    def forward(self, x):
        return x + self.block(x) * self.scale

class LaplaceConv(nn.Module):
    def __init__(self,channels = 64):
        super().__init__()
        Laplacian_Kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32)
        self.weight = nn.Parameter(Laplacian_Kernel.view(1,1,3,3).repeat(channels,1,1,1),requires_grad=False)
        self.groups = channels
    def forward(self,X):
        return F.conv2d(X,self.weight,padding=1,groups=self.groups)
        

class SubPixelConvBlock(nn.Module):
    def __init__(self,channels=64,up_scale=2):
        super(SubPixelConvBlock,self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels*(up_scale**2),kernel_size=3,padding=1,stride=1),
            nn.PixelShuffle(up_scale),
            nn.PReLU()
        )
    def forward(self,X):
        return self.Block(X)

class RefinementBlock(nn.Module):
    def __init__(self, output_channels = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(output_channels, output_channels,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels,kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class Custom_model(nn.Module):
    def __init__(self,channels = 64,n_MS_blocks = 1,n_LF_blocks = 12,n_HF_blocks = 6,SR_scale = 4):
        super(Custom_model,self).__init__()
        self.up_stream = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=channels,kernel_size=3,stride=1,padding=1),
            *[MultiScale_ResidualBlock(channels=channels) for _ in range(n_MS_blocks)]
            )
        
        self.low_frequency = nn.Sequential(
            *[ResidualBlock(channels=channels) for _ in range(n_LF_blocks)]
        )
        self.high_frequency = nn.Sequential(
            LaplaceConv(channels=channels),
            *[ResidualBlock(channels=channels) for _ in range(n_HF_blocks)]
        )
        self.fusion = nn.Conv2d(in_channels=2*channels,out_channels=channels,kernel_size=1,stride=1,padding=0)
        self.reconstruct = nn.Sequential(
            *[SubPixelConvBlock(channels=channels) for _ in range(SR_scale // 2)],
            nn.Conv2d(in_channels=channels,out_channels=3,kernel_size=1,padding=0,stride=1)
        )
        self.refinement = RefinementBlock(output_channels=3)         
        
    def forward(self,X):
        lr_B = F.interpolate(X, scale_factor=4,mode="bicubic", align_corners=False)
        X = self.up_stream(X)
        X = torch.cat([self.low_frequency(X),self.high_frequency(X)],dim = 1)
        X = self.fusion(X)
        X = self.reconstruct(X)
        X += lr_B
        return self.refinement(X)   
