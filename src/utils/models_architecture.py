import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
#------------------------------------

# SRCNN

#------------------------------------
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
    
#------------------------------------

# EDSR

#------------------------------------

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

#------------------------------------

#SRRESNET

#------------------------------------

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
    def __init__(self,channels=64,up_scale=2):
        super(SubPixelConvBlock,self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels*(up_scale**2),kernel_size=3,padding=1,stride=1),
            nn.PixelShuffle(up_scale),
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
            upsampling_layers.append(SubPixelConvBlock(channels=channels,up_scale=2))
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
    


#------------------------------------

# ESRGAN

#------------------------------------

# GENERATOR 
class Dense_layer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        out = self.relu(self.conv(x))
        # dim=1 concatination au niveau des channels
        return torch.cat([x, out], dim=1)

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, n_dense_layers=4, growth_rate=32, residual_scale=0.2):
        super().__init__()
        
        self.residual_scale = residual_scale
        self.rdb = nn.Sequential(*[Dense_layer(in_channels + i * growth_rate, growth_rate) for i in range(n_dense_layers)])
        self.conv = nn.Conv2d(in_channels + n_dense_layers * growth_rate, in_channels, 3, 1, 1)
        
    def forward(self, x):
        out = self.rdb(x)
        return self.conv(out) * self.residual_scale + x
    
class RRDB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out

class GENERATOR(nn.Module):
    def __init__(self, in_channels, num_features, num_blocks):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels, num_features, 9, 1, 9//2)
        
        self.RRDB = nn.Sequential(
            *[RRDB(num_features) for i in range(num_blocks)]
        )
        
        self.tail_res = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(num_features, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(num_features, 3, 9, 1, 9//2),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.first_conv(x)
        out = self.RRDB(x)
        out = self.tail_res(out) + x
        out = self.upsample(out)
        return out

# DISCRIMINATOR 
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_ch = 64
        for _ in range(3):
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_ch, in_ch*2, 3, 1, 1),
                nn.LeakyReLU(0.2)
            ))
            in_ch *= 2
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.fc(x)
        return x

# VGG FOR PERCIPTUAL LOSS
class VGGFeatureExtractor(nn.Module):
    def __init__(self, device='cuda', feature_layer=34):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.features = nn.Sequential(*list(vgg.children())[:feature_layer+1]).to(device)

    def forward(self, x):
        return self.features(x)
    

#------------------------------------

# custom model

#------------------------------------

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


class LaplaceConv(nn.Module):
    def __init__(self,channels = 64):
        super().__init__()
        Laplacian_Kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32)
        self.weight = nn.Parameter(Laplacian_Kernel.view(1,1,3,3).repeat(channels,1,1,1),requires_grad=False)
        self.groups = channels
    def forward(self,X):
        return F.conv2d(X,self.weight,padding=1,groups=self.groups)
        


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
