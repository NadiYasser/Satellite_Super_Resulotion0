import torch
from torch import nn
import torch.nn.init as init


# Functions used by ESRGAN model, cuz it requires diffrent parameters and setup



# function for psnr calculation the generator in ERSGAN uses Tanh as an output activation
def calculate_psnr(sr, hr, data_range=2.0):
    """
    FUNCTION Used by ESRGAN
    Calculates PSNR between SR and HR tensors in [-1, 1] range.
    sr, hr: Tensors of shape (B, C, H, W)
    """
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))



# function for weights intialisation for the first stage of training an ESRGAN model
def initialize_weights(m):
    """
    Standard Kaiming Initialization for ESRGAN components.
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        
        # Scaling down the initial weights slightly as suggested in ESRGAN paper
        m.weight.data *= 0.1 
        
        if m.bias is not None:
            init.zeros_(m.bias)
            
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.zeros_(m.bias)