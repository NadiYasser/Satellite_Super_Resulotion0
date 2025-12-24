import torch
import torch.nn as nn
import torchvision.models as models

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, layers=['relu3_3'], use_input_norm=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.use_input_norm = use_input_norm

        # normalize input as VGG expects
        if use_input_norm:
            # VGG normalization (ImageNet)
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        # Extract sub-modules up to the highest requested layer
        self.vgg_layers = nn.Sequential()
        for i, layer in enumerate(vgg):
            self.vgg_layers.add_module(str(i), layer)
            # Stop at the highest layer index needed
            if 'relu4_3' in layers and i >= 16:  # relu4_3 index
                break
            if 'relu3_3' in layers and i >= 9:   # relu3_3 index
                break
            if 'relu2_2' in layers and i >= 6:   # relu2_2 index
                break
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        outputs = {}
        for name, layer in self.vgg_layers._modules.items():
            x = layer(x)
            layer_name = f'relu{name}'
            if layer_name in self.layers:
                outputs[layer_name] = x
        return outputs

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu2_2'], criterion=nn.L1Loss()):
        super().__init__()
        self.vgg_extractor = VGG16FeatureExtractor(layers=layers)
        self.criterion = criterion

    def forward(self, sr, hr):
        # sr: super-resolved image
        # hr: high-resolution ground truth
        sr_features = self.vgg_extractor(sr)
        hr_features = self.vgg_extractor(hr)
        loss = 0
        for layer in sr_features:
            loss += self.criterion(sr_features[layer], hr_features[layer])
        return loss
