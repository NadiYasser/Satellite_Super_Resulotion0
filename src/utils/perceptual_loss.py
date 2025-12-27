import torch
import torch.nn as nn
import torchvision.models as models

class VGG16FeatureExtractor(nn.Module):
    """
    Extracts intermediate VGG16 features for perceptual loss.
    """

    VGG_LAYER_MAP = {
        3:  'relu1_2',
        6:  'relu2_2',
        9:  'relu3_3',
        16: 'relu4_3'
    }

    def __init__(self, layers=['relu3_3'], use_input_norm=True):
        super().__init__()

        self.layers = layers
        self.use_input_norm = use_input_norm

        vgg = models.vgg16(pretrained=True).features

        # Normalization buffers (ImageNet)
        if use_input_norm:
            self.register_buffer(
                'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        # Build VGG layers only up to highest required layer
        max_layer_idx = max(
            idx for idx, name in self.VGG_LAYER_MAP.items() if name in layers
        )

        self.vgg_layers = nn.Sequential()
        for i, layer in enumerate(vgg):
            self.vgg_layers.add_module(str(i), layer)
            if i >= max_layer_idx:
                break

        # Freeze VGG weights
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        outputs = {}
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.VGG_LAYER_MAP:
                name = self.VGG_LAYER_MAP[i]
                if name in self.layers:
                    outputs[name] = x
        return outputs

class PerceptualLoss(nn.Module):
    """
    VGG perceptual loss using L1 distance between feature maps.
    """

    def __init__(self, layers=['relu3_3']):
        super().__init__()
        self.vgg_extractor = VGG16FeatureExtractor(layers=layers)
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        sr_features = self.vgg_extractor(sr)
        hr_features = self.vgg_extractor(hr)

        loss = 0.0
        for key in sr_features:
            loss += self.criterion(sr_features[key], hr_features[key])

        return loss


class CustomLoss(nn.Module):
    def __init__(self, perceptual_criterion , pixel_criterion=nn.L1Loss(),factor = 0.05):
        super().__init__()
        self.perceptual_criterion = perceptual_criterion
        self.pixel_criterion = pixel_criterion
        self.factor = factor
    def forward(self, sr, hr):
        # sr: super-resolved image
        # hr: high-resolution ground truth
        loss = self.pixel_criterion(sr,hr)
        loss += self.factor * self.perceptual_criterion(sr,hr)
        return loss
