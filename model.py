import torch.nn as nn
from einops import rearrange
from collections import OrderedDict


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def create_layers(config, in_channels):
    layers = OrderedDict()

    for layer_cfg in config:
        layer_name = layer_cfg['name']
        layer_type = layer_cfg['type']
        if layer_type == 'conv':
            out_channels = layer_cfg['out_channels']
            kernel_size = layer_cfg['kernel_size']
            stride = layer_cfg['stride']
            padding = layer_cfg['padding']
            layers[layer_name] = nn.Conv2d(in_channels, out_channels,
                                           kernel_size, stride, padding)
            in_channels = out_channels
        elif layer_type == 'deconv':
            kernel_size = layer_cfg['kernel_size']
            stride = layer_cfg['stride']
            out_channels = layer_cfg['out_channels']
            layers[layer_name] = nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size, stride)
            in_channels = out_channels
        elif layer_type == 'gelu':
            layers[layer_name] = nn.GELU()
        elif layer_type == 'leaky_relu':
            layers[layer_name] = nn.LeakyReLU(negative_slope=layer_cfg['slope'])
        elif layer_type == 'sigmoid':
            layers[layer_name] = nn.Sigmoid()
        elif layer_type == 'maxpool':
            layers[layer_name] = nn.AdaptiveMaxPool2d(layer_cfg['output_shape'])
        elif layer_type == 'avgpool':
            layers[layer_name] = nn.AdaptiveAvgPool2d(layer_cfg['output_shape'])
        elif layer_type == 'dropout':
            layers[layer_name] = nn.Sequential(Reshape(layer_cfg['shape']),
                                               nn.Dropout(layer_cfg['rate']))
        elif layer_type == 'reshape':
            layers[layer_name] = Reshape(layer_cfg['shape'])
        elif layer_type == 'module':
            layers[layer_name] = create_layers(layer_cfg['layers'], in_channels)
            in_channels = layer_cfg['out_channels']
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

    return nn.Sequential(layers)


def create_AutoEncoder(config):
    in_channels = config['input_channels']
    layers = create_layers(config['layers'], in_channels)
    return layers
