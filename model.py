import torch.nn as nn
from einops import rearrange


def create_AutoEncoder(args):
    return AutoEncoder(embedding_dim=args.feature_channels_num,
                       kernel_size=args.kernel_size,
                       font_channels=args.font_channels_num)


def parse_model(yaml_config):
    pass

class ConvBlock:

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,
                 output_size):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding,
                              stride)
        self.gelu = nn.GELU()
        self.maxpool = nn.AdaptiveAvgPool2d((output_size, output_size))

    def forward(self, x):
        return self.maxpool(self.gelu(self.conv(x)))


class DeconvBlock:

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         padding, stride)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.deconv(x))


class AutoEncoder:

    def __init__(self, embedding_dim=512, kernel_size=5, font_channels=1):
        super(AutoEncoder, self).__init__()
        self.encoder = TianzigeCNN(kernel_size, font_channels, embedding_dim)
        self.decoder = Decoder(embedding_dim, font_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TianzigeCNN(nn.Module):

    def __init__(self,
                 kernel_size=5,
                 font_channels=1,
                 embedding_dim=512,
                 dropout=0.3):
        super(TianzigeCNN, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(font_channels,
                               embedding_dim // 2,
                               kernel_size=kernel_size,
                               padding=padding)
        self.maxpool1 = nn.AdaptiveAvgPool2d((8, 8))
        self.conv2 = nn.Conv2d(embedding_dim // 2,
                               embedding_dim,
                               kernel_size=kernel_size,
                               padding=padding)
        self.maxpool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.conv3 = nn.Conv2d(embedding_dim,
                               embedding_dim // 4,
                               kernel_size=1,
                               padding=0)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.gelu(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.dropout(x)
        return x


class Decoder(nn.Module):

    def __init__(self, embedding_dim=512, font_channels=1):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(embedding_dim // 4,
                                          32,
                                          kernel_size=2,
                                          stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 8, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(8,
                                          font_channels,
                                          kernel_size=4,
                                          stride=4)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, 'b (c h w) -> b c h w', h=2, w=2)
        x = self.deconv1(x)
        x = self.gelu(x)
        x = self.deconv2(x)
        x = self.gelu(x)
        x = self.deconv3(x)
        x = self.sigmoid(x)
        return x
