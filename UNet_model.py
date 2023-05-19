import sys

import torch
from torch import nn
from torchvision import models

resnet = models.resnet50(pretrained=True)
print(resnet)
for name, module in resnet.named_children():
    print(name)

sys.exit()

class UNet_nonTransferL(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_nonTransferL, self).__init__()

        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 縮小特徵圖像
        )
    # FCN : 透過反捲積做upsampling 到原始圖像大小
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # Feature Map 被放大了2倍
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # padding = 1 「 same padding = (kernel size -1)/2 」
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, out_channels, kernel_size=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder_block(x)

        x = self.decoder_block(x)

        return x

class Res_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_UNet, self).__init__()
        """self.encoder_block = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)  # 縮小特徵圖像
                )"""
        resnet = models.resnet50(pretrained = True)
        self.resnet_encoder = nn.Sequential(*list(resnet.children())[:-2]) # 獲取resnet 的encoder區塊



        # FCN : 透過反捲積做upsampling 到原始圖像大小
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Feature Map 被放大了2倍
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # padding = 1 「 same padding = (kernel size -1)/2 」
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, out_channels, kernel_size=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_block(x)

        x = self.decoder_block(x)

        return x