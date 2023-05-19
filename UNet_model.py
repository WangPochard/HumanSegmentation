# import sys
import sys

from torch.utils.data import Dataset
from torch import nn
from torchvision import models
import cv2

# 查看model structure，觀察整個ResNET中做了幾次downsampling & 縮小特徵圖像尺寸(Conv2d stride>2)
# resnet = models.resnet50(pretrained=True)
# print(resnet)
# for name, module in resnet.named_children():
#     print(name)
#
# sys.exit()

class SegmentationDatasets(Dataset):
    def __init__(self, image_paths, target_paths, transform=None):
        """
        :param image_paths: 總共的資料集路徑
        :param target_paths: 總共的資料集路徑
        :param transform:
        """
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths) # 可能是個陣列、資料表，有多個圖像路徑，是為了返回數據樣本的數量
    def __getitem__(self, index):
        image = self.load_images(self.image_paths[index])
        target = self.load_target(self.image_paths[index])

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
    def load_images(self, image_path):
        img = cv2.imread(image_path)
        return img
        # pass
    def load_target(self, target_path):
        target_img = cv2.imread(target_path)
        return target_img
        # pass



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