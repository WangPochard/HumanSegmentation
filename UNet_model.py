# import sys
import os
import sys

import torch
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
        image = self.load_images(index)
        target = self.load_target(index)

        if self.transform:
            image, target = self.transform(image, target)

        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        # print(image.shape, "\t", type(image))
        # print(target.shape, "\t", type(target))

        target = torch.from_numpy(target.transpose((2, 0, 1))).float()

        # target = torch.tensor(target)
        # target = target.unsqueeze(0)
        # print(target.shape, "\t", type(target))
        return image, target
    def load_images(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def load_target(self, index):
        target_img = cv2.imread(self.target_paths[index])
        b, g, r = cv2.split(target_img)
        target_img = cv2.merge((b, g))
        # target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        return target_img



class UNet_nonTransferL(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_nonTransferL, self).__init__()

        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 縮小特徵圖像
        )
    # FCN : 透過反捲積做upsampling 到原始圖像大小
    # 输出大小 = （输入大小 - 1）* stride - 2 * padding + kernel_size
        self.decoder_block = nn.Sequential(
            # input:(64*64) Out size = 63*2-2*1+ks
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0), # Feature Map 被放大了2倍 // kernel_size = 4 (2*2)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # padding = 1 「 same padding = (kernel size -1)/2 」
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0), #
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, out_channels, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder_block(x)
        # print(f"encoder shape:{x.shape}")
        x = self.decoder_block(x)
        # print(f"decoder shape:{x.shape}")
        return x
    def save_TrainedModel(self, path):
        model_path = os.path.join(path, "Semantic_Segmentation.pt")
        torch.save(self.state_dict(), model_path)
        print("saved...")
    @staticmethod
    def load_TrainedModel(path):
        model_path = os.path.join(path, "Semantic_Segmentation.pt")
        model = torch.load(model_path)
        print('loaded...')
        return model

class Res_UNet(nn.Module):
    def __init__(self, out_channels):
        """
        :param in_channels: ResNET50 default=3
        :param out_channels: 依據我們 semantic segmentation任務中，masked image 是由幾種像素組成
        """
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
        # resnet = models.resnet50(pretrained = True)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet_encoder = nn.Sequential(*list(resnet.children())[:-4]) # 獲取resnet 的encoder區塊 -2

        # FCN : 透過反捲積做upsampling 到原始圖像大小
        self.decoder_block = nn.Sequential(
# -3
            # nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),  # Feature Map 被放大了2倍
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # # padding = 1 「 same padding = (kernel size -1)/2 」
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
# -4
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid() # 若我loss function選擇使用 BCEWithLogitsLoss則可以不用加上這一層，因此損失函數已經自動應用Sigmoid函數了
        )

    def forward(self, x):
        x = self.resnet_encoder(x)
        # print(x.shape)
        x = self.decoder_block(x)
        # print(x.shape)
        return x
    def save_TrainedModel(self, path):
        model_path = os.path.join(path, "Semantic_Segmentation.pt")
        torch.save(self.state_dict(), model_path)
        print("saved...")
    @staticmethod
    def load_TrainedModel(path):
        model_path = os.path.join(path, "Semantic_Segmentation.pt")
        model = torch.load(model_path)
        print('loaded...')
        return model


if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 1024, 1024)
    print(input_tensor.shape)
    out = input_tensor[:,0,:,:]
    print(out)
    print(out.shape)

    """resnet = models.resnet50(pretrained = True)
    print(len(list(resnet.children())))
    # print(list(resnet.named_children()))
    for name, layer in resnet.named_children():
        # if isinstance(layer, torch.nn.Conv2d):
        print(name)
    sys.exit()"""

    # model = UNet_nonTransferL(3, 2)
    model = Res_UNet(2)
    # print(model)

    out = model(input_tensor)
