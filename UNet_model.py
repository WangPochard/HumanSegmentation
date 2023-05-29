# import sys
import os
import sys

import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
# from pre_processing import *
from PIL import Image
from skimage.io import imread
import numpy as np

# 查看model structure，觀察整個ResNET中做了幾次downsampling & 縮小特徵圖像尺寸(Conv2d stride>2)
# resnet = models.resnet50(pretrained=True)
# print(resnet)
# for name, module in resnet.named_children():
#     print(name)
#
# sys.exit()
def custom_collate(batch):
    images, labels = zip(*batch)

    # 转换图像为张量
    transform = transforms.ToTensor()
    images = [transform(image) for image in images]
    labels = [transform(label) for label in labels]

    return images, labels
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
        # 將小圖像的尺寸放大到超過目標圖像尺寸
        if image.shape[0] < 512 or image.shape[1] < 512:
            image = self.resize_img(image)
            target = self.resize_img(target)
        # 圖像裁剪
        image = self.crop(image, 512)
        target = self.crop(target, 512)
        '''
        # 調整圖像明亮度 (只需調整原始圖像就好)
        # pixel_add = randint(-20, 20)
        # image = change_brightness(image, pixel_add)
        '''
        if self.transform:
            image, target = self.transform(image, target)

        return image, target
    def load_images(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def load_target(self, index):
        target_img = cv2.imread(self.target_paths[index])
        """b, g, r = cv2.split(target_img)
        target_img = cv2.merge((b, g))"""
        # target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        return target_img

    def crop(self, img, crop_size):
        img = np.array(img)
        dim1 = (int(img.shape[0])-int(crop_size))//2
        dim2 = (int(img.shape[1])-int(crop_size))//2
        crop_image = img[dim1:(dim1+crop_size), dim2:(dim2+crop_size)]
        return crop_image
    def resize_img(self, img, target_size=512):
        height, width = img.shape[:2]
        if height>width:
            dim = width
        else:
            dim = height
        scale = int(target_size//dim) + 1
        resize_img = cv2.resize(img, (width*scale, height*scale))
        return resize_img
    '''def resize_img_inter(self, img, target_size=512):
        # 用雙線性插值的方式做圖像放大 (但會有嚴重失真)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()

        # 計算尺寸比例
        height, width = img.shape[:2]
        scale = max(target_size / height, target_size / width)

        # 計算目標尺寸
        target_H = int(height * scale)
        target_W = int(width * scale)

        # 使用雙線性插值，進行圖像放大
        resize_img = F.interpolate(img_tensor, size=(target_H, target_W),
                                   mode='bilinear', align_corners=False)

        resize_img = resize_img.squeeze().numpy()
        return resize_img'''

# class SegmentationDatasets_skimage(Dataset):
#     def __init__(self, image_paths: list, target_paths: list, transform=None):
#         """
#         :param image_paths: 總共的資料集路徑
#         :param target_paths: 總共的資料集路徑
#         :param transform:
#         """
#         self.image_paths = image_paths
#         self.target_paths = target_paths
#         self.transform = transform
#     def __len__(self):
#         return len(self.image_paths) # 可能是個陣列、資料表，有多個圖像路徑，是為了返回數據樣本的數量
#     def __getitem__(self, index):
#         image = self.image_paths[index]
#         target = self.target_paths[index]
#
#         x, y = imread(image), imread(target)
#         if self.transform is not None:
#             x, y = self.transform(x, y)
#         x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.long)
#         return x, y


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
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # torchvision 新版的pretrained model 寫法
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
    import matplotlib.pyplot as plt
    from matplotlib import MatplotlibDeprecationWarning
    import torchvision.transforms.functional as TF
    import numpy as np
    from glob import glob
    from torch.utils.data import DataLoader

    import warnings
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

    path = os.path.join(os.getcwd(), "resize_dataset")
    src_path = os.path.join(path, "src")
    target_path = os.path.join(path, "masked")

    src_paths = glob(os.path.join(src_path, '*.png'))
    target_paths = glob(os.path.join(target_path, '*.png'))

    # print(src_paths)

    dataset = SegmentationDatasets(image_paths=src_paths, target_paths=target_paths)
    print(dataset)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # 根据需要划分训练集和测试集
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

    min_height = min(image.shape[0] for image, label in train_dataset)
    min_width = min(image.shape[1] for image, label in train_dataset)
    print(min_height,"\t", min_width)

    # sys.exit()

    data_iter = iter(train_dataloader)
    for images, labels in data_iter:

        img = images[0]

        img_np = TF.to_pil_image(img)
        img_np = TF.to_grayscale(img_np)
        img_np = TF.to_tensor(img_np)
        img_np = img_np.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))

        plt.imshow(img_np, cmap="gray")
        plt.imshow(img_np)
        plt.title(f"iter image")
        plt.axis('off')
        plt.show()

        img = labels[0]

        img_np = TF.to_pil_image(img)
        img_np = TF.to_grayscale(img_np)
        img_np = TF.to_tensor(img_np)
        img_np = img_np.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))

        plt.imshow(img_np, cmap="gray")
        plt.imshow(img_np)
        plt.title(f"iter image")
        plt.axis('off')
        plt.show()
        break

    data_iter = iter(test_dataloader)
    for images, labels in data_iter:
        img = images[0]

        img_np = TF.to_pil_image(img)
        img_np = TF.to_grayscale(img_np)
        img_np = TF.to_tensor(img_np)
        img_np = img_np.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))

        plt.imshow(img_np, cmap="gray")
        plt.imshow(img_np)
        plt.title(f"iter image")
        plt.axis('off')
        plt.show()

        img = labels[0]

        img_np = TF.to_pil_image(img)
        img_np = TF.to_grayscale(img_np)
        img_np = TF.to_tensor(img_np)
        img_np = img_np.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))

        plt.imshow(img_np, cmap="gray")
        plt.imshow(img_np)
        plt.title(f"iter image")
        plt.axis('off')
        plt.show()
        break
    model = Res_UNet(3)
    print(model)
    sys.exit()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    import numpy as np
    path = f"{os.getcwd()}"
    path = os.path.join(path, "resize_dataset")
    path = os.path.join(path, "masked")

    file_path = os.path.join(path, "0_1.png")
    print(file_path)

    img = cv2.imread(file_path)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # r,g,b = cv2.split(img)
    # img3 = cv2.merge((r,g))


    print(img.shape)
    print(img2.shape)
    # print(img3.shape)

    plt.imshow(img)
    plt.title("src")
    plt.show()

    plt.imshow(img2)
    plt.title("src bgr to gray")
    plt.show()


    # img = np.transpose(img, (1,2,0))
    # img = torch.tensor(img)
    print(img.shape)

    img_np = TF.to_pil_image(img)
    img_np = TF.to_grayscale(img_np)
    img_np = TF.to_tensor(img_np)
    img_np = img_np.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))

    plt.imshow(img_np, cmap="gray")
    plt.imshow(img_np)
    plt.title(f"")
    plt.axis('off')
    plt.show()

    """plt.imshow(img3)
    plt.title("src bgr to rgb")
    plt.show()"""

    # cv2.imshow("src",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow("src bgr to rgb", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()






# if __name__ == "__main__":
#     input_tensor = torch.randn(1, 3, 1024, 1024)
#     print(input_tensor.shape)
#     out = input_tensor[:,0,:,:]
#     print(out)
#     print(out.shape)
#
#     """resnet = models.resnet50(pretrained = True)
#     print(len(list(resnet.children())))
#     # print(list(resnet.named_children()))
#     for name, layer in resnet.named_children():
#         # if isinstance(layer, torch.nn.Conv2d):
#         print(name)
#     sys.exit()"""
#
#     # model = UNet_nonTransferL(3, 2)
#     model = Res_UNet(2)
#     # print(model)
#
#     out = model(input_tensor)
