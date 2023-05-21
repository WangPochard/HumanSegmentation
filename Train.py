import os
import sys
import time
from glob import glob

import torch.cuda
from torch.backends import cudnn

from UNet_model import UNet_nonTransferL, SegmentationDatasets
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler, SGD
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
import torch.cuda as cuda

CUDA_LAUNCH_BLOCKING="1"
torch.autograd.set_detect_anomaly(True) # 梯度檢測
cudnn.benchmark = True
torch.cuda.set_device(0)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train_step(model, optimizer, criterion, src_imgs, target_imgs, total_pixels, correct_pixels):
# 使用GPU與否
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)


    criterion = criterion.cuda().to(device, dtype=torch.float)
# ------------------------------------
    outputs = model(src_imgs)
    # print(outputs.shape)
    target_labels = target_imgs[:, 0, :, :]
    predict_labels = torch.argmax(outputs, dim=1)
    # print(predict_labels)
    # print(predict_labels.shape)
    total_pixels += outputs.numel() # 計算總pixel 數值
    correct_pixels += (predict_labels == target_labels).sum().item()

    optimizer.zero_grad()
    # print(outputs)
    # print(outputs.shape)
    # print(target_imgs.shape)
    # time.sleep(10)
    batch_loss = criterion(outputs, target_imgs)

    batch_loss.backward()
    optimizer.step()

    return batch_loss.item(), total_pixels, correct_pixels

def Train(model, dataset, batch_sizes=16, epoches=50, learning_rate=1e-2):
# 使用GPU與否
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    correct_pixels = 0
    total_pixels = 0
# ---------------------------
    model = model.to(device)

# 損失函數、優化器、scheduler(學習率調適器) 選擇
# BCEWithLogitsLoss : default activation func - Sigmoid
# CrossEntropyLoss : Softmax
    criterion = CrossEntropyLoss() # BCEWithLogitsLoss
    criterion = criterion.cuda().to(device, dtype=torch.float)
    optimizer = SGD(model.parameters(), lr = learning_rate, momentum=0.9) # Adam
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # step_size 可以根據你的epoch大小來調整，其會自動追蹤目前是第幾個epoch來更新學習率。

    dataloader = DataLoader(dataset, batch_size=batch_sizes, shuffle=True)
    for epoch in range(epoches):
        scheduler.step()
        running_loss = 0.0
        for batch_images, batch_targets in dataloader:
            src_imgs = batch_images.to(device, dtype = torch.float32)
            target_imgs = batch_targets.to(device, dtype = torch.float32)
            loss, total_pixels, correct_pixels = train_step(model=model, optimizer = optimizer, criterion=criterion,
                       src_imgs=src_imgs, target_imgs=target_imgs, total_pixels = total_pixels, correct_pixels = correct_pixels)

            running_loss += loss
        pixel_acc = correct_pixels/total_pixels # 計算pixel 準確率
        print(f"Epoch [{epoch+1}/{epoches}]")
        print(f"\tBatch Loss: {running_loss / len(dataloader)}")
        print(f"\tPixel Accuracy : {pixel_acc}")
        cuda.empty_cache()


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "resize_dataset")
    src_path = os.path.join(path, "src")
    target_path = os.path.join(path, "masked")

    src_paths = glob(os.path.join(src_path, '*.png'))
    target_paths = glob(os.path.join(target_path, '*.png'))

    print(src_paths)

    dataset = SegmentationDatasets(image_paths = src_paths, target_paths = target_paths)
    print(dataset)

    model = UNet_nonTransferL(3, 2)

    # print(model)
    # sys.exit()

    # 超參數設定
    lr = 1e-4
    batch_sizes = 16
    epoches = 50

    Train(model, dataset, batch_sizes, epoches, learning_rate=lr)

    model.save_TrainedModel(path)