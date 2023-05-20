import os
import sys
from glob import glob

import torch.cuda

from UNet_model import UNet_nonTransferL, SegmentationDatasets
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import BCELoss, BCEWithLogitsLoss
import torch.cuda as cuda


def train_step(model, optimizer, criterion, src_imgs, target_imgs):
# 使用GPU與否
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    criterion = criterion.cuda().to(device, dtype=torch.float)
# ------------------------------------
    optimizer.zero_grad()
    outputs = model(src_imgs)

    loss = criterion(outputs, target_imgs)

    loss.backward()

    optimizer.step()

    return loss.item()

def Train(model, dataset, batch_sizes=16, epoches=50, learning_rate=1e-2):
# 使用GPU與否
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
# ---------------------------
    model = model.to(device)


# 損失函數、優化器、scheduler(學習率調適器) 選擇
    criterion = BCEWithLogitsLoss()
    criterion = criterion.cuda().to(device, dtype=torch.float)
    optimizer = Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # step_size 可以根據你的epoch大小來調整，其會自動追蹤目前是第幾個epoch來更新學習率。

    dataloader = DataLoader(dataset, batch_size=batch_sizes, shuffle=True)
    for epoch in range(epoches):
        scheduler.step()
        running_loss = 0.0
        for batch_images, batch_targets in dataloader:
            src_imgs = batch_images.to(device, dtype = torch.float32)
            target_imgs = batch_targets.to(device, dtype = torch.float32)
            loss = train_step(model=model, optimizer = optimizer, criterion=criterion,
                       src_imgs=src_imgs, target_imgs=target_imgs)

            running_loss += loss

        print(f"Epoch [{epoch}/{epoches}], Batch Loss: {running_loss / len(dataloader)}")
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

    print(model)
    sys.exit()

    # 超參數設定
    lr = 1e-2
    batch_sizes = 8
    epoches = 50

    Train(model, dataset, batch_sizes, epoches, learning_rate=lr)

    model.save_TrainedModel(path)