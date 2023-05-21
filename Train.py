import os
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


def train_step(model, optimizer, criterion, dataloader):
# 使用GPU與否
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)


    criterion = criterion.cuda().to(device, dtype=torch.float)
# ------------------------------------
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    for batch_images, batch_targets in dataloader:
        src_imgs = batch_images.to(device, dtype=torch.float32)
        target_imgs = batch_targets.to(device, dtype=torch.float32)

        outputs = model(src_imgs)
        target_labels = target_imgs[:, 0, :, :]
        predict_labels = torch.argmax(outputs, dim=1)
        total_pixels += outputs.numel()  # 計算總pixel 數值
        correct_pixels += (predict_labels == target_labels).sum().item()

        optimizer.zero_grad()
        batch_loss = criterion(outputs, target_imgs)

        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    avg_loss = total_loss / len(dataloader)
    acc = correct_pixels / total_pixels
    return avg_loss, acc
def test_step(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    with torch.no_grad():
        for batch_images, batch_targets in dataloader:
            src_imgs = batch_images.to(device, dtype=torch.float32)
            target_imgs = batch_targets.to(device, dtype=torch.long)
            target_labels = target_imgs[:, 0, :, :]

            outputs = model(src_imgs)
            loss = criterion(outputs, target_labels)

            total_loss += loss.item()
            total_pixels += target_labels.numel()
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_pixels += (predicted_labels == target_labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_pixels / total_pixels

    return avg_loss, accuracy

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
# BCEWithLogitsLoss : default activation func - Sigmoid
# CrossEntropyLoss : Softmax
    criterion = CrossEntropyLoss() # BCEWithLogitsLoss
    criterion = criterion.cuda().to(device, dtype=torch.float)
    optimizer = Adam(model.parameters(), lr = learning_rate)#, momentum=0.9) # Adam
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # step_size 可以根據你的epoch大小來調整，其會自動追蹤目前是第幾個epoch來更新學習率。

    # dataloader = DataLoader(dataset, batch_size=batch_sizes, shuffle=True)
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # 根据需要划分训练集和测试集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False)

    for epoch in range(epoches):
        scheduler.step()
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, criterion=criterion, optimizer = optimizer)
        test_loss, test_acc = test_step(model = model, dataloader=test_dataloader, criterion=criterion)
        print(f"Epoch [{epoch+1}/{epoches}]")
        print(f"\tTraining Loss: {train_loss}")
        print(f"\tTraining Pixel Accuracy : {train_acc}")
        print(f"\tTesting Loss: {test_loss}")
        print(f"\tTesting Pixel Accuracy : {test_acc}")

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