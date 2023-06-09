import os
from glob import glob

import torch.cuda
from torch.backends import cudnn
import numpy as np
from UNet_model import UNet_nonTransferL, SegmentationDatasets, Res_UNet
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler, SGD, RMSprop
from torch.autograd import Variable
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
import torch.cuda as cuda
import matplotlib.pyplot as plt
import cv2

CUDA_LAUNCH_BLOCKING="1"
torch.autograd.set_detect_anomaly(True) # 梯度檢測
cudnn.benchmark = True
torch.cuda.set_device(0)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def custom_collate(batch):
    images, labels = zip(*batch)

    # 轉換圖像為張量
    transform = transforms.ToTensor()
    images = [transform(image) for image in images]
    labels = [transform(label) for label in labels]

    # 組合成張量批次
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels
def PlotAccLoss(abs_path, acc, loss, dataset_name, epochs):

    x1 = range(0, epochs)
    x2 = range(0, epochs)
    y1 = acc
    y2 = loss
    #plt.subplots(211)
    plt.plot(x1, y1, 'o-')
    plt.title(f"{dataset_name} accuracy vs. epoches")
    plt.ylabel(f"{dataset_name} accuracy")
    plt.savefig(f"{abs_path}/{dataset_name}_accuracy.png",
                box_inches="tight")
    plt.show()
    #plt.subplots(212)
    plt.plot(x2, y2, 'o-')
    plt.title(f"{dataset_name} loss vs. epoches")
    plt.ylabel(f"{dataset_name} loss")
    plt.savefig(f"{abs_path}/{dataset_name}_loss.png",
                box_inches="tight")
    plt.show()
    plt.close()


def dataloader_plt(imgs, title):
    img = imgs[0]
    img_np = TF.to_pil_image(img)
    img_np = TF.to_grayscale(img_np)
    img_np = TF.to_tensor(img_np)
    img_np = img_np.numpy()
    img_np = np.transpose(img_np, (1,2,0))

    plt.imshow(img_np, cmap="gray")
    plt.imshow(img_np)
    plt.title(f"{title}")
    plt.axis('off')
    plt.show()

def train_step(model, optimizer, criterion, dataloader):
# 使用GPU與否
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    model.train()


    criterion = criterion.cuda().to(device, dtype=torch.float)
# ------------------------------------
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    for batch_images, batch_targets in dataloader:
        # src_imgs = batch_images.to(device, dtype=torch.float32)
        # target_imgs = batch_targets.to(device, dtype=torch.float32) / 255.0 # 歸一化

        src_imgs = Variable(batch_images.cuda())
        target_imgs = Variable(batch_targets.cuda())

        outputs = model(src_imgs)
        target_labels = target_imgs[:, 0, :, :]

        # image = outputs[0, :, :, :]
        # image_np = torch.transpose(image, 0, 2)
        # print(image_np.shape)
        # plt.imshow(image_np.detach().cpu().numpy(), cmap=None)
        # plt.axis("off")
        # plt.show()
        #
        # cv2.imshow("rgb image",image_np)
        # cv2.waitkey(0)
        # cv2.destropAllWindows()

        predict_labels = torch.argmax(outputs, dim=1)
        total_pixels += outputs.numel()  # 計算總pixel 數值
        correct_pixels += (predict_labels == target_labels).sum().item()

        optimizer.zero_grad()
        # print(type(outputs))
        # print(type(target_imgs))
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
        model.eval()
        for batch_images, batch_targets in dataloader:
            # src_imgs = batch_images.to(device, dtype=torch.float32)

            src_imgs = Variable(batch_images.cuda())
            target_imgs = Variable(batch_targets.cuda())

            # dataloader_plt(batch_images, "src image")
            # dataloader_plt(batch_targets, "src masked image")

            # target_imgs = batch_targets.to(device, dtype=torch.float32) / 255.0 # long
            target_labels = target_imgs[:, 0, :, :]
            outputs = model(src_imgs)
            loss = criterion(outputs, target_imgs)

            # dataloader_plt(outputs, "predict image")

            total_loss += loss.item()
            total_pixels += target_labels.numel()
            predicted_labels = torch.argmax(outputs, dim=1) #, keepdim=True)
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
    criterion = CrossEntropyLoss()# BCEWithLogitsLoss()# BCELoss()
    criterion = criterion.cuda().to(device, dtype=torch.float)
    optimizer = RMSprop(model.parameters(), lr = learning_rate)#, momentum=0.9) # Adam
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # step_size 可以根據你的epoch大小來調整，其會自動追蹤目前是第幾個epoch來更新學習率。
    scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.5)

    # dataloader = DataLoader(dataset, batch_size=batch_sizes, shuffle=True)
    train_size = int(0.9*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # 根据需要划分训练集和测试集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, collate_fn=custom_collate)

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    for epoch in range(epoches):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, criterion=criterion, optimizer = optimizer)
        test_loss, test_acc = test_step(model = model, dataloader=test_dataloader, criterion=criterion)
        print(f"Epoch [{epoch+1}/{epoches}]")
        print(f"\tTraining Loss: {train_loss}")
        print(f"\tTraining Pixel Accuracy : {train_acc}")
        print(f"\tTesting Loss: {test_loss}")
        print(f"\tTesting Pixel Accuracy : {test_acc}")
        scheduler.step()

        cuda.empty_cache()


        train_loss_all.append(train_loss / len(train_dataset))
        train_acc_all.append(train_acc / len(train_dataset))
        val_loss_all.append(test_loss / len(test_dataset))
        val_acc_all.append(test_acc / len(test_dataset))


    PlotAccLoss(os.path.join(os.getcwd(), "resize_dataset"), train_acc_all, train_loss_all, dataset_name="Train", epochs=epoches)
    PlotAccLoss(os.path.join(os.getcwd(), "resize_dataset"), val_acc_all, val_loss_all, dataset_name="Validation", epochs=epoches)



if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "resize_dataset")
    src_path = os.path.join(path, "src")
    target_path = os.path.join(path, "masked")

    src_paths = glob(os.path.join(src_path, '*.png'))
    target_paths = glob(os.path.join(target_path, '*.png'))

    print(src_paths)

    dataset = SegmentationDatasets(image_paths = src_paths, target_paths = target_paths)
    print(dataset)

    model = Res_UNet(3)
    # model = UNet_nonTransferL(3, 2)

    # print(model)
    # sys.exit()

    # 超參數設定
    lr = 1e-4
    batch_sizes = 8
    epoches = 100

    Train(model, dataset, batch_sizes, epoches, learning_rate=lr)

    model.save_TrainedModel(path)