import os
# import sys
# import time

import pandas as pd
import numpy as np
import glob
import json
import cv2

try:
    os.mkdir(f"{os.getcwd()}/resize_dataset/")
    os.mkdir(f"{os.getcwd()}/resize_dataset/src/")
    os.mkdir(f"{os.getcwd()}/resize_dataset/masked/")
except:
    pass

target_path = f"{os.getcwd()}/resize_dataset/"

def get_json_files(path):
    search_path = os.path.join(path, '*.json')
    json_files = glob.glob(search_path)

    png_files = glob.glob(os.path.join(path, "*.png"))
    files_name = [os.path.basename(file_name) for file_name in png_files]
    df_FileName = pd.DataFrame(pd.Series(files_name), columns=["img_0"])
    df_FileName["img_1"] = df_FileName["img_0"]
    df_FileName["img_0"] = df_FileName["img_0"].apply(lambda x: f"0_{x}")
    df_FileName["img_1"] = df_FileName["img_1"].apply(lambda x: f"1_{x}")
    print(df_FileName)
    return json_files, df_FileName

def MaskedImage_Save(src_img, src_JsonParam, path):
    """
    :param src_img: 透過cv2 imread 讀取的 image
    :param src_JsonParam: image 的標註參數
    :param path: file path
    :return: message (Failed or Success)
    """
    fileName = str(src_JsonParam["imagePath"].iloc[0])
    # resize shape
    widgh = 1000
    height = 1000
    try:
        df_shape = pd.DataFrame.from_dict(src_JsonParam["shapes"].iloc[0])
        arr_pts = np.array(df_shape["points"].values[0], np.int32)
        # Color轉換成tuple形式之後還是沒辦法傳入cv2的func裡 (無論是 polylines or fillPoly 都不行) ↓
        # arr_lineColor = np.array(df["lineColor"].iloc[0], np.int32)
        # arr_fileColor = np.array(df["fillColor"].iloc[0], np.int32)
        # Create 一個 value為0的 img_mat
        img_zero = np.zeros(src_img.shape, dtype=np.uint8)
        masked_img = cv2.fillPoly(img_zero, [arr_pts], color=(0, 255, 0))
        # print(fileName)
# Processing Steps
    # 1. resize image
    # 2. up sampling (方向轉換(平移)、隨機翻轉、旋轉)
        Src_images = dict()
        src_img = cv2.resize(src_img, (widgh, height))
        Src_images.update({f"0_{fileName}": src_img})
        Src_images.update({f"1_{fileName}": cv2.flip(src_img, 1)}) # 水平翻轉

        Mask_images = dict()
        masked_img = cv2.resize(masked_img, (widgh, height))
        Mask_images.update({f"0_{fileName}": masked_img})
        Mask_images.update({f"1_{fileName}": cv2.flip(masked_img, 1)}) # 水平翻轉
        # Save image
        for ImgName in Src_images.keys():
            # print(f"{path}src/{ImgName}")
            # print(f"FileName:\t{ImgName}")
            SrcImg = Src_images[ImgName]
            # print(type(SrcImg))
            cv2.imwrite(f"{path}src/{ImgName}", SrcImg)  # resize source image
        for ImgName in Mask_images.keys():
            # print(f"{path}src/{ImgName}")
            MaskImg = Mask_images[ImgName]
            cv2.imwrite(f"{path}masked/{ImgName}", MaskImg)  # resize masked image

    # 3. 資料集切割(分另一個.py檔處理)

        # cv2.imwrite(f"{path}/src/{fileName}",src_image) # resize source image
        # cv2.imwrite(f"{path}/masked/{fileName}",masked_image) # resize masked image

        msg = "Success!!!"
    except:
        print(f"{fileName}\t{src_JsonParam.columns}\t{src_JsonParam['shapes'].iloc[0]}")
        msg = "Failed!!!"
    return msg


if __name__ == "__main__":
    path = f"{os.getcwd()}/dataset/"
    process_path = f"{os.getcwd()}/resize_dataset/"
    files_path, files_name = get_json_files(path)
    files_name.to_csv(f"{process_path}FilesName.csv", encoding="utf_8_sig", index=False)
    # print(files_path)
    for file_path in files_path:
        # print(file_path)
        f = open(file=file_path)
        json_detail = json.load(f)
        df_json_detail = pd.json_normalize(json_detail)  # JsonParam : json_detail
        # print(df_json_detail)
        img = cv2.imread(file_path.replace(".json", ".png"))
        msg = MaskedImage_Save(img, df_json_detail, process_path)
        # print(msg)
# print(df)
# print(df.columns)
#
# print(df["shapes"].iloc[0])
#
# df_shape = pd.DataFrame.from_dict(df["shapes"].iloc[0])
# print(df_shape)
#
# # print(df_shape["points"].values[0])
#
# arr_pts = np.array(df_shape["points"].values[0], np.int32)
# arr_lineColor = np.array(df["lineColor"].iloc[0], np.int32)
# arr_fileColor = np.array(df["fillColor"].iloc[0], np.int32)
# file_name = str(df["imagePath"].iloc[0])
#
# # print(df_shape["points"].values[0])
# # sys.exit()
# print(f"---\npts arr : {arr_pts}")
# print(arr_pts.shape)
#
# img = cv2.imread(files[0].replace(".json", ".png"))
# print(f"---\nimg shape : {img.shape}")
#
# print(df["lineColor"].iloc[0])
# print(arr_lineColor[0])
# print(arr_lineColor[:3][::-1])
# color = tuple(arr_lineColor[:3][::-1])
# print(color)
# print(type(color))
# print(type((0,255,0)))
#
#
# # 創建masked image (0,1) , cv2要求 image 的數據類型要是uint8的格式
# img_zero = np.zeros(img.shape, dtype=np.uint8)
# # img_zero = np.zeros((540, 960, 3), dtype=np.uint8)
#
# # print(img)
# # print(img_zero)
#
# print(arr_pts.shape)
# print(img_zero.shape)
#
# # thickness 指線寬 , polylines 繪製多邊形的邊框, fillPoly 則是填充多邊形
# # image = cv2.polylines(img_zero, [arr_pts], isClosed = True, color = (0,255,0), thickness = 2)
# image = cv2.fillPoly(img_zero, [arr_pts], color = (0,255,0))
#
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# while(1):
#     cv2.imshow('image', image)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()