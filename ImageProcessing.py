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
    return json_files

def MaskedImage_Save(src_img, src_JsonParam, path):
    """
    :param src_img: 透過cv2 imread 讀取的 image
    :param src_JsonParam: image 的標註參數
    :param path: file path
    :return: message (Failed or Success)
    """

    try:
        df_shape = pd.DataFrame.from_dict(src_JsonParam["shapes"].iloc[0])
        arr_pts = np.array(df_shape["points"].values[0], np.int32)

    # Color轉換成tuple形式之後還是沒辦法傳入cv2的func裡 (無論是 polylines or fillPoly 都不行) ↓
                                                            # arr_lineColor = np.array(df["lineColor"].iloc[0], np.int32)
                                                            # arr_fileColor = np.array(df["fillColor"].iloc[0], np.int32)
    # Create 一個 value為0的 img_mat
        img_zero = np.zeros(src_img.shape, dtype=np.uint8)
        image = cv2.fillPoly(img_zero, [arr_pts], color=(0, 255, 0))

        fileName = str(src_JsonParam["imagePath"].iloc[0])
# Processing Steps
    # 1. resize image
    # 2. up sampling (方向轉換(平移)、隨機翻轉、旋轉)
    # 3. 資料集切割(分另一個.py檔處理)
        cv2.imwrite() # resize source image
        cv2.imwrite(f"{path}/masked/{fileName}",image) # resize masked image

        msg = "Success!!!"
    except:
        msg = "Failed!!!"
    return msg


    # break

if __name__ == "__main__":
    path = f"{os.getcwd()}/dataset/"
    files_path = get_json_files(path)
    for file_path in files_path:
        print(file_path)
        f = open(file=file_path)
        json_detail = json.load(f)
        df_json_detail = pd.json_normalize(json_detail)  # JsonParam : json_detail
        print(df_json_detail)
        img = cv2.imread(file_path.replace(".json", ".png"))
        msg = MaskedImage_Save(img, df_json_detail, path)
        print(msg)
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