# HumanSegmentation

筆記 : https://www.evernote.com/shard/s724/sh/cc63c896-338d-1591-ff3f-8108b516af22/EwlNPlkikI2i-HrstgzfNUBdww-Zqe6NmiYkk7she2gPCs6mU3Od5mta7w

---

### 人像分割任務，自我學習專案


目前圖像預處理有特別做resize、crop，resize的部分有嘗試使用torch的雙線性插值去做圖像放大，但實驗結果會導致圖像嚴重失真，使用兩種mode: 1.bicubic, 2.bilinear都沒有太好的效果，
固後來決定使用等比例放大的方式去做resize，但內插的方法還是有在dataset中，做為紀錄使用，目前的訓模過程中，難以達到良好的訓練成效

可能的原因有以下

1.資料量不夠多，資料集是kaggle上的開源資料庫

2.模型複雜度不夠 (因本人的硬體設備蠻弱小的，只使用了各4層卷積、反卷積的UNet架構模型，encoder的卷積層是使用預訓練模型的ResNet50，但在decoder做解碼的部分4層反卷積恐怕是不夠的)

---
目前想透過「數據增強」的方式去做資料量的擴增，就像醫學影像常用的做法一樣，也許對模型訓練上會有明顯的效果。
