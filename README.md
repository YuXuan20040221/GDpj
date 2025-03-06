# 畢業專題公告區

**🎉這裡是公告區🎉**   
進度 : 影像處理  
先摸摸看OpenCV怎麼用~ 反正報告第三周開始~

----------

GIT網址: [GIT這邊😺](https://github.com/YuXuan20040221/GDpj)  
筆記本: [onenote📒](https://1drv.ms/o/c/12296dadc52ae07c/ElysFhkG1M9On5YW2SsNj_EB2Z-11IqTPEMgS-L8gNm2CQ?e=WKIdJp)  
實驗記錄表格: [Google sheet✏️](https://docs.google.com/spreadsheets/d/1NRx1Qe7GxQ4leoN1b7u8ACHFELsfheebKNrmKK1_b4c/edit?usp=sharing)

--------------

第一次(還沒有虛擬環境時)，幫我:  
`python -m venv env`
(權限不足請開管理員模式 ! )

自己改好的程式要上傳幫我:  
`git branch 你的分支名字`  
`git checkout 分支名`  
再傳

資料集跟env名字幫我統一一下~  
才不會肥肥的整包上GIT~

(其他想到再補充)

# 會用到的指令

## 虛擬環境
**要先確認進入資料夾**

- *開啟虛擬環境:*  
```env\Scripts\activate```

- *退出虛擬環境:*  
```deactivate```

## 套件管理
**-requirements.txt是寫要裝什麼軟體的地方-**
### 安裝指令
- *安裝requirements.txt裡面寫的東西:*  
```pip install -r requirements.txt```

- *在requirements.txt裡面的都刪掉:*  
```pip uninstall -r requirements.txt``` 

- *安裝的東西寫進requirements.txt:*  
```pip freeze > requirements.txt```  

## GIT
**用 git bash 做**
### 提交到遠端
**(照順序做就OK)**
1. *檔案加進去暫存:*  
    ```git add "檔名"``` -加一個檔  
    ```git add . ``` -全加

2. *檔案提交:*  
    ```git commit -m "要講的訊息/備註"```

3. *把提交的東西丟雲端:*  
    ```git push```

### 從遠端拉進來
- ```git pull```

### 開分支
- *看目前有哪些分支:*  
    ```git branch```

- *建新分支:*  
    ```git branch 分支名```

- *切換分支:*  
    ```git checkout 分支名```  
    *(終端後面會出現`()`跟分支名字)*

### git flow
[gitflow介紹與規範](https://www.cnblogs.com/kevin-ying/p/14329768.html)

## 腳本
*預設參數:*  
`./train_yolo.sh`


*自訂參數:*  
`./train_yolo.sh 1280 32 100`

# 設定

## 目錄架構
```
|-- *dataset
|    |--origin_data (輸入資料集)  
|    |--processed_data (輸出資料集)   
|-- *env
|--img_processing  
|    |--main.py (讀取圖片)  
|    |--process_images.py (處理圖片)  
|--train_model
|    |--train_model.py (訓練模型)
|    |--test_model.py (測試模型)
|--.gitignore  
|--app.py  
|--README.md  
|--requirements.txt  
```
### - img_processing
#### - main.py
設定來源與輸出資料夾
- `input_dir`: 要處理的圖片放的位置
- `output_dir`: 處理完的圖片放的位置
#### - process_images.py
圖片處理
- `img`: 處理前圖片
- `processed_img`: 處理後圖片

### - model (yolov8s_customX)

|檔案|作用|
|----|----|
|args.yaml|記錄訓練時的超參數|
|F1.png	|F1-score 曲線（模型準確性指標）|
|results.csv|記錄訓練過程數據|
|results.png|訓練結果的圖表（loss, mAP, precision, recall）|
|train_batch0.png|訓練圖片 + 標註框 預覽|
|val_batch0.png	|驗證圖片 + 標註框 預覽|
|confusion_matrix.png|	類別混淆矩陣（查看哪些類別容易搞混）|
|weights/best.pt|最好的模型（可用於推論）|
|weights/last.pt|最後一輪的模型|
#### - args.yaml
紀錄訓練時用的各參數
#### - F1.png
F1-score 曲線，記錄訓練進展(上升=變好)
#### - result
損失（loss）曲線 📉：顯示模型的損失下降情況
mAP（均值平均精度）曲線 🎯：衡量模型對不同 IoU 門檻的表現
精準度 & 召回率曲線 🔍：顯示偵測的準確性

## 下載項目
- numpy: 建立數值矩陣
- OpenCV: 影像處理

# 資源

## 學程式

OpenCV官網 : 
[OpenCV](https://docs.opencv.org/4.x/)

YOLO資料解讀:
[训练结果](https://blog.csdn.net/matt45m/article/details/135620472)

## 資料集
