# 會用到的指令:

## 虛擬環境:

**要先確認進入資料夾**

*開啟虛擬環境*
    ```env\Scripts\activate```

*退出虛擬環境*
    ```deactivate```

## 套件管理

### 安裝指令
**requirements.txt是寫要裝什麼軟體的地方**

*安裝requirements.txt裡面寫的東西*
    ```pip install -r requirements.txt```

*在requirements.txt裡面的都刪掉*
    ```pip uninstall -r requirements.txt``` 

*安裝的東西寫進requirements.txt*
    ```pip freeze > requirements.txt```
    
### 已安裝套件
**OpenCV**:影像處理函式庫

## GIT

**用 git bash 做**

### 提交到遠端

**照順序做就OK**
**要先退出虛擬環境**

1. *檔案加進去暫存*
    ```git add "檔名"``` -加一個檔
    ```git add . ``` -全加

2. *檔案提交*
    ```git commit -m "要講的訊息/備註"```

3. *把提交的東西丟雲端*
    ```git push```

### 從遠端拉進來
```git pull```

### 開分支

*看目前有哪些分支*
    ```git branch```

*建新分支*
    ```git branch 分支名```

*切換分支*
    ```git checkout 分支名```
*(終端後面會出現'()'跟分支名字)*



# 學習資源:

## OpenCV官網
[OpenCV](https://docs.opencv.org/4.x/)