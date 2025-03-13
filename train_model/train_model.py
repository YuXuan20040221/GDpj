import os
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 創建 YOLOv8 模型 
model = YOLO("yolov8n.pt") 

# 設定訓練參數
model.train(
    data="datasets/train.yaml",  # 數據集設定
    epochs=20,          # 訓練回合數
    imgsz=640,          # 圖片大小
    project="model",    # 變更儲存目錄
    name="yolov8n_custom"  # 設定模型名稱
)
