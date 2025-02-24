from ultralytics import YOLO

# 創建 YOLOv8 模型 
model = YOLO("yolov8s.pt") 

# 設定訓練參數
model.train(
    data="train.yaml",  # 數據集設定
    epochs=50,          # 訓練回合數
    imgsz=640,          # 圖片大小
    project="model",    # 變更儲存目錄
    name="yolov8s_custom"  # 設定模型名稱
)
