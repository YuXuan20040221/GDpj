from ultralytics import YOLO

# 創建 YOLOv8 模型（可以用預訓練權重來微調）
model = YOLO("yolov8s.pt")

# 訓練模型，使用自訂的數據集
model.train(data="你的數據集.yaml", epochs=50, imgsz=640)
