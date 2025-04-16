import glob
import os
import sys
import time
from pathlib import Path
from process_images import process_image  # 引入處理函式
from ultralytics import YOLO
sys.path.append(os.path.join(os.path.dirname(__file__), '../train_model/'))
from test_model import predict

# 設定輸入與輸出資料夾
input_dir = "./datasets/valid/images/"  # 原始圖片資料夾

output_dir = (
    f"runs/detect/{time.strftime('%Y%m%d_%H%M%S')}/"  # 處理後圖片的資料夾
)
os.makedirs(output_dir, exist_ok=True)  # 資料夾不在就生一個

# 找出所有圖片檔案
image_paths = (
    glob.glob(input_dir + "*.jpg")
    + glob.glob(input_dir + "*.png")
    + glob.glob(input_dir + "*.jpeg")
)

print(f"找到 {len(image_paths)} 張圖片，開始處理...")

# 逐一處理圖片
for img_path in image_paths:
    # 取得圖片檔名
    filename = os.path.basename(img_path)

    # 設定輸出路徑
    output_path = os.path.join(output_dir, filename)

    model_path = Path("./model/yolov8n_custom/weights/best.pt")
    assert model_path.exists(), f"❌ 模型檔案不存在：{model_path}"

    model = YOLO(str(model_path))
    predict(model, filename)

print("🎉 全部圖片處理完成！")
