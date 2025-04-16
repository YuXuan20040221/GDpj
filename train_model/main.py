import glob
import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), '../train_model/'))
from test_model import predict

# 設定輸入與輸出資料夾
input_dir = "./datasets/valid/images/"  # 原始圖片資料夾

# 自動產生輸出資料夾（如 runs/detect/20250411_140530）
output_dir = f"runs/detect/{time.strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(output_dir, exist_ok=True)

# 找出所有圖片檔案
image_paths = glob.glob(input_dir + "*.jpg") + glob.glob(input_dir + "*.png") + glob.glob(input_dir + "*.jpeg")
print(f"找到 {len(image_paths)} 張圖片，開始處理...")

# 確認模型存在
model_path = Path("./model/yolov8n_custom/weights/best.pt")
assert model_path.exists(), f"❌ 模型檔案不存在：{model_path}"

# 只載入一次模型
model = YOLO(str(model_path))

start_time = time.time()

# 開始逐張處理
for img_path in image_paths:
    predict(model, img_path)
    
end_time = time.time()

print("🎉 全部圖片處理完成！")
print(f"- 耗時 : {end_time - start_time} 秒")
