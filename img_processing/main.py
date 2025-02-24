import glob
import os
import time
from process_images import process_image  # 引入處理函式

# 設定輸入與輸出資料夾
input_dir = "dataset/archive/pothole_dataset_v8/pothole_dataset_v8/train_to_valid/images/"  # 原始圖片資料夾

output_dir = f"dataset/Processed/{time.strftime('%Y%m%d_%H%M%S')}/"  # 處理後圖片的資料夾
os.makedirs(output_dir, exist_ok=True)  # 資料夾不在就生一個

# 找出所有圖片檔案
image_paths = glob.glob(input_dir + "*.jpg") + glob.glob(input_dir + "*.png")

print(f"找到 {len(image_paths)} 張圖片，開始處理...")

# 逐一處理圖片
for img_path in image_paths:
    # 取得圖片檔名
    filename = os.path.basename(img_path)

    # 設定輸出路徑
    output_path = os.path.join(output_dir, filename)

    # 呼叫 process_image 來處理圖片
    process_image(img_path, output_path)

print("🎉 全部圖片處理完成！")
