import cv2
import os
import sys
import time
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    model = YOLO("yolov8n.pt")  # 載入模型

    # 檢查是否有輸入圖片路徑
    if len(sys.argv) < 2:
        print("❌ 請提供圖片路徑")
        sys.exit(1)

    # 取得輸入圖片路徑
    img_path = sys.argv[1]

    # 確保圖片存在
    if not os.path.exists(img_path):
        print(f"❌ 找不到圖片：{img_path}")
        sys.exit(1)
        
    start_time = time.time()
    output_img_path = predict(model, img_path)  # 模型預測
    end_time = time.time()
    print(f"- 耗時 : {end_time - start_time} 秒")
    output_img = cv2.imread(output_img_path)

    if output_img is None:
        print(f"❌ 無法讀取處理後的圖片：{output_img_path}")
        sys.exit(1)

    cv2.imshow("Processed", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict(model, img_path):
    results = model(img_path, save=True)

    save_dir = results[0].save_dir
    filename = os.path.basename(img_path)
    output_path = os.path.join(save_dir, filename)

    print(f"✅ {filename} 處理完成 → {output_path}")
    return output_path

if __name__ == "__main__":
    main()
