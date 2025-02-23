import cv2
import os
import sys
from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")  # 載入模型

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

    output_img_path = predict(model, img_path)  # 模型預測
    output_img = cv2.imread(output_img_path)

    if output_img is None:
        print(f"❌ 無法讀取處理後的圖片：{output_img_path}")
        sys.exit(1)

    cv2.imshow("Processed", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict(model, img):
    results = model(img, save=True)  # 進行物件偵測，並存檔

    # 取得 YOLO 存放的資料夾
    save_dir = results[0].save_dir  # YOLO 會自動存圖片的資料夾
    filename = os.path.basename(img)  # 取得圖片檔名

    output_path = os.path.join(save_dir, filename)  # 組合路徑
    print(f"✅ 處理後的圖片儲存於: {output_path}")  # 確認儲存位置

    return output_path


if __name__ == "__main__":
    main()
