import cv2
import numpy as np
import os

def process_image(input_path, output_path):
    """讀取圖片、進行處理、存圖片"""
    try:
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 直接讀取灰階圖片
        if image is None:
            print(f"⚠️ 無法讀取 {input_path}")
            return False

        # 處理圖片（視角校正 -> 光源校正）
        processed_image = processing(image)

        # 確保輸出資料夾存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 存檔
        cv2.imwrite(output_path, processed_image)

        print(f"✅ 處理完成：{output_path}")
        return True

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")
        return False


def processing(img):
    """影像處理：視角校正 -> 光源校正"""
    h, w = img.shape

    # 1. **視角校正**
    src_pts = np.float32([
        [w * 0.1, h * 0.55],  # 左上
        [w * 1.0, h * 0.55],  # 右上
        [w * 0.1, h * 0.665],  # 左下
        [w * 0.9, h * 0.665]   # 右下
    ])
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, matrix, (w, h))
    cv2.imwrite("step1_corrected.jpg", corrected)  # 保存中間結果

    # 2. **光源校正**
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))  # 降低對比強度
    illuminated = clahe.apply(corrected)  # 這裡修正為 `corrected`
    cv2.imwrite("step2_illuminated.jpg", illuminated)  # 保存中間結果

    return illuminated


if __name__ == "__main__":
    # 測試
    input_image_path = "China_Drone_000048.jpg"  # 你的影像路徑
    output_image_path = "output_illuminated.jpg"

    success = process_image(input_image_path, output_image_path)

    if success:
        img = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("Processed Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
