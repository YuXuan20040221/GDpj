import cv2
import numpy as np
import os

def process_image(input_path, output_path):
    """讀取圖片、進行處理、存圖片"""
    try:
        #image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 直接讀取灰階圖片
        image = cv2.imread(input_path)  # 保留彩色圖
        if image is None:
            print(f"⚠️ 無法讀取 {input_path}")
            return False

        # 處理圖片（僅視角校正）
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
    """影像處理：僅進行視角校正"""
    h, w = img.shape[:2]  # ✅ 改這裡，支援彩色圖片

    # 視角校正
    src_pts = np.float32([
        [w * 0.1, h * 0.55],  # 左上
        [w * 1.0, h * 0.55],  # 右上
        [w * 0.1, h * 0.665],  # 左下
        [w * 0.9, h * 0.665]   # 右下
    ])
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, matrix, (w, h))

    return corrected
