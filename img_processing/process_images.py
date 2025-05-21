import cv2
import numpy as np
import os

def process_image(input_path, output_path):
    """讀取圖片、進行處理、存圖片"""
    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ 無法讀取 {input_path}")
            return False

        # 處理圖片
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
    """影像處理：視角校正 + 光源校正 + 提高對比度 + 濾波"""
    h, w = img.shape[:2]

    # 視角校正
    src_pts = np.float32([
        [w * 0.1, h * 0.55],
        [w * 1.0, h * 0.55],
        [w * 0.1, h * 0.665],
        [w * 0.9, h * 0.665]
    ])
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, matrix, (w, h))

    # 光源校正
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    corrected_light = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 增強對比度
    alpha = 1.2
    beta = 2.5
    enhanced = cv2.convertScaleAbs(corrected_light, alpha=alpha, beta=beta)

    # 雙邊濾波 (Bilateral Filter)
    # 優點：能保留邊緣的同時去除雜訊（比 Gaussian 更聰明）
    #適用：需要去除雜訊又不想模糊邊界時
    blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    return blurred
