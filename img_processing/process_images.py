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
        [w * 0.1, h * 0.66],  # 左下
        [w * 0.9, h * 0.66]   # 右下
    ])
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, matrix, (w, h))
    # 2. 灰階 + 模糊處理
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 邊緣偵測
    edges = cv2.Canny(blurred, 50, 150)

    # 4. 膨脹處理使輪廓更清楚
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 5. 找輪廓（坑洞/裂縫候選區）
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 6. 過濾並繪製符合區域
    for cnt in contours:
        area = cv2.contourArea(cnt)    #計算每個輪廓面積
        if 200 < area < 3000:  # 根據實測可調整範圍(面積在範圍內才會選中,過濾掉太大或太小的物體)
            #用紅色框框出符合條件的
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(corrected, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)


    return corrected
