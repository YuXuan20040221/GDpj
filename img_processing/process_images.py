import cv2
import numpy as np
import os

def process_image(input_path, output_path):
    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ 無法讀取 {input_path}")
            return False

        processed_image = processing(image)

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, processed_image)
        print(f"✅ 處理完成：{output_path}")
        return True

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")
        return False

def processing(img):
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
    
    # 灰階與模糊
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 邊緣與膨脹
    edges = cv2.Canny(thresh, 20, 100)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 找輪廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    selected_contours = []
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 20000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.9:
                selected_contours.append(cnt)
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                boxes.append((x, y, w_box, h_box))  # 保存用於標記

    # 遮罩
    mask = np.zeros_like(corrected)
    cv2.drawContours(mask, selected_contours, -1, (0, 0, 255), thickness=cv2.FILLED)

    # 疊加遮罩
    overlay = cv2.addWeighted(corrected, 1, mask, 0.5, 0)

    # 只針對每個外接框標記一次 pothole
    for (x, y, w_box, h_box) in boxes:
        cv2.putText(overlay, "pothole", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return overlay