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

    # 道路區域遮罩
    road_mask = np.zeros((h, w), dtype=np.uint8)
    roi_pts = np.array([
        [int(w * 0.35), int(h * 0.2)],
        [int(w * 0.85), int(h * 0.2)],
        [int(w * 0.65), h],
        [int(w * 0.25), h]
    ])
    cv2.fillPoly(road_mask, [roi_pts], 255)

    # 灰階 + 模糊
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ⬛ 陰影遮罩：使用 YCrCb 的亮度（Y）判斷
    ycrcb = cv2.cvtColor(corrected, cv2.COLOR_BGR2YCrCb)
    y_channel, _, _ = cv2.split(ycrcb)
    shadow_mask = cv2.inRange(y_channel, 0, 100)  # Y<70 視為陰影
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

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
        if 4500 < area < 25000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.9:
                # 建立該輪廓遮罩
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)

                # 判斷是否在道路內
                overlap_road = cv2.bitwise_and(mask, road_mask)
                if cv2.countNonZero(overlap_road) == 0:
                    continue

                # 判斷是否與陰影區重疊
                overlap_shadow = cv2.bitwise_and(mask, shadow_mask)
                if cv2.countNonZero(overlap_shadow) > 0:
                    continue

                selected_contours.append(cnt)
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                boxes.append((x, y, w_box, h_box))

    # 畫遮罩
    mask = np.zeros_like(corrected)
    cv2.drawContours(mask, selected_contours, -1, (0, 0, 255), thickness=cv2.FILLED)

    # 疊加遮罩
    overlay = cv2.addWeighted(corrected, 1, mask, 0.5, 0)

    # 標記文字
    for (x, y, w_box, h_box) in boxes:
        cv2.putText(overlay, "pothole", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return overlay
