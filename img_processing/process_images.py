import glob
import os
import time
import cv2
import numpy as np

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

def order_points(pts):
    """將4個點依序排序為 左上、右上、右下、左下 (順時針)"""
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]        # 左上角，x+y最小
    rect[2] = pts[np.argmax(s)]        # 右下角，x+y最大
    rect[1] = pts[np.argmin(diff)]     # 右上角，x-y最小
    rect[3] = pts[np.argmax(diff)]     # 左下角，x-y最大

    return rect

def detect_road_corners(img):
    """動態偵測路面四角點，用於透視校正"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append([x1, y1])
            points.append([x2, y2])
    else:
        return None

    points = np.array(points)
    if len(points) < 4:
        return None

    # 找四個角點候選
    sum_pts = points[:, 0] + points[:, 1]
    diff_pts = points[:, 0] - points[:, 1]

    top_left = points[np.argmin(sum_pts)]
    bottom_right = points[np.argmax(sum_pts)]
    top_right = points[np.argmin(diff_pts)]
    bottom_left = points[np.argmax(diff_pts)]

    raw_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    src_pts = order_points(raw_pts)
    return src_pts

def draw_points(img, pts):
    """畫出點以供檢查"""
    for idx, pt in enumerate(pts):
        cv2.circle(img, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
        cv2.putText(img, str(idx), tuple(pt.astype(int) + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Detected Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processing(img):
    """影像處理：視角校正(動態偵測)+光源校正+提高對比度+濾波"""
    h, w = img.shape[:2]

    src_pts = detect_road_corners(img)
    if src_pts is None:
        print("⚠️ 無法偵測路面角點，使用預設視角校正")
        src_pts = np.float32([
            [w * 0.1, h * 0.55],
            [w * 1.0, h * 0.55],
            [w * 0.9, h * 0.665],
            [w * 0.1, h * 0.665]
        ])
    else:
        # 如果你想看點位可以取消下一行註解
        # draw_points(img.copy(), src_pts)
        pass

    dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, matrix, (w, h))

    # 光源校正
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    corrected_light = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 對比度增強
    alpha = 1.2
    beta = 2.5
    enhanced = cv2.convertScaleAbs(corrected_light, alpha=alpha, beta=beta)

    # 雙邊濾波
    blurred = cv2.bilateralFilter(enhanced, d=3, sigmaColor=80, sigmaSpace=60)

    return blurred
