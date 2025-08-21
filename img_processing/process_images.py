import glob
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================== 用 Matplotlib 點四個點 ==================
def get_src_points(img):
    """用 Matplotlib 點選道路四角，回傳座標"""
    h, w = img.shape[:2]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("👉 請依序點選道路四角 (左上 → 右上 → 左下 → 右下)")
    pts = plt.ginput(4, timeout=0)  # 點 4 個點，無時間限制
    plt.close()

    if len(pts) != 4:
        raise ValueError(f"⚠️ 你只點了 {len(pts)} 個點，請重新執行！")

    src_pts = np.float32(pts)
    print("✅ 選取的四個點：", src_pts)
    return src_pts

# ================== 圖片處理 ==================
def process_image(input_path, output_path, src_pts):
    """讀取圖片、進行處理、存圖片"""
    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ 無法讀取 {input_path}")
            return False

        processed_image = processing(image, src_pts)

        # 確保輸出資料夾存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, processed_image)
        print(f"✅ 處理完成：{output_path}")
        return True

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")
        return False

def processing(img, src_pts):
    """影像處理：視角校正 + 光源校正 + 對比度增強 + 高斯濾波"""
    h, w = img.shape[:2]
    (tl, tr, bl, br) = src_pts

    # 計算輸出大小
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # 調整輸出長寬比接近正方形
    if max_width > max_height:
        max_height = max_width
    else:
        max_width = max_height

    # 定義輸出四角點
    dst_pts = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [0, max_height - 1],
        [max_width - 1, max_height - 1]
    ])

    # 1. 視角校正
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, matrix, (max_width, max_height))

    # 2. 光源校正 (CLAHE)
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    corrected_light = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # 3. 增強對比度
    alpha, beta = 1.2, 2.5
    enhanced = cv2.convertScaleAbs(corrected_light, alpha=alpha, beta=beta)

    # 4. 高斯濾波 (Gaussian Blur)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=1.0)

    return blurred

# ================== MAIN ==================
if __name__ == "__main__":
    # 設定輸入與輸出資料夾
    input_dir = "../dataset/frames/"
    output_dir = f"../dataset/Processed/{time.strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(output_dir, exist_ok=True)

    # 找一張圖讓你先點選四角
    sample_img_path = glob.glob(input_dir + "*.jpg")[0]
    sample_img = cv2.imread(sample_img_path)
    src_pts = get_src_points(sample_img)  # 用滑鼠點道路四角

    # 找出所有圖片
    image_paths = glob.glob(input_dir + "*.jpg") + glob.glob(input_dir + "*.png")
    print(f"找到 {len(image_paths)} 張圖片，開始處理...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        process_image(img_path, output_path, src_pts)

    print("🎉 全部圖片處理完成！")
