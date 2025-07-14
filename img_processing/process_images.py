import cv2
import numpy as np
import os

def process_image(input_path, output_path, label_output_path=None):
    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ 無法讀取 {input_path}")
            return False, False

        processed_image, boxes, img_shape = processing(image)

        # 儲存影像
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, processed_image)
        print(f"✅ 影像儲存：{output_path}")

        # 儲存標註
        if label_output_path:
            os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
            has_label = save_yolo_labels(label_output_path, boxes, img_shape)
            if has_label:
                print(f"📝 標註儲存：{label_output_path}")
            else:
                print(f"⚠️ 無標註：{os.path.basename(label_output_path)}")
        else:
            has_label = False

        return True, has_label

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")
        return False, False


def processing(img):
    h, w = img.shape[:2]

    src_pts = np.float32([
        [w * 0.1, h * 0.55],
        [w * 1.0, h * 0.55],
        [w * 0.1, h * 0.665],
        [w * 0.9, h * 0.665]
    ])
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, matrix, (w, h))

    road_mask = np.zeros((h, w), dtype=np.uint8)
    roi_pts = np.array([
        [int(w * 0.35), int(h * 0.2)],
        [int(w * 0.85), int(h * 0.2)],
        [int(w * 0.65), h],
        [int(w * 0.25), h]
    ])
    cv2.fillPoly(road_mask, [roi_pts], 255)

    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ycrcb = cv2.cvtColor(corrected, cv2.COLOR_BGR2YCrCb)
    y_channel, _, _ = cv2.split(ycrcb)
    shadow_mask = cv2.inRange(y_channel, 0, 90)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    edges = cv2.Canny(thresh, 20, 100)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
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
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                if cv2.countNonZero(cv2.bitwise_and(mask, road_mask)) == 0:
                    continue
                if cv2.countNonZero(cv2.bitwise_and(mask, shadow_mask)) > 0:
                    continue
                selected_contours.append(cnt)
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                boxes.append((x, y, w_box, h_box))

    mask = np.zeros_like(corrected)
    cv2.drawContours(mask, selected_contours, -1, (0, 0, 255), thickness=cv2.FILLED)
    overlay = cv2.addWeighted(corrected, 1, mask, 0.5, 0)

    for (x, y, w_box, h_box) in boxes:
        cv2.putText(overlay, "pothole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return overlay, boxes, (w, h)


def save_yolo_labels(label_path, boxes, img_shape):
    w_img, h_img = img_shape
    if not boxes:
        return False

    with open(label_path, 'w') as f:
        for (x, y, w_box, h_box) in boxes:
            x_center = (x + w_box / 2) / w_img
            y_center = (y + h_box / 2) / h_img
            width = w_box / w_img
            height = h_box / h_img
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    return True


def batch_process(input_dir, output_img_dir, output_label_dir):
    supported_ext = ['.jpg', '.png', '.jpeg']
    total = 0
    with_label = 0
    no_label = 0
    no_label_list = []

    for filename in os.listdir(input_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in supported_ext:
            continue

        input_path = os.path.join(input_dir, filename)
        output_img_path = os.path.join(output_img_dir, f"{name}.jpg")
        output_txt_path = os.path.join(output_label_dir, f"{name}.txt")

        success, has_label = process_image(input_path, output_img_path, output_txt_path)
        if not success:
            continue

        total += 1
        if has_label:
            with_label += 1
        else:
            no_label += 1
            no_label_list.append(filename)

    print("\n📊 統計報告")
    print(f"🔢 處理總張數：{total}")
    print(f"✅ 有標註：{with_label}")
    print(f"⚠️ 無標註：{no_label}")

    if no_label_list:
        print("\n🗂️ 無標註圖片：")
        for fname in no_label_list:
            print(f" - {fname}")


# 🏁 主程式
if __name__ == "__main__":
    from datetime import datetime

    # 取得今天的日期
    today = datetime.now().strftime("%Y-%m-%d_%H%M")

    # 設定資料夾
    input_dir = "input_images/images" #照片路徑
    base_output_dir = os.path.join("output", today)
    output_img_dir = os.path.join(base_output_dir, "images")
    output_label_dir = os.path.join(base_output_dir, "labels")

    # 建立資料夾（若尚未存在）
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 執行批次處理
    batch_process(input_dir, output_img_dir, output_label_dir)
