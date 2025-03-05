import cv2
import os
import numpy as np
import sys


def main():
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

    input_img = cv2.imread(img_path)
    output_img = processing(input_img)
    if len(output_img.shape) == 2:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

    # 顯示原圖與處理後的結果
    combined = np.hstack(
        [cv2.resize(input_img, (500, 500)), cv2.resize(output_img, (500, 500))]
    )  # 水平拼接相片
    cv2.imshow("Original | Processed", combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(input_path, output_path):
    """讀取圖片、進行處理、存圖片"""
    try:
        image = cv2.imread(input_path)  # 讀取圖片
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
        return True  # 表示成功處理

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")
        return False


def processing(img):
    """圖片處理"""
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉為灰階
    processed_img = cv2.Canny(processed_img, 100, 200)
    return processed_img


if __name__ == "__main__":
    main()
