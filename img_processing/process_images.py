import cv2
import os

def process_image(input_path, output_path):
    """ 讀取圖片、進行處理、存圖片 """
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
    """ 圖片處理 """
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return processed_img