import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_text_cv2(img, text, pos, font_path='C:/Windows/Fonts/msjh.ttc', font_size=24, color=(0,0,255)):
    # img 是 OpenCV 影像 (BGR)
    # 先轉成 PIL Image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=color[::-1])  # PIL用RGB，OpenCV用BGR，需翻轉顏色

    # 再轉回 OpenCV 影像 (BGR)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def draw_trapezoid(frame):
    h, w = frame.shape[:2]
    pts = np.array([
        [int(w*0.3), int(h*0.3)],  
        [int(w*0.7), int(h*0.3)],  
        [int(w*0.95), int(h*0.95)], 
        [int(w*0.05), int(h*0.95)]  
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影像")
            break

        frame = draw_trapezoid(frame)
        # 用PIL畫中文
        frame = draw_text_cv2(frame, "請將路面調整到紅色梯形框內", (30, 30), font_size=32)

        cv2.imshow("初始安裝校正", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC鍵退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
