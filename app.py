import cv2
import numpy as np
import requests

# 圖片網址
url = "https://drive.google.com/file/d/14GG1cW7ZuoFJGPqnU7CINCmqL75s32AB/view?usp=drive_link"

# 發送 HTTP GET 請求取得圖片
response = requests.get(url, stream=True)
response.raise_for_status()  # 確保請求成功

# 讀取圖片數據
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

# 解析圖片
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# 顯示圖片
cv2.imshow("Remote Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
