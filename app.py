import subprocess
import sys

# 取得目前虛擬環境的 Python 執行路徑
python_path = sys.executable
# 設定要執行的腳本路徑
img_processing_path = "img_processing/main.py"

try:
    subprocess.run([python_path, img_processing_path], check=True)
    print(f"✅ {img_processing_path} 執行成功")
except subprocess.CalledProcessError as e:
    print(f"❌ {img_processing_path} 執行失敗：{e}")
