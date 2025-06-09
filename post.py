import requests

# ✅ NO incluyas el puerto :8080 en la URL final con HTTPS
url = "https://ocrticketing-production.up.railway.app/ocr"
file_path = r"C:\Users\carlo\Desktop\OCR\standalone app\Images\0000_20250529_170439.png"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
