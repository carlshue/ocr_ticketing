import requests

url = "http://localhost:8000/ocr"
file_path = r"C:\Users\carlo\Desktop\OCR\standalone app\Images\0.png"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
