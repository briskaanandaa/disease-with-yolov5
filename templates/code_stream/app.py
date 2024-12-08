from flask import Flask, render_template, request, jsonify
import torch
import io
import cv2
import numpy as np
from PIL import Image
import base64
import requests
from io import BytesIO

app = Flask(__name__)

# Load model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Asus/Downloads/file-yolo-test/yolov5/runs/train/exp4/weights/best.pt')

# Pemetaan indeks ke solusi penyakit
disease_solutions = {
    0: {"name": "Daun Bercak", "solution": "Solusi untuk penyakit daun bercak."},
    1: {"name": "Daun Busuk", "solution": "Solusi untuk penyakit daun busuk."},
    2: {"name": "Sehat", "solution": "Daun sehat, tidak ada masalah."}
}

# Fungsi untuk menggambar bounding box
def draw_bounding_boxes(img, detections):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for obj in detections:
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        label = f"{obj['name']} ({obj['confidence']:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Endpoint untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk prediksi gambar yang diunggah
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada gambar yang diunggah.'})
    
    image_file = request.files['image']
    img = Image.open(image_file.stream)

    # Deteksi dengan YOLOv5
    results = model(img)
    detections = results.pandas().xywh[0].to_dict(orient="records")

    # Proses hasil deteksi dan gambar
    img_with_boxes = draw_bounding_boxes(img, detections)

    # Menyandikan gambar ke base64 untuk ditampilkan
    buffered = BytesIO()
    img_with_boxes.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Menghasilkan hasil deteksi
    result_data = []
    for det in detections:
        disease = disease_solutions.get(det['class'], {"name": "Tidak Dikenal", "solution": "Solusi tidak tersedia."})
        result_data.append({
            'name': disease['name'],
            'solution': disease['solution']
        })

    return jsonify({'image': img_base64, 'detections': result_data})

# Endpoint untuk menangani streaming ESP CAM
@app.route('/capture')
def capture():
    # Ambil gambar dari ESP CAM (gunakan IP yang sesuai)
    esp_cam_url = "http://192.168.18.223:81/capture"  # Gantilah sesuai dengan IP ESP
    response = requests.get(esp_cam_url)

    if response.status_code == 200:
        # Proses gambar
        img = Image.open(BytesIO(response.content))

        # Deteksi dengan YOLOv5
        results = model(img)
        detections = results.pandas().xywh[0].to_dict(orient="records")

        # Proses gambar hasil deteksi dengan menggambar bounding box
        img_with_boxes = draw_bounding_boxes(img, detections)

        # Menyandikan gambar ke base64 untuk ditampilkan
        buffered = BytesIO()
        img_with_boxes.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Menghasilkan hasil deteksi
        result_data = []
        for det in detections:
            disease = disease_solutions.get(det['class'], {"name": "Tidak Dikenal", "solution": "Solusi tidak tersedia."})
            result_data.append({
                'name': disease['name'],
                'solution': disease['solution']
            })

        return jsonify({'image': img_base64, 'detections': result_data})
    else:
        return jsonify({'error': 'Gagal mengambil gambar dari ESP CAM.'})

if __name__ == '__main__':
    app.run(debug=True)
