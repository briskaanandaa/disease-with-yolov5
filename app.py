from flask import Flask, render_template, Response, redirect, url_for, flash
import cv2
import torch
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'secret_key'  # Untuk flash messages

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Asus/Downloads/file-yolo-test/yolov5/runs/train/exp4/weights/best.pt')

# Folder untuk menyimpan gambar
save_folder = 'static/detect'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Dictionary solusi penyakit
disease_solutions = {
    0: {"name": "Daun Bercak", "solution": "Daun selada yang mengalami bercak sering kali disebabkan oleh ketidakseimbangan pH atau kekurangan nutrisi tertentu. ..."},
    1: {"name": "Daun Busuk", "solution": "Daun selada yang mengalami pembusukan dapat disebabkan oleh beberapa faktor, termasuk ketidakseimbangan pH dan kekurangan nutrisi tertentu. ..."},
    2: {"name": "Sehat", "solution": "Daun selada yang sehat menunjukkan bahwa kondisi pH dan ketersediaan nutrisi berada dalam keseimbangan yang optimal. ..."}
}

# Variable global untuk menyimpan path gambar terakhir dan hasil deteksi
last_image_path = None
last_detections = []


def gen_stream_frames(source):
    try:
        camera = cv2.VideoCapture(source)
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # YOLOv5 detection
            results = model(frame)
            frame = results.render()[0]  # Render frame dengan deteksi
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error streaming: {e}")


@app.route('/')
def index():
    global last_image_path, last_detections, disease_solutions
    return render_template(
        'index.html',
        image_path=last_image_path,
        detections=last_detections,
        disease_solutions=disease_solutions,
        esp_cam_url="http://192.168.18.223:81/stream"
    )


@app.route('/video_feed')
def video_feed():
    return Response(gen_stream_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/esp_cam_feed')
def esp_cam_feed():
    return Response(gen_stream_frames("http://192.168.18.223:81/stream"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture/<device>')
def capture(device):
    global last_image_path, last_detections
    # Tentukan sumber video (webcam atau ESP CAM)
    source = 0 if device == 'webcam' else "http://192.168.18.223:81/stream"
    camera = cv2.VideoCapture(source)
    success, frame = camera.read()
    
    if success:
        # YOLOv5 detection
        results = model(frame)
        frame = results.render()[0]
        
        # Simpan gambar dengan timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = os.path.join(save_folder, f"capture_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        last_image_path = image_path  # Simpan path gambar terakhir

        # Analisis hasil deteksi
        detected_classes = results.xyxy[0][:, -1].tolist()  # Daftar label kelas
        last_detections = []
        for cls in detected_classes:
            cls_id = int(cls)
            if cls_id in disease_solutions:
                last_detections.append(disease_solutions[cls_id])
        
        if last_detections:
            flash("Deteksi selesai. Lihat hasil di bawah.", "success")
        else:
            flash("Tidak ada penyakit yang terdeteksi pada gambar.", "info")
    else:
        flash("Gagal mengambil gambar. Pastikan kamera tersedia.", "error")
    
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
