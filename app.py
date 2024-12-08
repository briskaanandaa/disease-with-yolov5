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

# Variable global untuk menyimpan nama file gambar terakhir yang diambil
last_image_path = None
detected_disease = None

# Disease solutions dictionary
disease_solutions = {
    0: {"name": "Daun Bercak", "solution": "Daun selada yang mengalami bercak sering kali disebabkan oleh ketidakseimbangan pH atau kekurangan nutrisi tertentu. ..."},
    1: {"name": "Daun Busuk", "solution": "Daun selada yang mengalami pembusukan dapat disebabkan oleh beberapa faktor, termasuk ketidakseimbangan pH dan kekurangan nutrisi tertentu. ..."},
    2: {"name": "Sehat", "solution": "Daun selada yang sehat menunjukkan bahwa kondisi pH dan ketersediaan nutrisi berada dalam keseimbangan yang optimal. ..."}
}

# Function to capture frames from webcam and ESP CAM
def gen_stream_frames(url=None):
    global last_image_path
    # Open webcam stream or ESP CAM stream
    if url:
        camera = cv2.VideoCapture(url)  # Use ESP CAM stream URL here
    else:
        camera = cv2.VideoCapture(0)  # Use webcam stream

    while True:
        # Capture frame from the camera
        success, frame = camera.read()
        if not success:
            break

        # Run YOLOv5 detection on the frame
        results = model(frame)
        frame = results.render()[0]  # Render bounding boxes on the frame

        # Encode the frame to be streamed
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global detected_disease
    return render_template('index.html', image_path=last_image_path, esp_cam_url="http://192.168.18.223:81/stream", detected_disease=detected_disease, disease_solutions=disease_solutions)

@app.route('/video_feed')
def video_feed():
    return Response(gen_stream_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/esp_cam_feed')
def esp_cam_feed():
    esp_cam_url = "http://192.168.18.223:81/stream"  # Replace with your ESP CAM URL
    return Response(gen_stream_frames(url=esp_cam_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture/<device>')
def capture(device):
    global last_image_path, detected_disease
    # Capture from webcam or ESP CAM based on the device
    if device == 'webcam':
        camera = cv2.VideoCapture(0)
    elif device == 'esp_cam':
        camera = cv2.VideoCapture("http://192.168.18.223:81/stream")  # Replace with your ESP CAM stream URL

    success, frame = camera.read()
    if success:
        # Perform detection with YOLOv5
        results = model(frame)
        frame = results.render()[0]

        # Save the captured image with a unique name based on timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = os.path.join(save_folder, f"capture_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)

        # Save the path of the last captured image
        last_image_path = image_path

        # Detect disease category based on YOLOv5 output
        if len(results.xyxy[0]) > 0:
            confidence = float(results.xyxy[0][0, 4].item())  # Confidence score of the first detected object
            if confidence < 0.25:
                detected_disease = "No Disease Detected"
            else:
                detected_disease = disease_solutions.get(int(results.xyxy[0][0, 5].item()), {}).get("name", "Unknown Disease")

    return redirect(url_for('index'))

@app.route('/generate_solution')
def generate_solution():
    global detected_disease, disease_solutions
    solution = disease_solutions.get(detected_disease, {}).get("solution", "No solution available.")
    flash(f"Detected disease: {detected_disease}. Solution: {solution}", "success")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
