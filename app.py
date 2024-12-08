import streamlit as st
import cv2
import torch
import os
from datetime import datetime
from PIL import Image

# Inisialisasi YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Asus/Downloads/file-yolo-test/yolov5/runs/train/exp4/weights/best.pt')

# Direktori untuk menyimpan gambar
save_folder = 'static/detect'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Dictionary solusi penyakit
disease_solutions = {
    0: {"name": "Daun Bercak", "solution": "Daun selada yang mengalami bercak sering kali disebabkan oleh ketidakseimbangan pH atau kekurangan nutrisi tertentu. ..."},
    1: {"name": "Daun Busuk", "solution": "Daun selada yang mengalami pembusukan dapat disebabkan oleh beberapa faktor, termasuk ketidakseimbangan pH dan kekurangan nutrisi tertentu. ..."},
    2: {"name": "Sehat", "solution": "Daun selada yang sehat menunjukkan bahwa kondisi pH dan ketersediaan nutrisi berada dalam keseimbangan yang optimal. ..."}
}

# Fungsi untuk mendeteksi gambar menggunakan YOLOv5
def detect_image(image):
    results = model(image)
    detections = results.xyxy[0]  # Ambil bounding box dan label kelas
    annotated_image = results.render()[0]  # Gambar dengan bounding box
    return annotated_image, detections

# Sidebar Streamlit
st.sidebar.title("YOLOv5 Lettuce Detection")
st.sidebar.markdown("Pilih sumber kamera dan mode operasi.")

# Pilihan kamera
camera_source = st.sidebar.radio(
    "Pilih Kamera:",
    options=["Webcam", "ESP CAM"],
    index=0
)

# URL ESP CAM
esp_cam_url = "http://192.168.18.223:81/stream"

# Tampilan halaman utama
st.title("YOLOv5 Lettuce Detection")
st.write("Aplikasi ini mendeteksi kondisi daun selada menggunakan model YOLOv5.")

# Pilihan untuk streaming kamera
if camera_source == "Webcam":
    cap = cv2.VideoCapture(0)
elif camera_source == "ESP CAM":
    cap = cv2.VideoCapture(esp_cam_url)

# Streaming kamera
frame_placeholder = st.empty()
capture_button = st.sidebar.button("Ambil Gambar")
detections_placeholder = st.empty()

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal membuka kamera.")
            break

        # Tampilkan frame kamera secara live
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Tindakan setelah tombol "Ambil Gambar" ditekan
        if capture_button:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(save_folder, f"capture_{timestamp}.jpg")

            # Deteksi YOLOv5
            annotated_image, detections = detect_image(frame)

            # Simpan gambar dengan bounding box
            cv2.imwrite(save_path, annotated_image)

            # Konversi ke RGB untuk ditampilkan di Streamlit
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Tampilkan hasil gambar dan bounding box
            st.image(annotated_image_rgb, caption="Hasil Deteksi dengan Bounding Box", use_column_width=True)

            # Analisis hasil deteksi
            st.subheader("Deteksi Penyakit:")
            if len(detections) > 0:
                for *xyxy, conf, cls in detections.tolist():
                    cls_id = int(cls)
                    if cls_id in disease_solutions:
                        solution = disease_solutions[cls_id]
                        st.success(f"Penyakit: {solution['name']}")
                        st.write(f"Solusi: {solution['solution']}")
            else:
                st.info("Tidak ada penyakit terdeteksi.")
            break

    cap.release()
else:
    st.error("Kamera tidak dapat diakses.")
