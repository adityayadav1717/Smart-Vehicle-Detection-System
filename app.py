from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

model = YOLO("yolov8n.pt")      # YOLO model

current_count = 0               # Vehicle count
cluster_level = "Waiting..."    # Cluster level
video_path = None               # Uploaded video path
processing = False              # Processing flag


def get_cluster_level(count):
    if count <= 15:
        return "Low Traffic"
    elif count <= 30:
        return "Medium Traffic"
    else:
        return "High Traffic"


def generate_frames():
    global current_count, cluster_level, processing

    cap = cv2.VideoCapture(video_path)

    while processing:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)

        vehicle_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:  # vehicles
                    vehicle_count += 1

        current_count = vehicle_count
        cluster_level = get_cluster_level(vehicle_count)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    global video_path

    file = request.files['video']
    upload_folder = "uploads"

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    video_path = os.path.join(upload_folder, file.filename)
    file.save(video_path)

    return jsonify({"status": "success"})


@app.route("/start_processing")
def start_processing():
    global processing
    processing = True
    return jsonify({"status": "started"})


@app.route("/stop_processing")
def stop_processing():
    global processing
    processing = False
    return jsonify({"status": "stopped"})


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/get_count")
def get_count():
    return jsonify({"count": current_count})


@app.route("/get_cluster")
def get_cluster():
    return jsonify({"cluster": cluster_level})


if __name__ == "__main__":
    app.run(debug=True)


