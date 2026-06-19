from flask import Flask, render_template_string, request, jsonify, Response
from ultralytics import YOLO
import cv2
import os
import uuid
import shutil
import time

app = Flask(__name__)

# =========================
# MODEL PATHS
# =========================
DEFAULT_MODELS = {
    "pedestrian": r"E:\Summer_internship_26-Object_detection_using_dashboard\yolo_pedestrian\best.pt",
    "rickshaw": r"E:\Summer_internship_26-Object_detection_using_dashboard\yolo_rickshaw\final\Rickshaw_Model\rickshaw_yolo11s_25epoch\weights\best.pt",
    "2_wheeler": r"E:\Summer_internship_26-Object_detection_using_dashboard\yolo_pedestrian\yolov8s_e150_best.pt",
    "speed_breaker": r"E:\Summer_internship_26-Object_detection_using_dashboard\yolo_bump\best.pt",
    "car": r"E:\Summer_internship_26-Object_detection_using_dashboard\yolo_car\best.pt"
}

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
DATABASE_DIR = "database"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

models = {}
latest_frame = None
progress_data = {"progress": 0, "status": "Idle", "output": ""}

COLORS = {
    "pedestrian": (0, 255, 0),
    "rickshaw": (255, 0, 0),
    "2_wheeler": (0, 255, 255),
    "speed_breaker": (0, 0, 255),
    "car": (255, 165, 0)
}


# =========================
# LOAD MODEL
# =========================
def get_model(name, custom_model_path=None):
    if custom_model_path and os.path.exists(custom_model_path):
        return YOLO(custom_model_path)

    if name not in models:
        path = DEFAULT_MODELS.get(name)
        if path and os.path.exists(path):
            models[name] = YOLO(path)
        else:
            raise FileNotFoundError(f"Model not found for {name}: {path}")

    return models[name]


# =========================
# DRAW LABEL WITH PADDING
# =========================
def draw_box_with_label(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad_x = 6
    pad_y = 5

    label_x1 = x1
    label_y1 = max(0, y1 - text_h - baseline - pad_y * 2)
    label_x2 = min(frame.shape[1], x1 + text_w + pad_x * 2)
    label_y2 = y1

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)

    text_x = label_x1 + pad_x
    text_y = label_y2 - pad_y - baseline

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# =========================
# SAVE DETECTION
# =========================
def save_detection(frame, detection_name, database_video_folder, frame_count, box, total_saved, limit):
    if total_saved[detection_name] >= limit:
        return

    image_folder = os.path.join(database_video_folder, detection_name, "images")
    label_folder = os.path.join(database_video_folder, detection_name, "labels")

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    img_name = f"{detection_name}_{frame_count}.jpg"
    label_name = f"{detection_name}_{frame_count}.txt"

    img_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder, label_name)

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    cv2.imwrite(img_path, frame)

    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    total_saved[detection_name] += 1


# =========================
# CLEAN IMAGE/LABEL MISMATCH
# =========================
def clean_database_folder(folder):
    for root, dirs, files in os.walk(folder):
        if root.endswith("images"):
            image_folder = root
            label_folder = root.replace("images", "labels")

            if not os.path.exists(label_folder):
                continue

            images = {
                os.path.splitext(f)[0]: os.path.join(image_folder, f)
                for f in os.listdir(image_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            }

            labels = {
                os.path.splitext(f)[0]: os.path.join(label_folder, f)
                for f in os.listdir(label_folder)
                if f.lower().endswith(".txt")
            }

            for name, img_path in images.items():
                if name not in labels:
                    os.remove(img_path)

            for name, lbl_path in labels.items():
                if name not in images:
                    os.remove(lbl_path)


# =========================
# PROCESS VIDEO
# =========================
def process_video(video_path, selected_detections, custom_model_path, custom_db_path):
    global latest_frame, progress_data

    job_id = str(uuid.uuid4())[:8]

    result_folder = os.path.join(OUTPUT_DIR, f"result_{job_id}")
    os.makedirs(result_folder, exist_ok=True)

    database_base = custom_db_path if custom_db_path else DATABASE_DIR
    os.makedirs(database_base, exist_ok=True)

    database_video_folder = os.path.join(database_base, f"video_{job_id}")
    os.makedirs(database_video_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        progress_data["status"] = "Video open failed"
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(result_folder, "output_video.mp4")

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0

    save_limit = 15 if custom_db_path else 999999
    total_saved = {name: 0 for name in selected_detections}

    loaded_models = {}

    for detection in selected_detections:
        if custom_model_path:
            loaded_models[detection] = get_model(detection, custom_model_path)
        else:
            loaded_models[detection] = get_model(detection)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        annotated_frame = frame.copy()

        for detection_name in selected_detections:
            model = loaded_models[detection_name]
            color = COLORS.get(detection_name, (255, 255, 255))

            results = model(frame, conf=0.7, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    label = f"{detection_name.upper()} {conf:.2f}"

                    draw_box_with_label(
                        annotated_frame,
                        x1, y1, x2, y2,
                        label,
                        color
                    )

                    save_detection(
                        frame,
                        detection_name,
                        database_video_folder,
                        frame_count,
                        (x1, y1, x2, y2),
                        total_saved,
                        save_limit
                    )

        out.write(annotated_frame)

        latest_frame = annotated_frame.copy()

        frame_count += 1
        progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
        progress_data["progress"] = progress
        progress_data["status"] = "Processing"

    cap.release()
    out.release()

    clean_database_folder(database_video_folder)

    progress_data["progress"] = 100
    progress_data["status"] = "Completed"
    progress_data["output"] = output_path


# =========================
# LIVE PREVIEW
# =========================
def generate_frames():
    global latest_frame

    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode(".jpg", latest_frame)

        if not ret:
            continue

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


# =========================
# HTML
# =========================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ADAS Dashboard</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f3f4f8;
        }

        .navbar {
            text-align: center;
            padding: 15px;
            font-size: 15px;
        }

        .navbar a {
            text-decoration: none;
            color: black;
            margin: 0 5px;
        }

        .main {
            display: flex;
            justify-content: center;
            gap: 18px;
            padding: 20px;
        }

        .card {
            background: white;
            width: 420px;
            min-height: 520px;
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }

        .preview-card {
            width: 420px;
        }

        h2 {
            margin-top: 10px;
        }

        label {
            font-size: 14px;
        }

        .checkbox-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin: 8px 0;
        }

        .checkbox-card {
            background: #f7f7ff;
            padding: 12px;
            border: 1px solid #d8d8e8;
            border-radius: 7px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        input[type="file"],
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-top: 6px;
            margin-bottom: 8px;
            border: 1px solid #ccc;
            border-radius: 7px;
            box-sizing: border-box;
        }

        .small-note {
            font-size: 12px;
            color: #444;
            margin-top: -5px;
            margin-bottom: 8px;
        }

        button {
            width: 100%;
            padding: 12px;
            background: #5b2be0;
            color: white;
            border: none;
            border-radius: 7px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #4720bd;
        }

        .progress-container {
            width: 100%;
            background: #e5e5e5;
            border-radius: 15px;
            margin-top: 12px;
            overflow: hidden;
        }

        .progress-bar {
            height: 20px;
            background: #20a843;
            width: 0%;
            color: white;
            text-align: center;
            font-size: 12px;
            font-weight: bold;
            line-height: 20px;
        }

        #status {
            margin-top: 12px;
            font-size: 14px;
        }

        .preview-box {
            margin-top: 25px;
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
            background: #111;
        }

        .preview-box img {
            width: 100%;
            display: block;
        }
    </style>
</head>
<body>

<div class="navbar">
    <a href="#">Pedestrian</a> |
    <a href="#">Rickshaw</a> |
    <a href="#">2-Wheeler</a> |
    <a href="#">Car</a> |
    <a href="#">Speed Breaker</a> |
    <a href="#">Lane Detection</a>
</div>

<div class="main">

    <div class="card">
        <h2>Upload Video</h2>

        <form id="uploadForm" enctype="multipart/form-data">

            <label>Select Detection:</label>

            <div class="checkbox-grid">
                <label class="checkbox-card">
                    <input type="checkbox" name="detections" value="pedestrian">
                    Pedestrian
                </label>

                <label class="checkbox-card">
                    <input type="checkbox" name="detections" value="rickshaw">
                    Rickshaw
                </label>

                <label class="checkbox-card">
                    <input type="checkbox" name="detections" value="2_wheeler">
                    2-Wheeler
                </label>

                <label class="checkbox-card">
                    <input type="checkbox" name="detections" value="speed_breaker">
                    Speed Breaker
                </label>

                <label class="checkbox-card">
                    <input type="checkbox" name="detections" value="car">
                    Car
                </label>

                <label class="checkbox-card">
                    <input type="checkbox" name="detections" value="lane">
                    Lane
                </label>
            </div>

            <label>Upload Video:</label>
            <input type="file" name="video" accept="video/*" required>

            <label>Optional Custom best.pt:</label>
            <input type="file" name="bestpt" accept=".pt">

            <div class="small-note">
                If empty, your default best.pt models will be used.
            </div>

            <label>Optional Database Save Path:</label>
            <input type="text" name="db_path" placeholder="Example: D:\\ADAS_Database">

            <div class="small-note">
                If custom path is given, only 15 detected frames will be saved.
            </div>

            <button type="submit">Start Detection</button>
        </form>

        <div class="progress-container">
            <div class="progress-bar" id="progressBar">0%</div>
        </div>

        <div id="status">Idle</div>
    </div>

    <div class="card preview-card">
        <h2>Live Preview</h2>
        <div class="preview-box">
            <img src="/video_feed">
        </div>
    </div>

</div>

<script>
document.getElementById("uploadForm").addEventListener("submit", function(e) {
    e.preventDefault();

    let formData = new FormData(this);

    fetch("/start", {
        method: "POST",
        body: formData
    });

    let interval = setInterval(function() {
        fetch("/progress")
        .then(response => response.json())
        .then(data => {
            let bar = document.getElementById("progressBar");
            bar.style.width = data.progress + "%";
            bar.innerText = data.progress + "%";

            document.getElementById("status").innerText = data.status;

            if (data.progress >= 100) {
                clearInterval(interval);
            }
        });
    }, 1000);
});
</script>

</body>
</html>
"""


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/start", methods=["POST"])
def start():
    global progress_data

    progress_data = {"progress": 0, "status": "Starting", "output": ""}

    video = request.files.get("video")
    custom_bestpt = request.files.get("bestpt")
    selected_detections = request.form.getlist("detections")
    db_path = request.form.get("db_path", "").strip()

    if not video:
        return jsonify({"error": "No video uploaded"}), 400

    if not selected_detections:
        return jsonify({"error": "No detection selected"}), 400

    if "lane" in selected_detections:
        selected_detections.remove("lane")

    video_path = os.path.join(UPLOAD_DIR, video.filename)
    video.save(video_path)

    custom_model_path = None

    if custom_bestpt and custom_bestpt.filename:
        custom_model_path = os.path.join(UPLOAD_DIR, f"custom_{uuid.uuid4().hex}.pt")
        custom_bestpt.save(custom_model_path)

    import threading
    thread = threading.Thread(
        target=process_video,
        args=(video_path, selected_detections, custom_model_path, db_path)
    )
    thread.start()

    return jsonify({"message": "Detection started"})


@app.route("/progress")
def progress():
    return jsonify(progress_data)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
