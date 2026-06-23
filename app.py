from flask import Flask, request, render_template_string, jsonify, send_from_directory, abort
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import os
import uuid
import threading
import zipfile
import matplotlib.pyplot as plt

app = Flask(__name__)

BASE_DIR = r"E:\Auto_labeling_database\website"
DEFAULT_DATABASE_DIR = os.path.join(BASE_DIR, "database")
os.makedirs(DEFAULT_DATABASE_DIR, exist_ok=True)

MODELS = {
    "pedestrian": r"E:\Auto_labeling_database\models\pedestrian.pt",
    "rickshaw": r"E:\Auto_labeling_database\models\rickshaw.pt",
    "two_wheel": r"E:\Auto_labeling_database\models\bike.pt",
    "car": r"E:\Auto_labeling_database\models\car.pt",
    "lane": r"E:\Auto_labeling_database\models\lane.pt",
}

OUT_WIDTH = 960
OUT_HEIGHT = 540
FRAME_SKIP = 6
CONF = 0.70
FREE_FRAME_LIMIT = 15

jobs = {}

print("Loading default models...")
DEFAULT_LOADED_MODELS = {}

for key, path in MODELS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    DEFAULT_LOADED_MODELS[key] = YOLO(path)

print("All default models loaded!")


HTML = """
<!DOCTYPE html>
<html>
<head>
<title>VisionLabeler</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>
*{box-sizing:border-box;}
body{margin:0;font-family:Arial,sans-serif;background:#f3f5fb;color:#222;}
.container{width:95%;max-width:1200px;margin:auto;padding:20px;}
.header{text-align:center;margin-bottom:20px;}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
.card{background:white;padding:20px;border-radius:16px;box-shadow:0 5px 20px rgba(0,0,0,0.08);}
input,button{width:100%;padding:13px;margin-top:10px;border-radius:10px;border:1px solid #ccc;font-size:15px;}
button{background:#5b2be0;color:white;border:none;cursor:pointer;font-weight:bold;}
button:hover{background:#4320aa;}

.preview-wrap{
    width:100%;
    height:320px;
    background:white;
    border:2px dashed #ccc;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    color:#777;
    font-size:18px;
    font-weight:bold;
    overflow:hidden;
}

.preview{
    width:100%;
    height:100%;
    object-fit:contain;
    display:none;
}

.progress{width:100%;background:#ddd;height:25px;border-radius:30px;overflow:hidden;margin-top:15px;}
#bar{width:0%;height:100%;background:#28a745;text-align:center;color:white;line-height:25px;font-size:13px;}
.stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;}
.stat{background:#eef0ff;padding:15px;border-radius:12px;text-align:center;}
.stat h3{margin:0;font-size:25px;color:#5b2be0;}
video,img{width:100%;border-radius:12px;margin-top:15px;}
a{display:inline-block;margin-top:10px;margin-right:8px;padding:10px 14px;background:#111;color:white;border-radius:8px;text-decoration:none;}
.folder{background:#f7f7f7;padding:10px;border-radius:8px;font-size:14px;word-break:break-all;}
.check-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;}
.check-item{background:#f7f7ff;padding:10px;border-radius:10px;border:1px solid #ddd;}
.check-item input{width:auto;margin-right:8px;}
small{color:#666;}
.premium-msg{margin-top:10px;color:#b45309;font-weight:bold;}
@media(max-width:768px){
    .grid{grid-template-columns:1fr;}
    .stats{grid-template-columns:1fr;}
    .container{width:100%;padding:12px;}
    .check-row{grid-template-columns:1fr;}
}
</style>
</head>

<body>
<div class="container">

<div class="header">
<h1>VisionLabeler</h1>
<p>Smart Video Detection and Auto Labeling Database</p>
<p>Pedestrian | Rickshaw | 2-Wheeler | Car | Lane Detection</p>
</div>

<div class="grid">

<div class="card">
<h2>Upload Video</h2>

<form id="uploadForm" enctype="multipart/form-data">

<label>Select Detection:</label>

<div class="check-row">
<label class="check-item"><input type="checkbox" name="detect_type[]" value="pedestrian"> Pedestrian</label>
<label class="check-item"><input type="checkbox" name="detect_type[]" value="rickshaw"> Rickshaw</label>
<label class="check-item"><input type="checkbox" name="detect_type[]" value="two_wheel"> 2-Wheeler</label>
<label class="check-item"><input type="checkbox" name="detect_type[]" value="car"> Car</label>
<label class="check-item"><input type="checkbox" name="detect_type[]" value="lane"> Lane</label>
</div>

<label>Upload Video:</label>
<input type="file" name="video" accept="video/*" required>

<label>Optional Custom Model Upload:</label>
<input type="file" name="custom_model" accept=".pt" multiple>

<small>
Free Plan: upload only one file named <b>best.pt</b>.<br>
Premium: upload optional model files with selected detection names, example:
<b>pedestrian.pt</b>, <b>rickshaw.pt</b>, <b>two_wheel.pt</b>, <b>car.pt</b>, <b>lane.pt</b>.<br>
If a selected model is not uploaded, your default model will be used.
</small>

<label>Optional Database Save Path:</label>
<input type="text" name="save_path" placeholder="Example: D:\\VisionLabeler_Database">

<small>
<b>Free Plan:</b> If you upload best.pt and select a custom database path, only 15 detected frames will be saved.
</small>

<label class="check-item" style="margin-top:10px;display:block;">
<input type="checkbox" name="premium_user" value="yes"> I have Premium subscription
</label>

<small>
Premium allows full video generation, multiple model files, and multiple detections.
</small>

<button type="submit">Start Detection</button>

</form>

<div class="progress">
<div id="bar">0%</div>
</div>

<p id="status">Waiting for upload...</p>
<p id="premiumMsg" class="premium-msg"></p>
</div>

<div class="card">
<h2>Live Preview</h2>
<div class="preview-wrap" id="previewWrap">
    <span id="previewText">Upload the video</span>
    <img id="preview" class="preview">
</div>
</div>

</div>

<div class="card" id="resultBox" style="display:none; margin-top:20px;">
<h2>Detection Statistics</h2>

<div class="stats">
<div class="stat"><h3 id="total">0</h3><p>Total Detections</p></div>
<div class="stat"><h3 id="frames">0</h3><p>Saved Frames</p></div>
<div class="stat"><h3 id="jobId">-</h3><p>Job ID</p></div>
</div>

<h3>Saved Folder</h3>
<div class="folder" id="jobFolder"></div>

<h3>Downloads</h3>
<a id="downloadOriginal" href="#" download>Download Original Video</a>
<a id="downloadCombined" href="#" download>Download Combined Output Video</a>
<a id="downloadJob" href="#" download>Download Full Job ZIP</a>

<h3>Detection Results</h3>
<div id="resultLinks"></div>

<h3>Combined Output Video Preview</h3>
<video id="outputVideo" controls></video>

<h3>Graph Preview</h3>
<img id="graphImg">

</div>

</div>

<script>
let currentJob = null;
let timer = null;

document.getElementById("uploadForm").addEventListener("submit", function(e){
    e.preventDefault();

    let selected = document.querySelectorAll('input[name="detect_type[]"]:checked');

    if(selected.length === 0){
        alert("Please select at least one detection.");
        return;
    }

    let formData = new FormData(this);

    document.getElementById("status").innerText = "Uploading...";
    document.getElementById("premiumMsg").innerText = "";
    document.getElementById("bar").style.width = "0%";
    document.getElementById("bar").innerText = "0%";
    document.getElementById("resultBox").style.display = "none";
    document.getElementById("resultLinks").innerHTML = "";

    document.getElementById("preview").removeAttribute("src");
    document.getElementById("preview").style.display = "none";
    document.getElementById("previewText").style.display = "block";
    document.getElementById("previewText").innerText = "Processing video...";

    fetch("/start", {
        method:"POST",
        body:formData
    })
    .then(res => res.json())
    .then(data => {
        if(data.error){
            alert(data.error);
            document.getElementById("status").innerText = "Waiting for upload...";
            document.getElementById("previewText").innerText = "Upload the video";
            return;
        }

        currentJob = data.job_id;
        document.getElementById("status").innerText = "Processing started...";
        timer = setInterval(checkProgress, 700);
    });
});

function checkProgress(){
    fetch("/progress/" + currentJob)
    .then(res => res.json())
    .then(data => {
        document.getElementById("bar").style.width = data.progress + "%";
        document.getElementById("bar").innerText = data.progress + "%";
        document.getElementById("status").innerText = data.status;

        if(data.premium_msg){
            document.getElementById("premiumMsg").innerText = data.premium_msg;
        }

        if(data.preview){
            document.getElementById("previewText").style.display = "none";
            document.getElementById("preview").style.display = "block";
            document.getElementById("preview").src = data.preview + "?t=" + Date.now();
        }

        if(data.done){
            clearInterval(timer);

            document.getElementById("resultBox").style.display = "block";

            document.getElementById("total").innerText = data.total || 0;
            document.getElementById("frames").innerText = data.frames || 0;
            document.getElementById("jobId").innerText = currentJob;
            document.getElementById("jobFolder").innerText = data.job_folder || "";

            if(data.original_video){
                document.getElementById("downloadOriginal").href = data.original_video;
            }

            if(data.combined_video){
                document.getElementById("downloadCombined").href = data.combined_video;
                document.getElementById("outputVideo").src = data.combined_video;
            }

            if(data.job_zip){
                document.getElementById("downloadJob").href = data.job_zip;
            }

            let html = "";

            if(data.results && data.results.length > 0){
                data.results.forEach(item => {
                    html += `<div class="folder">
                        <b>${item.type.toUpperCase()}</b><br>
                        Folder: ${item.folder}<br>
                        Detections: ${item.total}<br>
                        Frames Saved: ${item.frames}<br>
                        <a href="${item.graph}" download>Graph</a>
                        <a href="${item.labels_zip}" download>Labels ZIP</a>
                    </div><br>`;
                });

                if(data.results[0].graph){
                    document.getElementById("graphImg").src = data.results[0].graph + "?t=" + Date.now();
                }
            }

            document.getElementById("resultLinks").innerHTML = html;
        }
    });
}
</script>

</body>
</html>
"""


def write_log(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\\n")


def zip_folder(source_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                full_path = os.path.join(root, file)

                if os.path.abspath(full_path) == os.path.abspath(zip_path):
                    continue

                arcname = os.path.relpath(full_path, source_dir)
                zipf.write(full_path, arcname)


def save_yolo_label(label_path, cls_id, box, conf, width, height):
    x1, y1, x2, y2 = box

    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height

    with open(label_path, "a", encoding="utf-8") as f:
        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\\n")


def clean_image_label_pair(auto_image_dir, auto_label_dir):
    image_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(auto_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    label_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(auto_label_dir)
        if f.lower().endswith(".txt")
    }

    common = set(image_files.keys()) & set(label_files.keys())

    for base, filename in image_files.items():
        if base not in common:
            os.remove(os.path.join(auto_image_dir, filename))

    for base, filename in label_files.items():
        if base not in common:
            os.remove(os.path.join(auto_label_dir, filename))

    return len(common)


def get_model_for_detection(detect_type, custom_model_paths=None):
    if custom_model_paths and detect_type in custom_model_paths:
        return YOLO(custom_model_paths[detect_type])

    return DEFAULT_LOADED_MODELS[detect_type]


def get_color_and_text(detect_type, conf):
    if detect_type == "pedestrian":
        return (0, 255, 0), f"PEDESTRIAN {conf:.2f}"

    if detect_type == "rickshaw":
        return (255, 0, 0), f"RICKSHAW {conf:.2f}"

    if detect_type == "two_wheel":
        return (255, 255, 0), f"2-WHEELER {conf:.2f}"

    if detect_type == "car":
        return (0, 0, 255), f"CAR {conf:.2f}"

    if detect_type == "lane":
        return (0, 255, 255), f"LANE {conf:.2f}"

    return (255, 255, 255), f"{detect_type.upper()} {conf:.2f}"


def process_job(job_id, original_video_path, selected_detections, job_dir, custom_model_paths, frame_save_limit):
    results_dir = os.path.join(job_dir, "results")
    stats_dir = os.path.join(job_dir, "stats")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    detection_data = {}

    for detect_type in selected_detections:
        detection_dir = os.path.join(results_dir, detect_type)
        auto_image_dir = os.path.join(detection_dir, "auto_images")
        auto_label_dir = os.path.join(detection_dir, "auto_labels")

        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(auto_image_dir, exist_ok=True)
        os.makedirs(auto_label_dir, exist_ok=True)

        detection_data[detect_type] = {
            "folder": detection_dir,
            "auto_images": auto_image_dir,
            "auto_labels": auto_label_dir,
            "counts": [],
            "total": 0,
            "saved": 0
        }

    models = {}

    for detect_type in selected_detections:
        models[detect_type] = get_model_for_detection(detect_type, custom_model_paths)

    output_video_path = os.path.join(results_dir, "combined_output.mp4")
    preview_path = os.path.join(results_dir, "preview.jpg")

    cap = cv2.VideoCapture(original_video_path)

    if not cap.isOpened():
        jobs[job_id]["status"] = "Cannot open video"
        jobs[job_id]["done"] = True
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 25

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUT_WIDTH, OUT_HEIGHT)
    )

    frame_no = 0
    processed_frames = 0
    last_display_lines = []
    premium_limit_shown = False
    total_saved_free_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_no += 1
        frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT))

        display_lines = []

        if frame_no % FRAME_SKIP == 0:
            processed_frames += 1

            for detect_type in selected_detections:
                model = models[detect_type]
                current_count = 0

                frame_name = f"frame_{frame_no}.jpg"
                label_name = f"frame_{frame_no}.txt"

                image_path = os.path.join(detection_data[detect_type]["auto_images"], frame_name)
                label_path = os.path.join(detection_data[detect_type]["auto_labels"], label_name)

                results = model(frame, conf=CONF, imgsz=640, verbose=False)

                if detect_type == "lane" and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()

                    for mask in masks:
                        mask = cv2.resize(mask, (OUT_WIDTH, OUT_HEIGHT))

                        if mask.sum() < 300:
                            continue

                        current_count += 1
                        detection_data[detect_type]["total"] += 1

                        overlay = frame.copy()
                        overlay[mask > 0.25] = (0, 255, 0)
                        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                        with open(label_path, "a", encoding="utf-8") as f:
                            f.write("0 0.500000 0.500000 1.000000 1.000000 1.0000\\n")

                elif results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        current_count += 1
                        detection_data[detect_type]["total"] += 1

                        color, label_text = get_color_and_text(detect_type, conf)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                        cv2.putText(
                            frame,
                            label_text,
                            (x1, max(y1 - 10, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )

                        save_yolo_label(
                            label_path,
                            cls_id,
                            (x1, y1, x2, y2),
                            conf,
                            OUT_WIDTH,
                            OUT_HEIGHT
                        )

                detection_data[detect_type]["counts"].append(current_count)

                if current_count > 0:
                    display_lines.append(f"{detect_type.upper()}: {current_count}")

                    if frame_save_limit is None:
                        if os.path.exists(label_path):
                            cv2.imwrite(image_path, frame)
                            detection_data[detect_type]["saved"] += 1
                    else:
                        if total_saved_free_frames < frame_save_limit:
                            if os.path.exists(label_path):
                                cv2.imwrite(image_path, frame)
                                detection_data[detect_type]["saved"] += 1
                                total_saved_free_frames += 1
                        else:
                            if not premium_limit_shown:
                                jobs[job_id]["premium_msg"] = (
                                    "Free plan limit reached: 15 frames generated. "
                                    "Take Premium to generate the full video, upload multiple model files, "
                                    "and run multiple detections."
                                )
                                premium_limit_shown = True

            last_display_lines = display_lines

            cv2.imwrite(preview_path, frame)
            jobs[job_id]["preview"] = f"/preview/{job_id}/combined"

        cv2.rectangle(frame, (20, 20), (820, 120), (0, 0, 0), -1)

        if last_display_lines:
            y = 50
            for line in last_display_lines[:3]:
                cv2.putText(
                    frame,
                    line,
                    (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                y += 30
        else:
            cv2.putText(
                frame,
                "PROCESSING...",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2
            )

        out.write(frame)

        progress = int((frame_no / total_frames) * 100) if total_frames > 0 else 0
        jobs[job_id]["progress"] = min(progress, 99)
        jobs[job_id]["status"] = f"Combined detection processing frame {frame_no}/{total_frames}"

    cap.release()
    out.release()

    all_results = []
    total_detections = 0
    total_saved_frames = 0

    for detect_type in selected_detections:
        detection_dir = detection_data[detect_type]["folder"]
        auto_image_dir = detection_data[detect_type]["auto_images"]
        auto_label_dir = detection_data[detect_type]["auto_labels"]

        matched_count = clean_image_label_pair(auto_image_dir, auto_label_dir)

        graph_path = os.path.join(detection_dir, "stats.png")
        labels_zip_path = os.path.join(detection_dir, "auto_labels.zip")
        log_path = os.path.join(detection_dir, "processing_log.txt")

        counts = detection_data[detect_type]["counts"]

        plt.figure(figsize=(10, 5))

        if counts:
            plt.plot(range(1, len(counts) + 1), counts, marker="o")
        else:
            plt.plot([0], [0], marker="o")

        plt.title(f"{detect_type.upper()} Count Per Processed Frame")
        plt.xlabel("Processed Frame")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        zip_folder(auto_label_dir, labels_zip_path)

        write_log(log_path, f"Detection Type: {detect_type}")
        write_log(log_path, f"Confidence: {CONF}")
        write_log(log_path, f"Processed Frames: {processed_frames}")
        write_log(log_path, f"Total Detections: {detection_data[detect_type]['total']}")
        write_log(log_path, f"Matched Auto Images and Labels: {matched_count}")

        all_results.append({
            "type": detect_type,
            "folder": detection_dir,
            "total": detection_data[detect_type]["total"],
            "frames": matched_count,
            "graph": f"/file/{job_id}/results/{detect_type}/stats.png",
            "labels_zip": f"/file/{job_id}/results/{detect_type}/auto_labels.zip"
        })

        total_detections += detection_data[detect_type]["total"]
        total_saved_frames += matched_count

    job_zip_path = os.path.join(stats_dir, f"{job_id}_full_job.zip")
    zip_folder(job_dir, job_zip_path)

    jobs[job_id].update({
        "done": True,
        "progress": 100,
        "status": "Completed",
        "total": total_detections,
        "frames": total_saved_frames,
        "results": all_results,
        "combined_video": f"/file/{job_id}/results/combined_output.mp4",
        "job_zip": f"/file/{job_id}/stats/{job_id}_full_job.zip"
    })


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/start", methods=["POST"])
def start():
    selected_detections = request.form.getlist("detect_type[]")
    print("Selected detections:", selected_detections)

    if not selected_detections:
        return jsonify({"error": "No detection selected"}), 400

    video = request.files["video"]
    custom_models = request.files.getlist("custom_model")
    save_path = request.form.get("save_path", "").strip()
    is_premium = request.form.get("premium_user") == "yes"

    valid_custom_models = [
        model_file for model_file in custom_models
        if model_file and model_file.filename
    ]

    has_custom_model = len(valid_custom_models) > 0

    uploaded_file_names = [
        secure_filename(model_file.filename).lower()
        for model_file in valid_custom_models
    ]

    allowed_detection_files = [f"{d}.pt" for d in MODELS.keys()]

    if has_custom_model:
        if not is_premium:
            if len(valid_custom_models) > 1:
                return jsonify({
                    "error": "Free Plan allows only one model file. Upload only best.pt or take Premium."
                }), 400

            if uploaded_file_names[0] != "best.pt":
                return jsonify({
                    "error": "Free Plan allows only one file named best.pt."
                }), 400

        else:
            invalid_files = [
                file_name for file_name in uploaded_file_names
                if file_name not in allowed_detection_files
            ]

            if invalid_files:
                return jsonify({
                    "error": "Invalid model file name. Use only: " + ", ".join(allowed_detection_files)
                }), 400

            not_selected_files = [
                file_name for file_name in uploaded_file_names
                if os.path.splitext(file_name)[0] not in selected_detections
            ]

            if not_selected_files:
                return jsonify({
                    "error": "You uploaded model files for detections that are not selected: " + ", ".join(not_selected_files)
                }), 400

    if save_path and has_custom_model and not is_premium:
        if len(selected_detections) > 1:
            return jsonify({
                "error": "Free Plan allows only one detection when using custom best.pt and custom save path. Take Premium for multiple detections."
            }), 400

    database_root = save_path if save_path else DEFAULT_DATABASE_DIR
    os.makedirs(database_root, exist_ok=True)

    if save_path and has_custom_model and not is_premium:
        frame_save_limit = FREE_FRAME_LIMIT
    else:
        frame_save_limit = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:6]
    job_id = f"job_{timestamp}_{short_id}"

    job_dir = os.path.join(database_root, job_id)
    database_dir = os.path.join(job_dir, "database")
    models_dir = os.path.join(database_dir, "models")
    results_dir = os.path.join(job_dir, "results")
    stats_dir = os.path.join(job_dir, "stats")

    os.makedirs(database_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    video_filename = secure_filename(video.filename)
    original_video_path = os.path.join(database_dir, video_filename)
    video.save(original_video_path)

    custom_model_paths = {}

    if has_custom_model:
        if not is_premium:
            detect_type = selected_detections[0]
            model_save_path = os.path.join(models_dir, f"{detect_type}_best.pt")
            valid_custom_models[0].save(model_save_path)
            custom_model_paths[detect_type] = model_save_path

        else:
            for model_file in valid_custom_models:
                filename = secure_filename(model_file.filename).lower()
                detect_type = os.path.splitext(filename)[0]

                model_save_path = os.path.join(models_dir, filename)
                model_file.save(model_save_path)
                custom_model_paths[detect_type] = model_save_path

    initial_premium_msg = ""

    if save_path and has_custom_model and not is_premium:
        initial_premium_msg = (
            "Free Plan active: only 15 detected frames will be saved. "
            "Take Premium for full video generation, multiple model files, and multiple detections."
        )

    if is_premium:
        initial_premium_msg = (
            "Premium active: uploaded models will be used where provided; missing selected models will use your default models."
        )

    jobs[job_id] = {
        "progress": 0,
        "status": "Starting...",
        "done": False,
        "preview": "",
        "premium_msg": initial_premium_msg,
        "total": 0,
        "frames": 0,
        "job_folder": job_dir,
        "original_video": f"/file/{job_id}/database/{video_filename}",
        "results": [],
        "combined_video": "",
        "job_zip": ""
    }

    thread = threading.Thread(
        target=process_job,
        args=(
            job_id,
            original_video_path,
            selected_detections,
            job_dir,
            custom_model_paths,
            frame_save_limit
        ),
        daemon=True
    )

    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/progress/<job_id>")
def progress(job_id):
    return jsonify(
        jobs.get(
            job_id,
            {
                "status": "Invalid job",
                "done": True,
                "progress": 0,
                "preview": "",
                "premium_msg": "",
                "results": [],
                "combined_video": "",
                "job_zip": ""
            }
        )
    )


@app.route("/preview/<job_id>/<detect_type>")
def preview(job_id, detect_type):
    job_folder = jobs.get(job_id, {}).get("job_folder")

    if not job_folder:
        abort(404)

    if detect_type == "combined":
        preview_dir = os.path.join(job_folder, "results")
    else:
        preview_dir = os.path.join(job_folder, "results", detect_type)

    preview_path = os.path.join(preview_dir, "preview.jpg")

    if not os.path.exists(preview_path):
        abort(404)

    response = send_from_directory(preview_dir, "preview.jpg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


@app.route("/file/<job_id>/<path:filename>")
def serve_job_file(job_id, filename):
    job_folder = jobs.get(job_id, {}).get("job_folder")

    if not job_folder:
        abort(404)

    requested_path = os.path.abspath(os.path.join(job_folder, filename))
    safe_root = os.path.abspath(job_folder)

    if not requested_path.startswith(safe_root):
        abort(403)

    if not os.path.exists(requested_path):
        abort(404)

    return send_from_directory(
        os.path.dirname(requested_path),
        os.path.basename(requested_path)
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
