from pathlib import Path
import io

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from ultralytics import YOLO

FOLDER = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(FOLDER), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# Downloads the model weights the first time the server starts.
model = YOLO("yolo11n.pt")
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}


@app.get("/")
def home():
    return send_from_directory(FOLDER, "index.html")


@app.post("/api/detect")
def detect():
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify(error="Choose a photo first."), 400

    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception:
        return jsonify(error="Could not read that image."), 400

    width, height = image.size
    columns = min(4, max(1, (width + 299) // 300))
    rows = min(4, max(1, (height + 299) // 300))
    tile_width = min(width, round(width / (columns - 0.2 * (columns - 1))))
    tile_height = min(height, round(height / (rows - 0.2 * (rows - 1))))
    detections = []

    for row in range(rows):
        y = 0 if rows == 1 else round(row * (height - tile_height) / (rows - 1))

        for column in range(columns):
            x = 0 if columns == 1 else round(
                column * (width - tile_width) / (columns - 1)
            )
            crop = image.crop((x, y, x + tile_width, y + tile_height))
            result = model.predict(
                crop, imgsz=960, conf=0.12, iou=0.5, verbose=False
            )[0]

            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                if label not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                candidate = {
                    "class": label,
                    "bbox": [x1 + x, y1 + y, x2 - x1, y2 - y1],
                }

                # Avoid counting the same vehicle twice in overlapping crops.
                bx, by, bw, bh = candidate["bbox"]
                duplicate = False
                for old in detections:
                    ax, ay, aw, ah = old["bbox"]
                    overlap = (
                        max(0, min(ax + aw, bx + bw) - max(ax, bx))
                        * max(0, min(ay + ah, by + bh) - max(ay, by))
                    )
                    if overlap / max(1, min(aw * ah, bw * bh)) > 0.65:
                        duplicate = True
                        break

                if not duplicate:
                    detections.append(candidate)

    return jsonify(width=width, height=height, detections=detections)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)