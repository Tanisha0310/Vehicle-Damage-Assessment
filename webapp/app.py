from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("../damage_type_model/runs/detect/train/weights/best.pt")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    results = model.predict(
        source=filepath,
        save=True,
        conf=0.25
    )

    latest_predict = sorted(
        os.listdir("runs/detect")
    )[-1]

    predicted_image = os.path.join(
        "runs/detect",
        latest_predict,
        file.filename
    )

    output_path = os.path.join(
        RESULT_FOLDER,
        file.filename
    )

    shutil.copy(predicted_image, output_path)

    best_detection = None

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if best_detection is None or conf > best_detection["confidence"] / 100:
                best_detection = {
                    "class": model.names[cls],
                    "confidence": round(conf * 100, 2)
                }

    detections = [best_detection] if best_detection else []

    return render_template(
        "result.html",
        image=file.filename,
        detections=detections
    )

if __name__ == "__main__":
    app.run(debug=True)