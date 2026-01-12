from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import time
import uuid
import logging
import numpy as np
import threading

# -----------------------
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cropcare-backend")

app = Flask(__name__)
CORS(app)

# Upload configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "tiff"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------
# Load TFLite model
MODEL_PATH = os.environ.get("MODEL_PATH", "model/best_efficientnet_model.tflite")
logger.info("📌 Loading TFLite model from: %s", MODEL_PATH)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("✅ TFLite model loaded successfully.")

# Model input shape
input_shape = input_details[0]['shape']  # e.g., [1, 256, 256, 3]
input_height, input_width = input_shape[1], input_shape[2]

# Thread lock for TFLite
lock = threading.Lock()

# Class names
class_names = [
    "Aphid", "Brown Rust", "Healthy", "Leaf Blight",
    "Mildew", "Mite", "Septoria", "Smut","unknown","Yellow Rust"
]

# -----------------------
@app.route("/")
def home():
    return jsonify({"status": "API is running"})

@app.route("/predict", methods=["POST"])
def web_predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")

    if not allowed_file(file.filename):
        return render_template("index.html", error="File type not allowed")

    orig_name = secure_filename(file.filename)
    unique_name = f"{os.path.splitext(orig_name)[0]}_{uuid.uuid4().hex}{os.path.splitext(orig_name)[1]}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    return _predict_response(filepath, render_html=True)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed", "allowed": list(ALLOWED_EXTENSIONS)}), 400

    orig_name = secure_filename(file.filename)
    timestamp = int(time.time())
    unique_name = f"{os.path.splitext(orig_name)[0]}_{timestamp}_{uuid.uuid4().hex}{os.path.splitext(orig_name)[1]}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    return _predict_response(filepath, render_html=False)

# -----------------------
def _predict_response(filepath: str, render_html: bool = False):
    try:
        # Load and preprocess image with model's input size
        img = load_img(filepath, target_size=(input_height, input_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0).astype(input_details[0]['dtype'])

        # Normalize if model expects float input
        if np.issubdtype(input_details[0]['dtype'], np.floating):
            img_array = img_array / 255.0

        # Thread-safe inference
        with lock:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get top-1 prediction
        idx = int(prediction.argmax())
        label = class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
        confidence = round(float(prediction[idx]) * 100, 2)

        # File URL
        file_url = url_for('static', filename=f"uploads/{os.path.basename(filepath)}", _external=True)

        if render_html:
            return render_template(
                "index.html",
                file_path=file_url,
                result=label,
                confidence=confidence
            )

        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "image_url": file_url
        })

    except Exception as e:
        logger.exception("Prediction error for file %s", filepath)
        if render_html:
            return render_template("index.html", error=f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
