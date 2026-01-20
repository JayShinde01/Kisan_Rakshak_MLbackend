from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import threading
import logging
import os
from PIL import Image
import io

# -----------------------
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cropcare-backend")

app = Flask(__name__)
CORS(app)

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
input_shape = input_details[0]["shape"]   # [1, height, width, 3]
input_height, input_width = input_shape[1], input_shape[2]

# Thread lock (VERY IMPORTANT for TFLite)
lock = threading.Lock()

# Class names
class_names = [
    "Aphid", "Brown Rust", "Healthy", "Leaf Blight",
    "Mildew", "Mite", "Septoria", "Smut", "unknown", "Yellow Rust"
]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "tiff"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------
@app.route("/")
def home():
    return jsonify({"status": "API is running"})

# -----------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type",
            "allowed": list(ALLOWED_EXTENSIONS)
        }), 400

    try:
        # -------- Load image from memory (NO DISK)
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((input_width, input_height))

        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0).astype(input_details[0]["dtype"])

        # Normalize if model expects float input
        if np.issubdtype(input_details[0]["dtype"], np.floating):
            img_array = img_array / 255.0

        # -------- Thread-safe inference
        with lock:
            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]["index"])[0]

        idx = int(np.argmax(prediction))
        label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        confidence = round(float(prediction[idx]) * 100, 2)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
