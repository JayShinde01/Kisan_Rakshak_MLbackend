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
# Logging
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

input_shape = input_details[0]["shape"]
input_height, input_width = input_shape[1], input_shape[2]
input_dtype = input_details[0]["dtype"]

logger.info("✅ Model loaded")
logger.info("📌 Input shape: %s | dtype: %s", input_shape, input_dtype)

# -----------------------
# Thread lock
lock = threading.Lock()

# -----------------------
# Classes
class_names = [
    "Aphid", "Brown Rust", "Healthy", "Leaf Blight",
    "Mildew", "Mite", "Septoria", "Smut", "unknown", "Yellow Rust"
]

# -----------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "tiff"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------
@app.route("/")
def home():
    return jsonify({"status": "API running 🚀"})

# -----------------------
@app.route("/api/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # -----------------------
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize (match model)
        image = image.resize((input_width, input_height))

        # Convert to array
        img_array = np.array(image)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # -----------------------
        # 🔥 CORRECT PREPROCESSING (EfficientNet)
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_array = preprocess_input(img_array.astype(np.float32))

        # -----------------------
        # Handle dtype
        if input_dtype == np.uint8:
            # Quantized model → scale back
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.float32)

        # -----------------------
        # Inference (thread-safe)
        with lock:
            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]["index"])[0]

        # -----------------------
        # Result
        idx = int(np.argmax(prediction))
        label = class_names[idx]
        confidence = round(float(prediction[idx]) * 100, 2)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        logger.exception("❌ Prediction failed")
        return jsonify({"error": str(e)}), 500

# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)