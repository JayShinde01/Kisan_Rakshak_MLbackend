from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import time
import uuid
import logging

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
# Use TensorFlow Lite model for low RAM
MODEL_PATH = os.environ.get("MODEL_PATH", "model/best_efficientnet_model.tflite")
logger.info("📌 Loading TFLite model from: %s", MODEL_PATH)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("✅ TFLite model loaded successfully.")

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


def _predict_response(filepath: str, render_html: bool = False):
    try:
        # Load and preprocess image (smaller size 224x224)
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = img_array.astype(input_details[0]['dtype'])

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = int(prediction.argmax())
        label = class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
        confidence = round(float(prediction[idx]) * 100, 2)

        file_url = f"{request.host_url.rstrip('/')}/{app.config['UPLOAD_FOLDER'].replace(os.path.sep, '/')}/{os.path.basename(filepath)}"

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False to save RAM
    app.run(host="0.0.0.0", port=port, debug=False)
