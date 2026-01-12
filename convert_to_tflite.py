import tensorflow as tf
import os

# Path to your current Keras model
keras_model_path = "model/best_efficientnet_model.keras"

# Path where TFLite model will be saved
tflite_model_path = "model/best_efficientnet_model.tflite"

# Make sure the folder exists
os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)

# Load Keras model
model = tf.keras.models.load_model(keras_model_path, compile=False)
print("✅ Keras model loaded successfully.")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print("✅ Model converted to TFLite.")

# Save TFLite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"✅ TFLite model saved at: {tflite_model_path}")
