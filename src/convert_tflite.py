"""
convert_tflite.py — export the trained Keras model to TFLite, suitable for
on-device inference via Android's NNAPI delegate (Snapdragon NPU) on the
iQOO 15.

Usage:
    python src/convert_tflite.py
"""
import os
import tensorflow as tf

from config import KERAS_MODEL_PATH, TFLITE_MODEL_PATH, MODELS_DIR


def main():
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"No trained model found at {KERAS_MODEL_PATH}. Run src/train.py first.")
        return

    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Dynamic range quantization: smaller model, faster NPU/CPU inference,
    # minimal accuracy loss — a good default for a hackathon demo.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_MODEL_PATH) / 1024
    print(f"TFLite model saved to {TFLITE_MODEL_PATH} ({size_kb:.1f} KB)")
    print("\nNext: load this .tflite file in the Android app via the")
    print("NNAPI delegate to run inference on the Snapdragon NPU.")


if __name__ == "__main__":
    main()
