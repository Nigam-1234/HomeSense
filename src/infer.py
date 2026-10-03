"""
infer.py — run the trained model on a single .wav file, or continuously on
live microphone input, and print detected events with confidence.

This is the desktop stand-in for what the Android app will do on-device:
listen -> classify -> trigger action.

Usage:
    python src/infer.py --file path/to/clip.wav
    python src/infer.py --live          # requires a working microphone
"""
import argparse
import time

import numpy as np
import tensorflow as tf

from config import (
    CLASSES,
    SAMPLE_RATE,
    CLIP_DURATION_SEC,
    TFLITE_MODEL_PATH,
    EVENT_CONFIDENCE_THRESHOLD,
)
from features import load_and_window, extract_logmel


def load_interpreter(model_path=TFLITE_MODEL_PATH):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, feat):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = feat[np.newaxis, ..., np.newaxis].astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    return probs


def trigger_action(label, confidence):
    """Placeholder for the real action layer (push notification, Bluetooth
    smart plug/light flash, etc). Wire this up to whatever you build for
    the live demo."""
    print(f"  -> ACTION: event='{label}' confidence={confidence:.2f} "
          f"(hook up notification / smart-plug trigger here)")


def run_on_file(path, interpreter):
    audio = load_and_window(path)
    feat = extract_logmel(audio)
    probs = predict(interpreter, feat)

    top_idx = int(np.argmax(probs))
    label = CLASSES[top_idx]
    confidence = float(probs[top_idx])

    print(f"File: {path}")
    print(f"Prediction: {label} ({confidence:.2%})")
    for cls, p in zip(CLASSES, probs):
        print(f"   {cls:15s} {p:.2%}")

    if confidence >= EVENT_CONFIDENCE_THRESHOLD and label != "background":
        trigger_action(label, confidence)


def run_live(interpreter):
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        return

    print("Listening live... (Ctrl+C to stop)")
    window_len = int(SAMPLE_RATE * CLIP_DURATION_SEC)

    try:
        while True:
            audio = sd.rec(window_len, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()

            feat = extract_logmel(audio)
            probs = predict(interpreter, feat)
            top_idx = int(np.argmax(probs))
            label = CLASSES[top_idx]
            confidence = float(probs[top_idx])

            if label != "background":
                print(f"[{time.strftime('%H:%M:%S')}] {label} ({confidence:.2%})")
                if confidence >= EVENT_CONFIDENCE_THRESHOLD:
                    trigger_action(label, confidence)
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to a .wav file to classify")
    parser.add_argument("--live", action="store_true", help="Classify live mic input")
    args = parser.parse_args()

    interpreter = load_interpreter()

    if args.file:
        run_on_file(args.file, interpreter)
    elif args.live:
        run_live(interpreter)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
