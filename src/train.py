"""
train.py — train a small CNN classifier on the extracted log-mel features.

Designed to be small enough to convert to TFLite and run efficiently on a
phone NPU (a handful of conv layers, no huge dense layers).

Usage:
    python src/train.py
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import CLASSES, FEATURES_DIR, MODELS_DIR, KERAS_MODEL_PATH


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    x_path = os.path.join(FEATURES_DIR, "X.npy")
    y_path = os.path.join(FEATURES_DIR, "y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print("Features not found. Run src/features.py first.")
        return

    X = np.load(x_path)
    y = np.load(y_path)

    if len(np.unique(y)) < 2:
        print("Need at least 2 classes with samples to train. "
              "Add more .wav clips under data/samples/<class>/.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X.shape[1:], len(CLASSES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=2,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(KERAS_MODEL_PATH)
    print(f"\nModel saved to {KERAS_MODEL_PATH}")

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    main()
