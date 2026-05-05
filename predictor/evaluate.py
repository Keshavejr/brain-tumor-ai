import tensorflow as tf
import os

# ===== CONFIG =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 👉 CHANGE THIS PATH
TEST_DIR = r"C:\Users\HP\brain_api\dataset\test"

# 👉 MODEL PATHS
CUSTOM_MODEL_PATH = r"C:\Users\HP\brain_api\predictor\models\basic_cnn_model.keras"
EFF_MODEL_PATH    = r"C:\Users\HP\brain_api\predictor\models\efficientnet_model.keras"


# ===== LOAD DATASET =====
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ===== PREPROCESS =====
# Custom CNN
test_custom = test_data.map(lambda x, y: (x / 255.0, y))

# EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input
test_eff = test_data.map(lambda x, y: (preprocess_input(x), y))


# ===== LOAD MODELS =====
custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
eff_model    = tf.keras.models.load_model(EFF_MODEL_PATH)


# ===== FIX LOSS FOR EVALUATION =====
custom_model.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

eff_model.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ===== EVALUATE =====
print("\nEvaluating Custom CNN...")
custom_loss, custom_acc = custom_model.evaluate(test_custom)

print("\nEvaluating EfficientNet...")
eff_loss, eff_acc = eff_model.evaluate(test_eff)


# ===== RESULTS =====
print("\n===== FINAL RESULTS =====")
print(f"Custom CNN Accuracy: {custom_acc * 100:.2f}%")
print(f"EfficientNet Accuracy: {eff_acc * 100:.2f}%")