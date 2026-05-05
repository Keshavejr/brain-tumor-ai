import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# PATHS
# =========================
train_dir = "dataset/train"
test_dir = "dataset/test"

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# =========================
# DATA GENERATORS
# =========================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=12,
    zoom_range=0.15,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    classes=classes
)

val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    classes=classes
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False,
    classes=classes
)

# =========================
# MODEL
# =========================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, verbose=1),
    ModelCheckpoint("predictor/models/efficientnet_model.keras",
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1)
]

# =========================
# TRAIN PHASE 1
# =========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=12,
    callbacks=callbacks,
    class_weight={
        0: 1.3,
        1: 1.5,
        2: 1.0,
        3: 1.0
    }
)

# =========================
# FINE-TUNING
# =========================
base_model.trainable = True

for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks,
    class_weight={
        0: 2,
        1: 2,
        2: 1.0,
        3: 1.0
    }
)

# =========================
# EVALUATION
# =========================
model = tf.keras.models.load_model("predictor/models/efficientnet_model.keras")

test_loss, test_acc = model.evaluate(test_data)
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

