from config.settings import BASE_DIR
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
from django.http import HttpResponse

import os
import tensorflow as tf
import numpy as np
import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
from predictor.download_models import download_models

download_models()

MODEL_PATH = os.path.join(BASE_DIR, "predictor", "models", "efficientnet_model.keras")
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(BASE_DIR, "predictor", "models", "efficientnet_model.keras")
        model = tf.keras.models.load_model(model_path)
    return model

CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "predictor", "models", "basic_cnn_model.keras")
custom_model = None

def get_custom_model():
    global custom_model
    if custom_model is None:
        custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    return custom_model

def make_gradcam_heatmap(img_array, model):

    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # STEP 1: forward model once
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.outputs
    )

    # STEP 2: get predictions
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)

        preds = grad_model(inputs, training=False)

        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, inputs)

    # ⚠️ fallback Grad-CAM (input-based approximation)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(inputs[0] * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img):

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

classes = ["glioma", "meningioma", "notumor", "pituitary"]

def home(request):
    return render(request, "index.html")

def preprocess_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # MUST match training

    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)
    return img

@api_view(['POST'])
def predict_tumor(request):
    image = request.FILES.get('image')

    if not image:
        return Response({"error": "No image uploaded"}, status=400)

    # Read image ONCE
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return Response({"error": "Invalid image"}, status=400)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    processed = cv2.resize(img_rgb, (224, 224))
    processed = processed.astype(np.float32)

    # ✅ GET MODEL TYPE FROM FRONTEND
    model_type = request.data.get("model_type", "efficient")

    # 🔥 MODEL SWITCH
    if model_type == "custom":
        processed = processed / 255.0
        processed = np.expand_dims(processed, axis=0)
        processed = tf.convert_to_tensor(processed, dtype=tf.float32)

        model = get_custom_model()
        model_name = "Custom CNN"

    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        processed = preprocess_input(processed)
        processed = np.expand_dims(processed, axis=0)
        processed = tf.convert_to_tensor(processed, dtype=tf.float32)

        model = get_model()
        model_name = "EfficientNet"

    # 🔥 Prediction
    preds = model.predict(processed)[0]
    class_index = np.argmax(preds)

    # 🔥 Grad-CAM
    heatmap = make_gradcam_heatmap(processed, model)
    overlay = overlay_heatmap(heatmap, img_rgb)

    # Save heatmap
    output_dir = os.path.join(BASE_DIR, "media")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "gradcam.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return Response({
        "model_used": model_name,
        "prediction": CLASS_NAMES[class_index],
        "confidence": round(float(preds[class_index]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
        },
        "gradcam_url": "/media/gradcam.jpg"
    })

def generate_pdf(request):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'

    doc = SimpleDocTemplate(response)
    styles = getSampleStyleSheet()

    content = []

    # Data from request
    prediction = request.GET.get("prediction", "N/A")
    confidence = request.GET.get("confidence", "N/A")

    # Paths
    gradcam_path = os.path.join(BASE_DIR, "media", "gradcam.jpg")

    # Title
    content.append(Paragraph("Brain Tumor Detection Report", styles['Title']))
    content.append(Spacer(1, 12))

    # Date
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    content.append(Spacer(1, 12))

    # Prediction info
    content.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles['Normal']))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence}%", styles['Normal']))
    content.append(Spacer(1, 20))

    # Grad-CAM image
    if os.path.exists(gradcam_path):
        content.append(Paragraph("Grad-CAM Visualization:", styles['Heading3']))
        content.append(Spacer(1, 10))

        img = Image(gradcam_path, width=4*inch, height=4*inch)
        content.append(img)
        content.append(Spacer(1, 20))

    # Footer note
    content.append(Paragraph(
        "Note: This is an AI-generated prediction and should not replace professional medical advice.",
        styles['Italic']
    ))

    doc.build(content)
    return response