import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.apps import apps

from .segmentation import get_largest_leaf_crop
from .models import PredictionHistory

CLASS_NAMES = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]


def preprocess_image_array(cv2_image, target_size=(224, 224)):
    """Takes a numpy BGR image (from OpenCV) and prepares it for the model."""
    img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_array = np.array(img_resized, dtype=np.float32)  # stays 0-255
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array


@api_view(['POST'])
def predict_disease(request):
    if 'image' not in request.FILES:
        return Response({'error': 'No image file provided. Send it as multipart/form-data with key "image".'}, status=400)

    uploaded_file = request.FILES['image']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        leaf_crop = get_largest_leaf_crop(tmp_path)

        if leaf_crop is None:
            return Response({'error': 'No leaf detected in the image. Try a clearer photo.'}, status=422)

        input_array = preprocess_image_array(leaf_crop)

        detector_config = apps.get_app_config('detector')
        model = detector_config.model

        predictions = model.predict(input_array)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])

        probs = {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 4) for i in range(len(CLASS_NAMES))
        }

        history_entry = PredictionHistory(
            predicted_class=CLASS_NAMES[predicted_index],
            confidence=round(confidence, 4),
            black_rot_prob=probs.get("Black Rot", 0),
            esca_prob=probs.get("ESCA", 0),
            healthy_prob=probs.get("Healthy", 0),
            leaf_blight_prob=probs.get("Leaf Blight", 0),
        )
        uploaded_file.seek(0)
        history_entry.image.save(uploaded_file.name, uploaded_file, save=True)

        return Response({
            'predicted_class': CLASS_NAMES[predicted_index],
            'confidence': round(confidence, 4),
            'all_probabilities': probs
        })

    finally:
        os.remove(tmp_path)


@api_view(['GET'])
def get_history(request):
    history = PredictionHistory.objects.all()[:50]
    data = [
        {
            'id': entry.id,
            'predicted_class': entry.predicted_class,
            'confidence': entry.confidence,
            'black_rot_prob': entry.black_rot_prob,
            'esca_prob': entry.esca_prob,
            'healthy_prob': entry.healthy_prob,
            'leaf_blight_prob': entry.leaf_blight_prob,
            'image_url': request.build_absolute_uri(entry.image.url) if entry.image else None,
            'created_at': entry.created_at.isoformat(),
        }
        for entry in history
    ]
    return Response(data)