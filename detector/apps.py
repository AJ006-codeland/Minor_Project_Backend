import os
from django.apps import AppConfig
from django.conf import settings


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    model = None

    def ready(self):
        import tensorflow as tf

        model_path = os.path.join(
            settings.BASE_DIR, 'detector', 'ml_models', 'efficientnetv2_grape_final.keras'
        )
        print(f"Loading model from: {model_path}")
        DetectorConfig.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")