from django.db import models


class PredictionHistory(models.Model):
    image = models.ImageField(upload_to='history_images/', blank=True, null=True)
    predicted_class = models.CharField(max_length=50)
    confidence = models.FloatField()
    black_rot_prob = models.FloatField(default=0)
    esca_prob = models.FloatField(default=0)
    healthy_prob = models.FloatField(default=0)
    leaf_blight_prob = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']  # newest first

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.2%}) - {self.created_at}"