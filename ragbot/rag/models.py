from django.db import models

# Create your models here.
class Documentizer(models.Model):
    name = models.CharField(max_length = 255)
    file = models.FileField(upload_to="pdf/")
    vector = models.JSONField(null=True)
    content = models.TextField()
