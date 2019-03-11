from django.db import models

# Create your models here.

class Input(models.Model):
    sentence = models.CharField(max_length=200)