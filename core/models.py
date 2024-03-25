from django.db import models


# Create your models here.

from datetime import datetime
import uuid

# Create your models here.

class Scan(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    #store the image in the media folder
    image = models.ImageField(upload_to='images/', blank=True)
    Patient_Name = models.CharField(max_length=100)
    date = models.DateTimeField(default=datetime.now)
    result = models.CharField(max_length=100,blank=True)

