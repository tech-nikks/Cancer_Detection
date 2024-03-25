from django.db import models
from django.contrib.auth.models import User
#get user model
from django.contrib.auth import get_user_model
User = get_user_model()


# Create your models here.

class Doctor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    fname = models.CharField(max_length=100)
    lname = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    phone = models.CharField(max_length=100)
    registration_number = models.CharField(max_length=100,blank=True)
    specialization = models.CharField(max_length=100,blank=True)
    hospital_name = models.CharField(max_length=100,blank=True)
    hospital_address = models.CharField(max_length=100,blank=True)
    hospital_phone = models.CharField(max_length=100,blank=True)
    hospital_email = models.EmailField(max_length=100,blank=True)
    hospital_pincode = models.CharField(max_length=100,blank=True)
    hospital_city = models.CharField(max_length=100,blank=True)
    verify = models.BooleanField(default=False)

    def __str__(self):
        return self.fname + ' ' + self.lname + '|' + self.registration_number
    
class Adminn(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    admin_flag = models.BooleanField(default=True)





