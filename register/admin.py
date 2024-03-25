from django.contrib import admin

# Register your models here.
from .models import Doctor, Adminn
admin.site.register(Doctor)
admin.site.register(Adminn)
