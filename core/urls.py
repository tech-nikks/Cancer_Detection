from django.urls import path
from . import views

urlpatterns = [
    
    path('', views.home, name='home'),
    path('menu/', views.menu, name='menu'),
    path('menu/brain/', views.brain, name='brain'),
    path('menu/lung/', views.lung, name='lung'),
    path('menu/kidney/', views.breast_cancer_detection, name='kidney'),
    path('menu/breast', views.breast, name="breast"),
    

]

