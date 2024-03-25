from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('approve/', views.approve, name='approve'),
    path('adminlogin/', views.adminlogin, name='adminlogin'),
    path('logout/', views.logout, name='logout'),
]