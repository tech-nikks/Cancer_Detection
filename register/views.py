from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from .models import Doctor, Adminn
#import messages
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from . import email_send




# Create your views here.

def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        if user is not None:
            doctor = Doctor.objects.get(user=user)
            if doctor.verify == True:
                auth_login(request, user)

                return redirect('home')

            else:
                messages.error(request, 'Please wait for approval')
                return render(request, 'register/login.html')
        else:
            messages.error(request, 'Invalid credentials')
            return render(request, 'register/login.html')
            

        
        
    return render(request, 'register/login.html')



def register(request):
    if request.method == 'POST':
        if 's1' in request.POST:
            fname = request.POST.get('fname')
            lname = request.POST.get('lname')
            email = request.POST.get('email')
            phone = request.POST.get('phone')
            password = request.POST.get('password')
            password1 = request.POST.get('password1')
            if User.objects.filter(username=email).exists():
                messages.error(request, 'Email already exists')
                return render(request, 'register/register.html')
            elif password != password1:
                messages.error(request, 'Passwords do not match')
                return render(request, 'register/register.html')
            else:
                user = User.objects.create_user(username=email, email=email, password=password)
                user.save()
                doctor = Doctor(user=user, fname=fname, lname=lname, email=email, phone=phone)
                doctor.save()
                return render(request, 'register/moreinfo.html',{'email':email})
        elif 's2' in request.POST:
            verify = request.POST.get('verify')
            registration_number = request.POST.get('rnumber')
            specialization = request.POST.get('specialization')
            hospital_name = request.POST.get('hname')
            hospital_address = request.POST.get('haddress')
            hospital_phone = request.POST.get('hphnum')
            hospital_email = request.POST.get('hemail')
            hospital_pincode = request.POST.get('hpin')
            hospital_city = request.POST.get('hcity')
            doctor = Doctor.objects.get(email=verify)
            doctor.registration_number = registration_number
            doctor.specialization = specialization
            doctor.hospital_name = hospital_name
            doctor.hospital_address = hospital_address
            doctor.hospital_phone = hospital_phone
            doctor.hospital_email = hospital_email
            doctor.hospital_pincode = hospital_pincode
            doctor.hospital_city = hospital_city
            doctor.save()
            return render(request, 'register/confirmpage.html')
    return render(request, 'register/register.html')
@login_required(login_url='/auth/adminlogin')
def approve(request):
    doctors = Doctor.objects.filter(verify=False)
    if request.method == 'POST':
        doctor_user = request.POST.get('doctor_user')
        doctor = Doctor.objects.get(email=doctor_user)
        user = User.objects.get(username=doctor_user)
        name = doctor.fname + ' ' + doctor.lname
        if 'verify' in request.POST:
            doctor.verify = True
            doctor.save()
            email_send.send_approval_email(doctor_user,name)
            return redirect('approve')
        elif 'reject' in request.POST:
            email_send.send_rejection_email(doctor_user,name)
            user.delete()
            return redirect('approve')
    return render(request, 'administration/approve.html',{'doctors':doctors})

def adminlogin(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        admin = Adminn.objects.get(user=user)
        if user is not None and admin is not None:
            auth_login(request, user)
            return redirect('approve')
        else:
            messages.error(request, 'Invalid credentials')
            return render(request, 'administration/adminlogin.html')
    return render(request, 'administration/adminlogin.html')

def logout(request):
    auth_logout(request)
    return redirect('login')