from django.shortcuts import render
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from .models import *
import numpy as np
from tensorflow import keras
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from PIL import Image
import os
import uuid
from register.models import Doctor
import tensorflow 
import cv2
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image
import pydicom
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from core.forms import BreastCancerForm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import pydicom
from PIL import Image
import base64
import io
# Create your views here.





    

# Create your views here.
# @login_required(login_url='auth/login')
def home(request):
    # doctor = Doctor.objects.get(user=request.user)
    return render(request, 'home.html')
# @login_required(login_url='auth/login')
def menu(request):
    return render(request, 'scanner/menu.html')
# @login_required(login_url='auth/login')
# def brain(request):
#     if request.method == 'POST':
#         # user = request.user
#         Patient_Name = request.POST.get('patient-name')
#         file = request.FILES['mri-scan']
#         # scan = Scan(user=user, Patient_Name=Patient_Name)
#         # scan.save()
#         # id = scan.id
#         #save the image in the media folder with the id as the name and the extension
#         unique_filename = str(uuid.uuid4())
#         extension = file.name.split('.')[1]
#         str_file = unique_filename + '.' + extension
#         #save the image in the media folder with the id as the name and the extension
#         file = ContentFile(file.read())
#         default_storage.save('images/'+str(str_file), file)
#         # scan.image = 'images/'+str(str_file)

#         print(str_file)
    
#         # scan.save()
#         img_url = 'media/images/'+str(str_file)
#         # img_url = f"{settings.MEDIA_URL}images/{str_file}"
#         model = tensorflow.keras.models.load_model('brain_tumor.h5')
#         img = Image.open(img_url)
#         x = np.array(img.resize((128,128)))
#         if len(x.shape) == 2:
#             x = np.stack((x,) * 3, axis=-1)
#         x = x.reshape(1,128,128,3)
#         # Predict the class of the input image
#         res = model.predict(x)
#         class_idx = np.where(res == np.amax(res))[1][0]
#         class_labels = ["Tumor","Not a Tumor"]
#         class_name = class_labels[class_idx]
#         # scan.result = class_name
#         # scan.save()
#         print('Predicted class: ', class_name)
#         print(img_url)
        
#         # return render(request, 'scanner/result.html', {'img_url':img_url})
#         return render(request, 'scanner/result.html', {
#             'image_url': img_url,
#             'predicted_class': class_name
#         })


#     return render(request, 'scanner/brain.html')
def brain(request):
    if request.method == 'POST':
        file = request.FILES['mri-scan']
        unique_filename = str(uuid.uuid4())
        extension = file.name.split('.')[-1]  # Safer way to get the extension
        filename = f"{unique_filename}.{extension}"
        
        # Save the file and get the path
        saved_path = default_storage.save(f'images/{filename}', ContentFile(file.read()))
        
        # Full path for model loading
        full_path = os.path.join(settings.MEDIA_ROOT, saved_path)
        
        # URL for accessing via web
        img_url = f"{settings.MEDIA_URL}{saved_path}"

        # Load model and predict
        model = tensorflow.keras.models.load_model('brain_tumor.h5')
        img = Image.open(full_path)  # Use full path here
        x = np.array(img.resize((128, 128)))
        if len(x.shape) == 2:
            x = np.stack((x,) * 3, axis=-1)
        x = x.reshape(1, 128, 128, 3)
        res = model.predict(x)
        class_idx = np.argmax(res)
        class_labels = ["Tumor", "Not a Tumor"]
        class_name = class_labels[class_idx]

        return render(request, 'scanner/result.html', {
            'image_url': img_url,
            'predicted_class': class_name
        })

    return render(request, 'scanner/brain.html')
# @login_required(login_url='auth/login')
def lung(request):
    if request.method == 'POST':
        user = request.user
        Patient_Name = request.POST.get('patient-name')
        file = request.FILES['mri-scan']
        scan = Scan(user=user, Patient_Name=Patient_Name)
        scan.save()
        id = scan.id
        #save the image in the media folder with the id as the name and the extension
        unique_filename = str(uuid.uuid4())
        extension = file.name.split('.')[1]
        str_file = unique_filename + '.' + extension
        #save the image in the media folder with the id as the name and the extension
        file = ContentFile(file.read())
        default_storage.save('images/'+str(str_file), file)
        scan.image = 'images/'+str(str_file)

        print(str_file)
    
        scan.save()
        img_url = 'media/images/'+str(str_file)
        model = tensorflow.keras.models.load_model('lung_model')
        img = image.load_img(img_url, target_size=(512, 512), color_mode='grayscale')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        preds = model.predict(x)
        class_idx = np.argmax(preds)
        class_labels = ['Benign Cancer Detected', 'Malignant Cancer Detected', 'No Cancer Detected']
        class_name = class_labels[class_idx]
        scan.result = class_name
        scan.save()

        print('Predicted class: ', class_name)
        
        
        return render(request, 'scanner/result.html', {'scan':scan, 'img_url':img_url})



    return render(request, 'scanner/lung.html')

# @login_required(login_url='auth/login')
# def kidney(request):
#      if request.method == 'POST':
#         user = request.user
#         Patient_Name = request.POST.get('patient-name')
#         file = request.FILES['mri-scan']
#         scan = Scan(user=user, Patient_Name=Patient_Name)
#         scan.save()
#         id = scan.id
#         #save the image in the media folder with the id as the name and the extension
#         unique_filename = str(uuid.uuid4())
#         extension = file.name.split('.')[1]
#         str_file = unique_filename + '.' + extension
#         #save the image in the media folder with the id as the name and the extension
#         file = ContentFile(file.read())
#         default_storage.save('images/'+str(str_file), file)
#         scan.image = 'images/'+str(str_file)
#         scan.save()
        
#         model = tensorflow.keras.models.load_model('kidney_model')
#         img = cv2.imread('media/images/'+str(str_file), cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (512, 512))
#         img = np.reshape(img, (1, 512, 512, 1))
#         img = img.astype('float32') / 255
#         predictions = model.predict(img)
#         predicted_class = np.argmax(predictions)
#         c=0
#         if predicted_class == 0:
#                 scan.result = 'Cyst Detected'
#                 scan.save()
#         elif predicted_class == 1:
#                     scan.result = 'No Cancer Detected'
#                     scan.save()
#                     c=1
#         elif predicted_class == 2:
#                     scan.result = 'Stone Detected'
#                     scan.save()
#         else:
#                 scan.result = 'Tumor Detected'
#                 scan.save()
#         img_url = 'media/images/'+str(str_file)
        
#         return render(request, 'scanner/result.html', {'scan':scan, 'img_url':img_url, 'c':c})



#      return render(request, 'scanner/kidney.html')

def breast(request):
    """ 
  
    Reading training data set. 
    """
    df = pd.read_csv('static/Breast_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, Y.shape)

    """ 
    
    Reading data from user. 
    """
    value = ''
    if request.method == 'POST':

        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

        """ 
        Creating our training model. 
        """
        rf = RandomForestClassifier(
            n_estimators=16, criterion='entropy', max_depth=5)
        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array(
            (radius,
             texture,
             perimeter,
             area,
             smoothness)
        ).reshape(1, 5)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'breast.html',
                  {
                      'result': value,
                      'title': 'Breast Cancer Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'breast': True,
                      'form': BreastCancerForm(),
                  })




# Load the pre-trained model
# model = tf.keras.models.load_model('my_model')

# def dicom_to_png(dicom_path):
#     # Load the DICOM file
#     dicom_image = pydicom.read_file(dicom_path).pixel_array
    
#     # Convert DICOM to PNG
#     png_image = Image.fromarray(dicom_image.astype(np.uint8))
    
#     # Resize the image to match the expected input shape
#     png_image = png_image.resize((48, 48))
    
#     # Convert the image to a numpy array and add the channel dimension
#     image = np.array(png_image)
#     image = np.expand_dims(image, axis=-1)
    
#     return image

# def predict_breast_cancer(model, dicom_path):
#     # Convert DICOM to array that can be used by the model
#     image = dicom_to_png(dicom_path)
#     image = image / 255.0  # Normalize to [0, 1]
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Predict
#     prediction = model.predict(image)
    
    
#     if prediction[0][0] >= 0.5:
#         result = "Breast cancer present"
#     else:
#         result = "No breast cancer"
    
#     return result, image[0] 

# def breast_cancer_detection(request):
#     if request.method == 'POST' and 'file' in request.FILES:
#         uploaded_file = request.FILES['file']
        
       
#         with open("uploaded_file.dcm", "wb") as f:
#             for chunk in uploaded_file.chunks():
#                 f.write(chunk)

       
#         prediction, image = predict_breast_cancer(model, "uploaded_file.dcm")
        
#         return render(request, 'result1.html', {'prediction': prediction, 'image': image})
#     else:
#         return render(request, 'kidney.html')

# Load the pre-trained model
model = tf.keras.models.load_model('my_model')

def dicom_to_png(dicom_path):
    try:
        # Load the DICOM file
        dicom_image = pydicom.read_file(dicom_path).pixel_array
        
        # Convert DICOM to PNG
        png_image = Image.fromarray(dicom_image.astype(np.uint8))
        
        # Resize the image to match the expected input shape
        png_image = png_image.resize((48, 48))
        
        # Convert the image to a numpy array
        image = np.array(png_image)
        
        return image
    except Exception as e:
        print(f"Error converting DICOM to PNG: {e}")
        return None

def predict_breast_cancer(model, dicom_path):
    try:
        # Convert DICOM to array that can be used by the model
        image = dicom_to_png(dicom_path)
        if image is None:
            return "Error converting DICOM to PNG", None
        
        image = image / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(image)
        
        # Convert prediction to a human-readable label
        if prediction[0][0] >= 0.5:
            result = "Breast cancer present"
        else:
            result = "No breast cancer"
        
        return result, image[0]  # Return the image without the batch dimension
    except Exception as e:
        print(f"Error predicting breast cancer: {e}")
        return "Error predicting breast cancer", None

def breast_cancer_detection(request):
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        
        # Save the uploaded file to disk
        file_name = default_storage.save(uploaded_file.name, uploaded_file)
        file_path = default_storage.path(file_name)
        
        # Predict on the uploaded DICOM file
        prediction, image = predict_breast_cancer(model, file_path)
        if image is None:
            return render(request, 'result1.html', {'prediction': prediction, 'image': None})
        
        # Convert the image to a format that can be displayed in HTML
        image = Image.fromarray((image * 255).astype(np.uint8))
        response = HttpResponse(content_type="image/png")
        image.save(response, "PNG")
        
        return render(request, 'result1.html', {'prediction': prediction, 'image': response.getvalue()})
    else:
        return render(request, 'kidney.html')
