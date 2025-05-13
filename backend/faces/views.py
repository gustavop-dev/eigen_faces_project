from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

import os
import numpy as np
import cv2
import json
import base64
from PIL import Image, ImageEnhance
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .models import Person, FaceImage, EigenfacesModel
from .eigenfaces_processor import EigenfacesProcessor

def train_eigenfaces_model(request):
    """
    Train a new eigenfaces model using all available face images
    """
    # Get all processed face images
    face_images = FaceImage.objects.filter(processed=True)
    
    if face_images.count() < 2:
        return JsonResponse({
            'success': False,
            'error': 'Need at least 2 face images to train the model'
        })
    
    # Initialize the eigenfaces processor
    processor = EigenfacesProcessor()
    
    try:
        # Train the model
        result = processor.train(face_images)
        
        # Save the model
        model = processor.save_model(name=f"EigenfacesModel_{face_images.count()}_images")
        
        # Set all other models to inactive
        EigenfacesModel.objects.exclude(pk=model.pk).update(is_active=False)
        
        return JsonResponse({
            'success': True,
            'model_id': model.id,
            'eigenfaces_count': len(result['eigenfaces'])
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
def process_image(request):
    """
    Process an uploaded image:
    - Save it to the database
    - Preprocess it
    - Mark it as processed
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    # Get person info
    person_name = request.POST.get('name')
    if not person_name:
        return JsonResponse({'error': 'Person name is required'}, status=400)
    
    # Get or create person
    person, created = Person.objects.get_or_create(name=person_name)
    
    # Check if image was uploaded
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file uploaded'}, status=400)
    
    image_file = request.FILES['image']
    
    # Save the face image
    face_image = FaceImage(person=person, image=image_file)
    face_image.save()
    
    # Preprocess the image
    try:
        processor = EigenfacesProcessor()
        img_vector = processor.preprocess_image(face_image.image.path)
        
        # Mark the image as processed
        face_image.processed = True
        face_image.features_vector = img_vector.tolist()
        face_image.save()
        
        return JsonResponse({
            'success': True,
            'person_id': person.id,
            'image_id': face_image.id
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
def recognize_face(request):
    """
    Recognize a face from an uploaded image
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    # Check if image was uploaded
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file uploaded'}, status=400)
    
    image_file = request.FILES['image']
    
    # Save the image temporarily
    path = default_storage.save('temp/temp_face.jpg', ContentFile(image_file.read()))
    temp_file_path = os.path.join(settings.MEDIA_ROOT, path)
    
    try:
        # Get the active eigenfaces model
        eigenfaces_model = EigenfacesModel.objects.filter(is_active=True).first()
        
        if not eigenfaces_model:
            return JsonResponse({
                'success': False,
                'error': 'No active eigenfaces model found'
            }, status=400)
        
        # Initialize the eigenfaces processor
        processor = EigenfacesProcessor()
        
        # Load the model
        if not processor.load_model(eigenfaces_model):
            return JsonResponse({
                'success': False,
                'error': 'Failed to load eigenfaces model'
            }, status=500)
        
        # Get all processed face images for recognition
        face_images = FaceImage.objects.filter(processed=True)
        processor.prepare_training_data(face_images)
        
        # Preprocess the uploaded image
        img_vector = processor.preprocess_image(temp_file_path)
        
        # Recognize the face
        result = processor.recognize_face(img_vector)
        
        # If recognized, get the person's information
        if result['recognized']:
            person = Person.objects.get(pk=result['person_id'])
            result['person_name'] = person.name
        
        # Delete the temporary file
        default_storage.delete(path)
        
        return JsonResponse({
            'success': True,
            'recognition_result': result
        })
    except Exception as e:
        # Ensure temporary file is deleted even if an error occurs
        if default_storage.exists(path):
            default_storage.delete(path)
        
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def visualize_eigenfaces(request, model_id=None):
    """
    Visualize the eigenfaces and mean face from a model
    """
    # Get the model
    if model_id:
        eigenfaces_model = get_object_or_404(EigenfacesModel, pk=model_id)
    else:
        eigenfaces_model = EigenfacesModel.objects.filter(is_active=True).first()
    
    if not eigenfaces_model:
        return JsonResponse({
            'success': False,
            'error': 'No eigenfaces model found'
        }, status=404)
    
    # Initialize the eigenfaces processor
    processor = EigenfacesProcessor()
    
    # Load the model
    if not processor.load_model(eigenfaces_model):
        return JsonResponse({
            'success': False,
            'error': 'Failed to load eigenfaces model'
        }, status=500)
    
    # Convert eigenfaces and mean face to images
    try:
        # Generate the mean face image
        mean_face_array = np.array(eigenfaces_model.mean_face).reshape(processor.image_size)
        mean_face_array = (mean_face_array * 255).astype(np.uint8)
        mean_face_img = Image.fromarray(mean_face_array)
        
        # Enhance contrast to make it more visible
        enhancer = ImageEnhance.Contrast(mean_face_img)
        mean_face_img = enhancer.enhance(2.0)
        
        # Convert to base64
        buffer = io.BytesIO()
        mean_face_img.save(buffer, format='PNG')
        mean_face_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Generate eigenface images (take first 16 or less)
        eigenfaces_array = np.array(eigenfaces_model.eigenfaces)
        num_eigenfaces = min(16, len(eigenfaces_array))
        eigenface_images = []
        
        for i in range(num_eigenfaces):
            eigenface = eigenfaces_array[i].reshape(processor.image_size)
            
            # Normalize to 0-255 range
            eigenface = ((eigenface - eigenface.min()) / (eigenface.max() - eigenface.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            eigenface_img = Image.fromarray(eigenface)
            
            # Enhance contrast to make it more visible
            enhancer = ImageEnhance.Contrast(eigenface_img)
            eigenface_img = enhancer.enhance(2.0)
            
            # Convert to base64
            buffer = io.BytesIO()
            eigenface_img.save(buffer, format='PNG')
            eigenface_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            eigenface_images.append({
                'index': i,
                'image': eigenface_base64
            })
        
        return JsonResponse({
            'success': True,
            'model_name': eigenfaces_model.name,
            'mean_face': mean_face_base64,
            'eigenfaces': eigenface_images,
            'total_eigenfaces': len(eigenfaces_array)
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
