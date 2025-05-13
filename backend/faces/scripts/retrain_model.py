#!/usr/bin/env python
import os
import sys
import django
import numpy as np
import argparse
import psutil

# Add the project root to the path so we can import Django settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# Import Django models
from faces.models import Person, FaceImage, EigenfacesModel
from faces.eigenfaces_processor import EigenfacesProcessor

def retrain_model(max_eigenfaces=40, batch_size=5, memory_report=True):
    """Retrain the eigenfaces model using all available processed face images"""
    
    if memory_report:
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f'Initial memory usage: {initial_memory:.2f} MB')
    
    # Count available training data
    persons = Person.objects.all()
    
    if persons.count() == 0:
        print('No persons found in the database')
        return
    
    print(f'Found {persons.count()} persons in the database')
    
    # Get all processed face images
    face_images = FaceImage.objects.filter(processed=True)
    total_images = face_images.count()
    
    if total_images < 2:
        print('Need at least 2 face images to train the model')
        return
    
    print(f'Training with {total_images} images from {persons.count()} persons')
    
    # Memory usage after selection
    if memory_report:
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f'Memory usage after image selection: {current_memory:.2f} MB (Δ {current_memory - initial_memory:.2f} MB)')
    
    # Initialize the eigenfaces processor
    processor = EigenfacesProcessor(batch_size=batch_size)
    
    try:
        # Train the model with memory optimization
        print('Training eigenfaces model with batch processing...')
        
        # Memory usage before training
        if memory_report:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f'Memory usage before training: {current_memory:.2f} MB')
        
        # Training step
        result = processor.train(face_images, max_eigenfaces=max_eigenfaces)
        
        # Memory usage after training
        if memory_report:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f'Memory usage after training: {current_memory:.2f} MB')
        
        # Save the model
        model = processor.save_model(name=f"Full_Model_{total_images}_images_{max_eigenfaces}_eigenfaces")
        
        # Set all other models to inactive
        EigenfacesModel.objects.exclude(pk=model.pk).update(is_active=False)
        
        print(f'Model trained successfully with {len(result["eigenfaces"])} eigenfaces')
        
        # Display some info about the model
        print(f'Mean face shape: {np.array(model.mean_face).shape}')
        print(f'Eigenfaces shape: {np.array(model.eigenfaces).shape}')
        
        # Final memory usage
        if memory_report:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f'Final memory usage: {current_memory:.2f} MB (Δ {current_memory - initial_memory:.2f} MB)')
        
    except Exception as e:
        print(f'Error training model: {str(e)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrain the eigenfaces model')
    parser.add_argument('--max-eigenfaces', type=int, default=40,
                        help='Maximum number of eigenfaces to compute (default: 40)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of images to process in one batch (default: 5)')
    parser.add_argument('--memory-report', action='store_true',
                        help='Show memory usage during training')
    
    args = parser.parse_args()
    retrain_model(args.max_eigenfaces, args.batch_size, args.memory_report) 