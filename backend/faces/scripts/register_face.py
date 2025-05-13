#!/usr/bin/env python
import os
import sys
import django
import cv2
import shutil
import argparse
from datetime import datetime
import numpy as np

# Add the project root to the path so we can import Django settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# Import Django models
from faces.models import Person, FaceImage
from faces.eigenfaces_processor import EigenfacesProcessor

def register_new_face(name, image_path):
    """Register a new face in the database"""
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Try to load the image to verify it's a valid image file
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Couldn't read image file: {image_path}")
            return
    except Exception as e:
        print(f"Error reading image: {str(e)}")
        return
    
    try:
        # Create or get person
        person, created = Person.objects.get_or_create(name=name)
        if created:
            print(f"Created new person: {name}")
        else:
            print(f"Using existing person: {name}")
        
        # Create media directory if it doesn't exist
        media_dir = os.path.join(project_root, 'media', 'faces')
        os.makedirs(media_dir, exist_ok=True)
        
        # Create a new filename
        file_ext = os.path.splitext(image_path)[1].lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{file_ext}"
        new_path = os.path.join(media_dir, new_filename)
        
        # Copy the image to the media directory
        shutil.copy2(image_path, new_path)
        
        # Get relative path for database
        rel_path = os.path.join('faces', new_filename)
        
        # Process the image and extract features
        processor = EigenfacesProcessor()
        img_vector = processor.preprocess_image(new_path)
        
        # Create the FaceImage record
        face_image = FaceImage(
            person=person,
            image=rel_path,
            processed=True,
            features_vector=img_vector.tolist()
        )
        face_image.save()
        
        print(f"Successfully registered face image for {name}")
        print(f"Image saved as: {new_path}")
        print(f"Features extracted and stored in database")
        
        # Show the processed image
        display_image = (img_vector.reshape(processor.image_size) * 255).astype(np.uint8)
        cv2.imshow("Processed Face", display_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error registering face: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register a new face in the system')
    parser.add_argument('name', help='Name of the person')
    parser.add_argument('image_path', help='Path to face image file')
    
    args = parser.parse_args()
    register_new_face(args.name, args.image_path) 