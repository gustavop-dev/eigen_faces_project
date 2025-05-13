#!/usr/bin/env python
import os
import sys
import django
import numpy as np
import cv2
from PIL import Image
import argparse

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

def recognize_face(image_path, threshold=12.0):
    """Test recognition on a custom face image"""
    
    # Load the active model
    model = EigenfacesModel.objects.filter(is_active=True).first()
    
    if not model:
        print("No active eigenfaces model found")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"Testing recognition with model: {model.name}")
    print(f"Image: {image_path}")
    
    # Initialize eigenfaces processor and load the model
    processor = EigenfacesProcessor()
    processor.load_model(model)
    
    try:
        # Preprocess the input image
        img_vector = processor.preprocess_image(image_path)
        
        # Display the processed image
        display_image = (img_vector.reshape(processor.image_size) * 255).astype(np.uint8)
        cv2.imshow("Processed Input", display_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Try to recognize the face
        result = processor.recognize_face(img_vector, threshold=threshold)
        
        # Get the recognized person name
        if result["recognized"] and result["person_id"]:
            try:
                recognized_person = Person.objects.get(id=result["person_id"])
                print(f"\nRecognized as: {recognized_person.name}")
                print(f"Distance: {result['distance']:.4f}")
                
                # Display a sample image of the recognized person
                person_samples = FaceImage.objects.filter(person=recognized_person)[:1]
                if person_samples:
                    sample_img = cv2.imread(person_samples[0].image.path, cv2.IMREAD_GRAYSCALE)
                    cv2.imshow(f"Matched with: {recognized_person.name}", sample_img)
                    print("Press any key to close...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
            except Person.DoesNotExist:
                print(f"\nRecognized as unknown person (ID: {result['person_id']})")
        else:
            print(f"\nNot recognized as any known person")
            print(f"Distance to closest match: {result['distance']:.4f}")
            
        # Show the mean face for comparison
        mean_face_img = (np.array(model.mean_face).reshape(processor.image_size) * 255).astype(np.uint8)
        cv2.imshow("Mean Face", mean_face_img)
        
        # Show closest eigenfaces
        eigenfaces = np.array(model.eigenfaces)
        for i in range(min(3, len(eigenfaces))):
            # Normalize eigenface for display
            eigenface = eigenfaces[i].reshape(processor.image_size)
            eigenface_norm = cv2.normalize(eigenface, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow(f"Eigenface {i+1}", eigenface_norm)
            
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during recognition: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test eigenfaces with your own image')
    parser.add_argument('image_path', help='Path to face image file')
    parser.add_argument('--threshold', type=float, default=12.0, 
                        help='Recognition threshold (default: 12.0)')
    
    args = parser.parse_args()
    recognize_face(args.image_path, args.threshold) 