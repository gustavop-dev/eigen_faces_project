#!/usr/bin/env python
import os
import sys
import random
import django
import numpy as np

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

def test_recognition():
    # Load the active model
    model = EigenfacesModel.objects.filter(is_active=True).first()
    
    if not model:
        print("No active eigenfaces model found")
        return
    
    print(f"Testing recognition with model: {model.name}")
    
    # Initialize eigenfaces processor and load the model
    processor = EigenfacesProcessor()
    processor.load_model(model)
    
    # Get persons for testing - use more to better evaluate performance
    test_persons = list(Person.objects.all()[:30])
    
    # For each person, try to recognize a face that wasn't used in training
    correct = 0
    total = 0
    recognition_distances = []
    
    # Print header
    print("\n{:<10} {:<15} {:<15} {:<10}".format("Person", "Recognized as", "Correct?", "Distance"))
    print("-" * 55)
    
    for person in test_persons:
        # Get images for this person that weren't used in training (assuming we used 7 per person)
        test_images = list(FaceImage.objects.filter(person=person)[7:])
        
        if not test_images:
            # print(f"No test images available for {person.name}")
            continue
        
        # Select a random test image
        test_img = random.choice(test_images)
        
        # Preprocess the image
        img_vector = processor.preprocess_image(test_img.image.path)
        
        # Try to recognize the face
        result = processor.recognize_face(img_vector, threshold=12.0)
        
        # Get the recognized person name
        recognized_name = "Unknown"
        is_correct = False
        
        if result["recognized"] and result["person_id"]:
            try:
                recognized_person = Person.objects.get(id=result["person_id"])
                recognized_name = recognized_person.name
                
                # Check if recognition was correct
                is_correct = (recognized_person.id == person.id)
                if is_correct:
                    correct += 1
            except Person.DoesNotExist:
                recognized_name = f"Unknown (ID: {result['person_id']})"
        
        # Print result with nice formatting
        print("{:<10} {:<15} {:<15} {:<10.4f}".format(
            person.name, 
            recognized_name,
            "✓" if is_correct else "✗",
            result["distance"]
        ))
        
        # Save the distance for statistics
        recognition_distances.append(result["distance"])
        
        total += 1
    
    # Print summary
    if total > 0:
        avg_distance = np.mean(recognition_distances)
        min_distance = np.min(recognition_distances)
        max_distance = np.max(recognition_distances)
        
        print("\nRecognition Results:")
        print(f"Accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")
        print(f"Average distance: {avg_distance:.4f}")
        print(f"Min distance: {min_distance:.4f}")
        print(f"Max distance: {max_distance:.4f}")
    else:
        print("\nNo tests were performed")

if __name__ == "__main__":
    test_recognition() 