import os
import numpy as np
import cv2
from django.core.management.base import BaseCommand
from django.conf import settings
from django.core.files.base import ContentFile

from faces.models import Person, FaceImage, EigenfacesModel
from faces.eigenfaces_processor import EigenfacesProcessor

class Command(BaseCommand):
    help = 'Test the Eigenfaces algorithm on a small dataset'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset-dir',
            type=str,
            help='Directory containing test face images',
            default=os.path.join(settings.BASE_DIR, 'test_dataset')
        )

    def create_test_dataset(self, dataset_dir):
        """Create a small test dataset if none exists"""
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        # Create subdirectories for different people
        person1_dir = os.path.join(dataset_dir, 'person1')
        person2_dir = os.path.join(dataset_dir, 'person2')
        
        if not os.path.exists(person1_dir):
            os.makedirs(person1_dir)
        if not os.path.exists(person2_dir):
            os.makedirs(person2_dir)
        
        # Generate simple test images if they don't exist
        # (In a real scenario, you would use actual face images)
        if len(os.listdir(person1_dir)) == 0:
            self.stdout.write('Creating test images for person1...')
            for i in range(5):
                # Create a gradient image (simulating a face)
                img = np.zeros((100, 100), dtype=np.uint8)
                for r in range(100):
                    for c in range(100):
                        img[r, c] = (r + c + i * 10) % 256
                
                # Save the image
                cv2.imwrite(os.path.join(person1_dir, f'face_{i}.jpg'), img)
        
        if len(os.listdir(person2_dir)) == 0:
            self.stdout.write('Creating test images for person2...')
            for i in range(5):
                # Create a different pattern image
                img = np.zeros((100, 100), dtype=np.uint8)
                for r in range(100):
                    for c in range(100):
                        img[r, c] = (abs(r - 50) + abs(c - 50) + i * 10) % 256
                
                # Save the image
                cv2.imwrite(os.path.join(person2_dir, f'face_{i}.jpg'), img)
        
        return True

    def load_images_to_db(self, dataset_dir):
        """Load images from the dataset into the database"""
        # Iterate through the dataset directory
        for person_dir in os.listdir(dataset_dir):
            person_path = os.path.join(dataset_dir, person_dir)
            
            if os.path.isdir(person_path):
                # Create or get person
                person, created = Person.objects.get_or_create(name=person_dir)
                
                if created:
                    self.stdout.write(f'Created person: {person_dir}')
                else:
                    self.stdout.write(f'Using existing person: {person_dir}')
                
                # Load images for this person
                for img_file in os.listdir(person_path):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_path, img_file)
                        
                        # Check if this image has already been processed
                        img_name = os.path.basename(img_path)
                        
                        if FaceImage.objects.filter(image__contains=img_name).exists():
                            self.stdout.write(f'Image {img_name} already exists, skipping...')
                            continue
                        
                        # Read the image and create a FaceImage instance
                        with open(img_path, 'rb') as f:
                            image_content = f.read()
                            
                            # Create the FaceImage
                            face_image = FaceImage(person=person)
                            face_image.image.save(img_name, ContentFile(image_content))
                            
                            # Process the image
                            processor = EigenfacesProcessor()
                            img_vector = processor.preprocess_image(face_image.image.path)
                            
                            # Save the processed features
                            face_image.processed = True
                            face_image.features_vector = img_vector.tolist()
                            face_image.save()
                            
                            self.stdout.write(f'Processed image: {img_name}')

    def train_model(self):
        """Train an eigenfaces model using the processed images"""
        # Get all processed face images
        face_images = FaceImage.objects.filter(processed=True)
        
        if face_images.count() < 2:
            self.stdout.write(self.style.ERROR('Need at least 2 face images to train the model'))
            return False
        
        # Initialize the eigenfaces processor
        processor = EigenfacesProcessor()
        
        try:
            # Train the model
            self.stdout.write('Training eigenfaces model...')
            result = processor.train(face_images)
            
            # Save the model
            model = processor.save_model(name=f"TestModel_{face_images.count()}_images")
            
            # Set all other models to inactive
            EigenfacesModel.objects.exclude(pk=model.pk).update(is_active=False)
            
            self.stdout.write(self.style.SUCCESS(f'Model trained successfully with {len(result["eigenfaces"])} eigenfaces'))
            return True
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error training model: {str(e)}'))
            return False

    def test_recognition(self):
        """Test face recognition using the trained model"""
        # Get the active model
        eigenfaces_model = EigenfacesModel.objects.filter(is_active=True).first()
        
        if not eigenfaces_model:
            self.stdout.write(self.style.ERROR('No active eigenfaces model found'))
            return
        
        # Get all processed face images
        face_images = FaceImage.objects.filter(processed=True)
        
        if face_images.count() == 0:
            self.stdout.write(self.style.ERROR('No face images found for testing'))
            return
        
        # Initialize the eigenfaces processor
        processor = EigenfacesProcessor()
        
        # Load the model
        if not processor.load_model(eigenfaces_model):
            self.stdout.write(self.style.ERROR('Failed to load eigenfaces model'))
            return
        
        # Prepare training data
        processor.prepare_training_data(face_images)
        
        # Test recognition on each face
        correct = 0
        total = 0
        
        for face in face_images:
            try:
                # Get the processed image vector
                img_vector = np.array(face.features_vector)
                
                # Try to recognize it
                result = processor.recognize_face(img_vector)
                
                # Check if recognition was correct
                if result['recognized'] and result['person_id'] == face.person.id:
                    correct += 1
                    self.stdout.write(f'Correctly recognized {face.person.name}')
                else:
                    if result['recognized']:
                        recognized_person = Person.objects.get(pk=result['person_id'])
                        self.stdout.write(self.style.WARNING(
                            f'Misidentified {face.person.name} as {recognized_person.name}'
                        ))
                    else:
                        self.stdout.write(self.style.WARNING(
                            f'Failed to recognize {face.person.name}'
                        ))
                
                total += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error testing face {face.id}: {str(e)}'))
        
        if total > 0:
            accuracy = (correct / total) * 100
            self.stdout.write(self.style.SUCCESS(f'Recognition accuracy: {accuracy:.2f}%'))
        else:
            self.stdout.write(self.style.ERROR('No faces were tested'))

    def handle(self, *args, **options):
        dataset_dir = options['dataset_dir']
        
        # Create test dataset if needed
        self.create_test_dataset(dataset_dir)
        
        # Load images to database
        self.load_images_to_db(dataset_dir)
        
        # Train the model
        success = self.train_model()
        
        if success:
            # Test recognition
            self.test_recognition()
        
        self.stdout.write(self.style.SUCCESS('Test completed')) 