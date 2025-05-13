import os
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings

from faces.models import Person, FaceImage, EigenfacesModel
from faces.eigenfaces_processor import EigenfacesProcessor

class Command(BaseCommand):
    help = 'Train eigenfaces model with AT&T/ORL faces (limited subset)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--max-persons',
            type=int,
            help='Maximum number of persons to use for training',
            default=10
        )
        parser.add_argument(
            '--max-images-per-person',
            type=int,
            help='Maximum number of images per person to use',
            default=5
        )

    def handle(self, *args, **options):
        max_persons = options['max_persons']
        max_images_per_person = options['max_images_per_person']
        
        self.stdout.write(f'Training with max {max_persons} persons, {max_images_per_person} images each')
        
        # Get a limited subset of people
        persons = Person.objects.all()[:max_persons]
        
        if persons.count() == 0:
            self.stdout.write(self.style.ERROR('No persons found in the database'))
            return
        
        self.stdout.write(f'Selected {persons.count()} persons for training')
        
        # Get processed images for these people, limited per person
        selected_face_images = []
        
        for person in persons:
            # Get a limited number of images for this person
            person_images = FaceImage.objects.filter(
                person=person, 
                processed=True
            )[:max_images_per_person]
            
            selected_face_images.extend(person_images)
            
            self.stdout.write(f'Selected {person_images.count()} images for {person.name}')
        
        total_images = len(selected_face_images)
        
        if total_images < 2:
            self.stdout.write(self.style.ERROR('Need at least 2 face images to train the model'))
            return
        
        self.stdout.write(f'Training with {total_images} images in total')
        
        # Initialize the eigenfaces processor
        processor = EigenfacesProcessor()
        
        try:
            # Train the model with memory optimization
            self.stdout.write('Training eigenfaces model...')
            result = processor.train(selected_face_images)
            
            # Save the model
            model = processor.save_model(name=f"ATT_Model_{total_images}_images")
            
            # Set all other models to inactive
            EigenfacesModel.objects.exclude(pk=model.pk).update(is_active=False)
            
            self.stdout.write(self.style.SUCCESS(
                f'Model trained successfully with {len(result["eigenfaces"])} eigenfaces'
            ))
            
            # Display some info about the model
            self.stdout.write(f'Mean face shape: {np.array(model.mean_face).shape}')
            self.stdout.write(f'Eigenfaces shape: {np.array(model.eigenfaces).shape}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error training model: {str(e)}')) 