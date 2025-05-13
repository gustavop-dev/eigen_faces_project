import os
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings
import psutil  # For memory monitoring

from faces.models import Person, FaceImage, EigenfacesModel
from faces.eigenfaces_processor import EigenfacesProcessor

class Command(BaseCommand):
    help = 'Train eigenfaces model with AT&T/ORL faces with memory optimization'

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
        parser.add_argument(
            '--max-eigenfaces',
            type=int,
            help='Maximum number of eigenfaces to compute',
            default=50
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            help='Number of images to process in one batch',
            default=10
        )
        parser.add_argument(
            '--memory-report',
            action='store_true',
            help='Show memory usage during training'
        )

    def handle(self, *args, **options):
        max_persons = options['max_persons']
        max_images_per_person = options['max_images_per_person']
        max_eigenfaces = options['max_eigenfaces']
        batch_size = options['batch_size']
        memory_report = options['memory_report']
        
        # Initial memory usage
        if memory_report:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            self.stdout.write(f'Initial memory usage: {initial_memory:.2f} MB')
        
        self.stdout.write(f'Training with max {max_persons} persons, {max_images_per_person} images each, {max_eigenfaces} eigenfaces')
        
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
        
        # Memory usage after selection
        if memory_report:
            current_memory = process.memory_info().rss / 1024 / 1024
            self.stdout.write(f'Memory usage after image selection: {current_memory:.2f} MB (Δ {current_memory - initial_memory:.2f} MB)')
        
        # Initialize the eigenfaces processor
        processor = EigenfacesProcessor()
        
        try:
            # Configure batch size
            # This will be used internally by prepare_training_data_batch method
            setattr(processor, 'batch_size', batch_size)
            
            # Train the model with memory optimization
            self.stdout.write('Training eigenfaces model with batch processing...')
            
            # Memory usage before training
            if memory_report:
                current_memory = process.memory_info().rss / 1024 / 1024
                self.stdout.write(f'Memory usage before training: {current_memory:.2f} MB')
            
            # Training step
            result = processor.train(selected_face_images, max_eigenfaces=max_eigenfaces)
            
            # Memory usage after training
            if memory_report:
                current_memory = process.memory_info().rss / 1024 / 1024
                self.stdout.write(f'Memory usage after training: {current_memory:.2f} MB')
            
            # Save the model
            model = processor.save_model(name=f"ATT_Model_{total_images}_images_{max_eigenfaces}_eigenfaces")
            
            # Set all other models to inactive
            EigenfacesModel.objects.exclude(pk=model.pk).update(is_active=False)
            
            self.stdout.write(self.style.SUCCESS(
                f'Model trained successfully with {len(result["eigenfaces"])} eigenfaces'
            ))
            
            # Display some info about the model
            self.stdout.write(f'Mean face shape: {np.array(model.mean_face).shape}')
            self.stdout.write(f'Eigenfaces shape: {np.array(model.eigenfaces).shape}')
            
            # Final memory usage
            if memory_report:
                current_memory = process.memory_info().rss / 1024 / 1024
                self.stdout.write(f'Final memory usage: {current_memory:.2f} MB (Δ {current_memory - initial_memory:.2f} MB)')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error training model: {str(e)}')) 