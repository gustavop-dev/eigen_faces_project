import os
import glob
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings

from faces.models import Person, FaceImage
from faces.eigenfaces_processor import EigenfacesProcessor

class Command(BaseCommand):
    help = 'Import AT&T/ORL faces dataset to the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset-dir',
            type=str,
            help='Directory containing the ORL faces dataset',
            default=os.path.join(settings.BASE_DIR, 'download_data', 'download_data', 'orl_faces')
        )

    def handle(self, *args, **options):
        dataset_dir = options['dataset_dir']
        
        if not os.path.exists(dataset_dir):
            self.stdout.write(self.style.ERROR(f'Dataset directory not found: {dataset_dir}'))
            return
        
        # Get all subject directories (s1, s2, s3, ...)
        subject_dirs = sorted(glob.glob(os.path.join(dataset_dir, 's*')))
        
        if not subject_dirs:
            self.stdout.write(self.style.ERROR(f'No subject directories found in {dataset_dir}'))
            return
        
        self.stdout.write(f'Found {len(subject_dirs)} subjects in the dataset')
        
        # Process each subject
        for subject_dir in subject_dirs:
            subject_name = os.path.basename(subject_dir)
            
            # Create or get person
            person, created = Person.objects.get_or_create(name=subject_name)
            
            if created:
                self.stdout.write(f'Created person: {subject_name}')
            else:
                self.stdout.write(f'Using existing person: {subject_name}')
            
            # Get all PGM images for this subject
            image_files = sorted(glob.glob(os.path.join(subject_dir, '*.pgm')))
            
            if not image_files:
                self.stdout.write(self.style.WARNING(f'No images found for {subject_name}'))
                continue
            
            self.stdout.write(f'Found {len(image_files)} images for {subject_name}')
            
            # Process each image
            for image_path in image_files:
                image_filename = os.path.basename(image_path)
                
                # Check if this image already exists
                if FaceImage.objects.filter(image__contains=f'{subject_name}_{image_filename}').exists():
                    self.stdout.write(f'Image {subject_name}_{image_filename} already exists, skipping...')
                    continue
                
                # Read the image and create a FaceImage instance
                with open(image_path, 'rb') as f:
                    image_content = f.read()
                    
                    # Create the FaceImage
                    face_image = FaceImage(person=person)
                    face_image.image.save(f'{subject_name}_{image_filename}', ContentFile(image_content))
                    
                    # Process the image
                    try:
                        processor = EigenfacesProcessor()
                        img_vector = processor.preprocess_image(face_image.image.path)
                        
                        # Save the processed features
                        face_image.processed = True
                        face_image.features_vector = img_vector.tolist()
                        face_image.save()
                        
                        self.stdout.write(f'Processed image: {subject_name}_{image_filename}')
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f'Error processing image {image_path}: {str(e)}'))
        
        # Summary
        total_faces = FaceImage.objects.filter(processed=True).count()
        self.stdout.write(self.style.SUCCESS(f'Successfully imported and processed {total_faces} face images')) 