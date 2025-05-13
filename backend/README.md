# Eigenfaces Facial Recognition System

This system uses eigenfaces (a PCA-based approach) for facial recognition. The system is built with Django and includes a complete backend for processing, training, and recognizing faces.

## Features

- Face image processing and feature extraction
- Eigenfaces model training with memory optimization
- Facial recognition with confidence scores
- Support for the AT&T/ORL Faces database
- Utilities for testing and registering new faces

## Current Model Status

The system is trained with 30 people (7 images per person) from the AT&T/ORL Faces database, achieving approximately 97% recognition accuracy on test images.

## Project Structure

- `faces/` - Django app with eigenfaces implementation
  - `eigenfaces_processor.py` - Core implementation of the eigenfaces algorithm
  - `models.py` - Database models for Person, FaceImage, and EigenfacesModel
  - `management/commands/` - Django commands for dataset import and training
  - `scripts/` - Utility scripts for face recognition tasks
- `eigenfaces.py` - Command-line interface for the system

## Usage

### Prerequisites

- Python 3.6+
- Django
- OpenCV
- NumPy
- PIL (Pillow)
- psutil

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Command-line Interface

The system provides a unified command-line interface through `eigenfaces.py`:

```bash
# Register a new face
./eigenfaces.py register "Your Name" path/to/your/face/image.jpg

# Test recognition with an image
./eigenfaces.py test path/to/your/face/image.jpg

# Retrain the model (after adding new faces)
./eigenfaces.py retrain --max-eigenfaces 40 --batch-size 5 --memory-report

# Evaluate model accuracy
./eigenfaces.py evaluate
```

### Adding Your Own Face

To add your own face to the database:

1. Take a well-lit, front-facing photo of your face
2. Run the register command:
   ```bash
   ./eigenfaces.py register "Your Name" path/to/your/face/image.jpg
   ```
3. After adding faces, retrain the model:
   ```bash
   ./eigenfaces.py retrain
   ```
4. Test recognition with another photo:
   ```bash
   ./eigenfaces.py test path/to/another/photo.jpg
   ```

## Model Performance

The current trained model achieves approximately 97% accuracy on the test dataset, with an average distance of around 5.2.

## Eigenfaces Theory

Eigenfaces is a Principal Component Analysis (PCA) based approach for facial recognition:

1. Face images are converted to vectors
2. The mean face is computed and subtracted
3. PCA is performed to find the principal components (eigenfaces)
4. Faces are represented by their projections onto the eigenspace
5. Recognition is performed by comparing these projections

## Docker Support

Coming soon...

## Frontend Interface

Coming soon...

## License

This project is licensed under the MIT License - see the LICENSE file for details. 