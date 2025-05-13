# Eigenfaces Project

This project implements a facial recognition system based on the Eigenfaces technique, which uses Principal Component Analysis (PCA) to reduce the dimensionality of facial images and enable efficient recognition.

## Project Structure

The project is organized into two main components:

- **Backend**: Django REST API for image processing and eigenfaces calculations
- **Frontend**: Vue.js interface (coming soon)

## Architecture

- **Backend**: Django + DRF (Django Rest Framework)
- **Image Processing**: OpenCV + NumPy
- **Storage**: SQLite (for development)
- **Frontend** (planned): Vue.js

## Backend Installation

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Apply migrations:
   ```
   python manage.py makemigrations
   python manage.py migrate
   ```

4. Run the server:
   ```
   python manage.py runserver
   ```

## Working with Face Datasets

### Testing with Synthetic Data
To test the system with a basic synthetic dataset:
```
cd backend
python manage.py test_eigenfaces
```
This command creates a test dataset, trains a model, and evaluates its accuracy.

### Using the AT&T/ORL Faces Dataset
1. Download the dataset:
   ```
   cd backend
   mkdir -p download_data
   cd download_data
   wget https://github.com/260963172/ORL/raw/main/ORL.zip
   unzip ORL.zip
   cd ..
   ```

2. Import the dataset into the database:
   ```
   python manage.py import_att_faces
   ```

3. Train the model with a subset of the data (to avoid memory issues):
   ```
   python manage.py train_att_faces --max-persons=10 --max-images-per-person=5
   ```
   You can adjust the parameters to use more data for training.

## Eigenfaces Algorithm Flow

The system implements the Eigenfaces algorithm through these steps:

1. **Image Preprocessing**:
   - Conversion to grayscale
   - Resizing to standard dimensions
   - Pixel normalization

2. **Eigenfaces Calculation**:
   - Calculation of the mean face
   - Normalization by subtracting the mean face
   - Calculation of the covariance matrix
   - Extraction of eigenvectors (eigenfaces)

3. **Recognition**:
   - Projection of a test image into the eigenfaces space
   - Comparison of distances with the training set images
   - Classification based on the minimum distance

## Backend API Endpoints

- `POST /api/faces/process/`: Processes a facial image to include it in the training set
- `POST /api/faces/train/`: Trains the eigenfaces model with the processed images
- `POST /api/faces/recognize/`: Recognizes a face in an uploaded image
- `GET /api/faces/visualize/`: Visualizes the mean face and eigenfaces of the active model

## Management Commands

- `test_eigenfaces`: Creates and tests synthetic data
- `import_att_faces`: Imports the AT&T/ORL faces dataset
- `train_att_faces`: Trains the model with a configurable subset of data

## Frontend (Coming Soon)

The Vue.js frontend will provide:
- User-friendly interface for uploading and recognizing faces
- Visualization of eigenfaces
- Interactive face recognition demo
- Admin panel for training and managing the system

## Project Development Plan

This project is being developed in phases:

1. **Week 8 (Current)**: Basic image processing and Eigenfaces calculation in the backend
   - Models for people, images, and eigenfaces
   - Eigenfaces processor with methods for training and recognition
   - Basic API for processing, training, and recognition
   - Management command for tests with synthetic data

2. **Week 9-10 (Upcoming)**: Frontend development and integration
   - Vue.js frontend for face upload and recognition
   - Visualization of eigenfaces and mean face
   - Test with real faces 