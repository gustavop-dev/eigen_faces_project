import os
import numpy as np
import cv2
from PIL import Image
from django.conf import settings
import json

class EigenfacesProcessor:
    def __init__(self, image_size=(100, 100)):
        """
        Initialize the eigenfaces processor with the specified image size
        
        Args:
            image_size (tuple): Standard dimensions to resize all face images (width, height)
        """
        self.image_size = image_size
        self.mean_face = None
        self.eigenfaces = None
        self.projected_training_faces = []
        self.training_labels = []
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for eigenfaces:
        - Convert to grayscale
        - Resize to standard dimensions
        - Normalize pixel values
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Flattened image vector with normalized values
        """
        # Read image using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Try with PIL if OpenCV fails
            pil_img = Image.open(image_path).convert('L')
            img = np.array(pil_img)
        
        # Resize to standard dimensions
        img = cv2.resize(img, self.image_size)
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        # Flatten the image into a vector
        img_vector = img.flatten()
        
        return img_vector
    
    def prepare_training_data(self, face_images):
        """
        Prepare training data from a list of FaceImage objects
        
        Args:
            face_images (list): List of FaceImage model instances
            
        Returns:
            numpy.ndarray: Matrix of training images as row vectors
        """
        training_data = []
        self.training_labels = []
        
        for face in face_images:
            img_path = face.image.path
            person_id = face.person.id
            
            try:
                # If the image has already been processed and has features_vector,
                # use it directly to save memory and processing time
                if face.processed and face.features_vector:
                    img_vector = np.array(face.features_vector)
                else:
                    img_vector = self.preprocess_image(img_path)
                
                training_data.append(img_vector)
                self.training_labels.append(person_id)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        return np.array(training_data)
    
    def train(self, face_images, max_eigenfaces=150):
        """
        Train the eigenfaces model with memory optimization:
        - Calculate the mean face
        - Subtract mean from each face
        - Compute eigenfaces (eigenvectors of covariance matrix)
        
        For large datasets, limit the number of eigenfaces to save memory
        
        Args:
            face_images (list): List of FaceImage model instances
            max_eigenfaces (int): Maximum number of eigenfaces to retain
            
        Returns:
            dict: Dictionary containing the mean_face and eigenfaces as lists
        """
        # Prepare training data
        training_data = self.prepare_training_data(face_images)
        
        if len(training_data) == 0:
            raise ValueError("No valid training images found")
        
        # Calculate mean face
        self.mean_face = np.mean(training_data, axis=0)
        
        # Subtract mean from each face
        normalized_faces = training_data - self.mean_face
        
        # Use SVD method for eigenface calculation (more memory efficient)
        # We're using the trick that for high-dimensional data, we can compute
        # the eigenvectors of X*X^T instead of X^T*X
        print("Computing SVD decomposition...")
        
        # Compute reduced SVD to save memory
        U, s, Vt = np.linalg.svd(normalized_faces, full_matrices=False)
        
        # Limit the number of eigenfaces to save memory
        max_k = min(max_eigenfaces, len(s))
        print(f"Using top {max_k} eigenfaces out of {len(s)} possible")
        
        # The eigenfaces are the eigenvectors of the covariance matrix,
        # which we can get from the V matrix of SVD
        self.eigenfaces = Vt[:max_k]
        
        # Project training faces into eigenspace
        # We can do this with a batch operation to save memory
        self.projected_training_faces = np.dot(normalized_faces, self.eigenfaces.T)
        
        # Return the model for saving
        return {
            'mean_face': self.mean_face.tolist(),
            'eigenfaces': self.eigenfaces.tolist()
        }
    
    def project_faces(self, faces):
        """
        Project faces into the eigenspace
        
        Args:
            faces (numpy.ndarray): Face vectors to project
            
        Returns:
            numpy.ndarray: Projected face vectors in eigenspace
        """
        if self.eigenfaces is None or self.mean_face is None:
            raise ValueError("Eigenfaces model not trained")
        
        # Ensure faces is a 2D array
        if faces.ndim == 1:
            faces = faces.reshape(1, -1)
        
        # Normalize faces by subtracting mean face
        normalized_faces = faces - self.mean_face
        
        # Project faces onto eigenspace
        return np.dot(normalized_faces, self.eigenfaces.T)
    
    def project_face(self, face_vector):
        """
        Project a single face into the eigenspace
        
        Args:
            face_vector (numpy.ndarray): Face vector to project
            
        Returns:
            numpy.ndarray: Projected face vector in eigenspace
        """
        return self.project_faces(face_vector)[0]
    
    def recognize_face(self, face_vector, threshold=5.0):
        """
        Recognize a face by finding the closest match in the training set
        
        Args:
            face_vector (numpy.ndarray): Face vector to recognize
            threshold (float): Maximum distance to consider a match
            
        Returns:
            dict: Recognition result with person_id, distance, and recognition status
        """
        if len(self.projected_training_faces) == 0:
            raise ValueError("No training faces available")
        
        # Project the face
        projected_face = self.project_face(face_vector)
        
        # Calculate distances to all training faces
        distances = []
        for projected_training_face in self.projected_training_faces:
            distance = np.linalg.norm(projected_face - projected_training_face)
            distances.append(distance)
        
        # Find the closest match
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # If distance is below threshold, return the match
        if min_distance < threshold:
            return {
                'person_id': self.training_labels[min_distance_idx],
                'distance': float(min_distance),
                'recognized': True
            }
        else:
            return {
                'person_id': None,
                'distance': float(min_distance),
                'recognized': False
            }
    
    def load_model(self, eigenfaces_model):
        """
        Load an eigenfaces model from the database
        
        Args:
            eigenfaces_model (EigenfacesModel): Model instance to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if eigenfaces_model.mean_face and eigenfaces_model.eigenfaces:
            self.mean_face = np.array(eigenfaces_model.mean_face)
            self.eigenfaces = np.array(eigenfaces_model.eigenfaces)
            return True
        return False
    
    def save_model(self, name="default_model"):
        """
        Save the current eigenfaces model to the database
        
        Args:
            name (str): Name to identify the model
            
        Returns:
            EigenfacesModel: The saved model instance
        """
        from .models import EigenfacesModel
        
        # Create a new model
        model = EigenfacesModel(
            name=name,
            mean_face=self.mean_face.tolist(),
            eigenfaces=self.eigenfaces.tolist()
        )
        model.save()
        
        return model 