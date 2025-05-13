from django.db import models

# Create your models here.

class Person(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class FaceImage(models.Model):
    person = models.ForeignKey(Person, related_name='face_images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='faces/')
    processed = models.BooleanField(default=False)
    features_vector = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Face of {self.person.name}"

class EigenfacesModel(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    mean_face = models.JSONField(null=True, blank=True)
    eigenfaces = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return self.name
