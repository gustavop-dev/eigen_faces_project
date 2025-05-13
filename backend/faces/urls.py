from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train_eigenfaces_model, name='train_eigenfaces_model'),
    path('process/', views.process_image, name='process_image'),
    path('recognize/', views.recognize_face, name='recognize_face'),
    path('visualize/', views.visualize_eigenfaces, name='visualize_eigenfaces'),
    path('visualize/<int:model_id>/', views.visualize_eigenfaces, name='visualize_eigenfaces_with_id'),
] 