from django.contrib import admin
from .models import Person, FaceImage, EigenfacesModel

@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    search_fields = ('name',)

@admin.register(FaceImage)
class FaceImageAdmin(admin.ModelAdmin):
    list_display = ('person', 'processed', 'created_at')
    list_filter = ('processed', 'person')
    search_fields = ('person__name',)

@admin.register(EigenfacesModel)
class EigenfacesModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_active', 'created_at')
    list_filter = ('is_active',)
    search_fields = ('name',)
