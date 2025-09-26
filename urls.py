from django.urls import path
from .views import detect_fake, home

urlpatterns = [
    path('', home, name='home'),  # Root URL mapped to home view
    path('predict/', detect_fake, name='detect_fake'),  # Prediction URL
]
