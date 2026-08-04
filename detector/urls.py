from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_disease, name='predict_disease'),
]