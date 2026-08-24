from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_disease, name='predict_disease'),
    path('history/', views.get_history, name='get_history'),
    path('history/<int:pk>/', views.delete_prediction, name='delete_prediction'),
]