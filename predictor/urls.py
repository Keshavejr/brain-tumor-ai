from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('predict/', views.predict_tumor),
    path('report/', views.generate_pdf),
]