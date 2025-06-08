from django.urls import path
from . import views

urlpatterns = [
    path('templates/', views.metodo_simplex_view, name='metodo_simplex_index'),
]