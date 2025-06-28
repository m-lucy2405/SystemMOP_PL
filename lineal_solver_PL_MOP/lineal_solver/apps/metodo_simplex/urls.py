from django.urls import path
from . import views

urlpatterns = [
    path('', views.metodo_simplex_view, name='metodo_simplex_index'),
]