from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='metodo_simplex_index'),
    path('solve/', views.solve, name='metodo_simplex_solve'),
]