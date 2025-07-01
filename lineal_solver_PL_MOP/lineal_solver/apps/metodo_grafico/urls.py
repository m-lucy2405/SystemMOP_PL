from django.urls import path
from . import views

urlpatterns = [
    path('', views.metodo_grafico_view, name='metodo_grafico_index'),
    #path('solve/', views.solve, name='metodo_grafico_solve'),
]