from django.urls import path
from . import views

# URL - parametros para el metodo grafico

urlpatterns = [
    path('', views.grafico_solver_view, name='grafico_solver'),
]
