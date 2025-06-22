from django.contrib import admin
from django.urls import path, include
from lineal_solver import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('simplex/', include('apps.metodo_simplex.urls')),
    path('grafico/', include('apps.metodo_grafico.urls')),
    path('', views.index, name='index'),
]