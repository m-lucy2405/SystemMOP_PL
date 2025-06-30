from django.contrib import admin
from django.urls import path, include
from core import views

urlpatterns = [
    path('', views.home, name='home'),
    path('admin/', admin.site.urls),
    path('usuario/', include('apps.usuarios.urls'), name='login'),
    path('simplex/', include('apps.metodo_simplex.urls')),
    path('grafico/', include('apps.metodo_grafico.urls')),
]