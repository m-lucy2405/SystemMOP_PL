from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('simplex/', include('apps.metodo_simplex.urls')),
    path('grafico/', include('apps.metodo_grafico.urls')),
]