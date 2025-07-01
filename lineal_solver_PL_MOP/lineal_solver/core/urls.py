from django.contrib import admin
from django.urls import path, include
from lineal_solver import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('admin/', admin.site.urls),
    path('usuario/', include('apps.usuarios.urls'), name='login'),
    path('simplex/', include('apps.metodo_simplex.urls')),
    path('grafico/', include('apps.metodo_grafico.urls')),
    path('', views.home, name='home'),
    path('usuario/', include('apps.usuarios.urls'), name='login'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)