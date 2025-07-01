from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path('login/', views.login, name='login'),  
    path('register/', views.register, name='register'),  
    path('logout/', views.logout_view, name='logout'),
    path('perfil/', views.perfil_view, name='perfil'),
    path('datos/', views.editar_datos_view, name='editar_datos'),
    path('usuario/',views.editar_usuario_view,name='editar_usuario'),
    path('pasword/',views.editar_password_view,name='editar_password')
] 