from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import (
    authenticate, login as auth_login, logout, update_session_auth_hash
)
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from .models import PerfilUsuario
from .forms import PerfilUsuarioForm, CambiarUsuarioForm, CambiarContrasenaForm
from apps.metodo_simplex.models import SimplexProblem
from apps.metodo_grafico.models import ProblemaGrafico

# ----------------------------------------
# VISTA LOGIN
# ----------------------------------------
def login(request):
    storage = messages.get_messages(request)
    for _ in storage:
        pass
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            auth_login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Usuario o contraseña incorrectos.')

    return render(request, "login.html")


# ----------------------------------------
# VISTA REGISTER
# ----------------------------------------
def register(request):
    storage = messages.get_messages(request)
    for _ in storage:
        pass
    if request.method == 'POST':
        username = request.POST.get('username')
        nombre = request.POST.get('nombre')
        apellido = request.POST.get('apellido')
        correo = request.POST.get('correo')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if password1 != password2:
            messages.error(request, 'Las contraseñas no coinciden.')
            return redirect('register')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'El nombre de usuario ya está en uso.')
            return redirect('register')

        if User.objects.filter(email=correo).exists():
            messages.error(request, 'El correo electrónico ya está registrado.')
            return redirect('register')

       

        try:
            validate_password(password1, user=User(username=username, email=correo))
        except ValidationError as ve:
            traducciones = {
        "This password is too short. It must contain at least 8 characters.": "La contraseña es demasiado corta. Debe contener al menos 8 caracteres.",
        "This password is too common.": "La contraseña es demasiado común.",
        "This password is entirely numeric.": "La contraseña no puede ser solo numérica.",
            }
            for error in ve.messages:
                error_traducido = traducciones.get(error, error)
            messages.error(request, error_traducido)
            return redirect('register') 



        user = User.objects.create_user(
            username=username,
            email=correo,
            password=password1,
            first_name=nombre,
            last_name=apellido
        )

        PerfilUsuario.objects.create(user=user)

        usuario_autenticado = authenticate(request, username=username, password=password1)
        if usuario_autenticado is not None:
            auth_login(request, usuario_autenticado)
            
            return redirect('home')

        
        return redirect('login')

    return render(request, "register.html")


# ----------------------------------------
# PERFIL DEL USUARIO
# ----------------------------------------
@login_required
def perfil_view(request):
    # Limpia mensajes previos para evitar repetidos
    storage = messages.get_messages(request)
    for _ in storage:
        pass

    # Obtiene el perfil o lo crea si no existe
    perfil, _ = PerfilUsuario.objects.get_or_create(user=request.user)

    # Filtra problemas simplex del usuario
    problemas_simplex = SimplexProblem.objects.filter(user=request.user).order_by('-fecha')
    problemas_grafico = ProblemaGrafico.objects.filter(user=request.user).order_by('-fecha_creacion')
    

    return render(request, 'perfil.html', {
        'perfil': perfil,
        'problemas_simplex': problemas_simplex,
        'problemas_grafico': problemas_grafico,
    })


# ----------------------------------------
# EDITAR DATOS DEL PERFIL
# ----------------------------------------
@login_required
def editar_datos_view(request):
    storage = messages.get_messages(request)
    for _ in storage:
        pass
    perfil, _ = PerfilUsuario.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        form = PerfilUsuarioForm(request.POST, request.FILES, instance=perfil, user=request.user)
        if form.is_valid():
            request.user.first_name = form.cleaned_data['first_name']
            request.user.last_name = form.cleaned_data['last_name']
            request.user.save()
            form.save()
            
            return redirect('perfil')
    else:
        form = PerfilUsuarioForm(instance=perfil, user=request.user)

    return render(request, 'editar-datos.html', {'form': form})




# ----------------------------------------
# EDITAR NOMBRE DE USUARIO Y CONTRASEÑA
# ----------------------------------------

@login_required
def editar_usuario_view(request):
    storage = messages.get_messages(request)
    for _ in storage:
        pass
    if request.method == 'POST':
        form = CambiarUsuarioForm(request.POST, user=request.user)
        if form.is_valid():
            nuevo_username = form.cleaned_data['nuevo_username']
            request.user.username = nuevo_username
            request.user.save()
            
            return redirect('perfil')
    else:
        form = CambiarUsuarioForm(user=request.user)

    return render(request, 'editar-usuario.html', {'form_usuario': form})
@login_required
def editar_password_view(request):
    storage = messages.get_messages(request)
    for _ in storage:
        pass

    if request.method == 'POST':
        form = CambiarContrasenaForm(request.POST, user=request.user)
        if form.is_valid():
            nueva_password = form.cleaned_data['nueva_password']
            confirmar_password = form.cleaned_data['confirmar_password']

            if nueva_password != confirmar_password:
                messages.error(request, "Las contraseñas no coinciden.")
                return redirect('editar_password')

            try:
                validate_password(nueva_password, user=request.user)
            except ValidationError as ve:
                traducciones = {
                    "This password is too short. It must contain at least 8 characters.": "La contraseña es demasiado corta. Debe contener al menos 8 caracteres.",
                    "This password is too common.": "La contraseña es demasiado común.",
                    "This password is entirely numeric.": "La contraseña no puede ser solo numérica.",
                }
                for error in ve.messages:
                    messages.error(request, traducciones.get(error, error))
                return redirect('editar_password')

            request.user.set_password(nueva_password)
            request.user.save()
            update_session_auth_hash(request, request.user)
            messages.success(request, "Contraseña actualizada correctamente.")
            return redirect('perfil')
    else:
        form = CambiarContrasenaForm(user=request.user)

    return render(request, 'editar-password.html', {'form_contrasena': form})
   
# ----------------------------------------
# CERRAR SESIÓN
# ----------------------------------------
def logout_view(request):
    storage = messages.get_messages(request)
    for _ in storage:
        pass
    logout(request)
    return redirect('home')
