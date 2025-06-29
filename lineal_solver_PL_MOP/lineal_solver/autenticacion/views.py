# autenticacion/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import logout




def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            auth_login(request, user)
            return redirect('home')  
        else:
            messages.error(request, 'Usuario o contraseña incorrectos.')

    return render(request, "autenticacion/login.html")

def register(request):
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

        user = User.objects.create_user(
            username=username,
            email=correo,
            password=password1,
            first_name=nombre,
            last_name=apellido
        )
        user.save()
        messages.success(request, 'Usuario registrado correctamente.')
        return redirect('login')

    return render(request, "autenticacion/register.html")

def logout_view(request):
    logout(request)
    return redirect('home')  # o a donde quieras redirigir después de cerrar sesión

