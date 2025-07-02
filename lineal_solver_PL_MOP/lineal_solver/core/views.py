from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def index(request):
    return render(request, 'index.html')

def documentacion(request):
    return render(request, 'documentacion.html')