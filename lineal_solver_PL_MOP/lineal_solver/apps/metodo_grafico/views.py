from django.shortcuts import render
from .forms import ProblemaGraficoForm
from .grafico_solver import resolver_problema
import json

def grafico_solver_view(request):
    resultado = None
    vertices = None
    optimo = None

    if request.method == 'POST':
        form = ProblemaGraficoForm(request.POST)
        if form.is_valid():
            # Datos del formulario
            restricciones = json.loads(form.cleaned_data['restricciones'])
            funcion_objetivo = json.loads(form.cleaned_data['funcion_objetivo'])

            # Resolver problema
            vertices, optimo = resolver_problema(restricciones, funcion_objetivo)

            resultado = {
                'vertices': vertices,
                'optimo': optimo,
                'restricciones': restricciones,
                'funcion_objetivo': funcion_objetivo
            }
    else:
        form = ProblemaGraficoForm()

    context = {
        'form': form,
        'resultado': resultado,
        'vertices': json.dumps(vertices) if vertices else None,
        'optimo': json.dumps(optimo) if optimo else None
    }
    return render(request, 'metodo_grafico/grafico_solver.html', context)
