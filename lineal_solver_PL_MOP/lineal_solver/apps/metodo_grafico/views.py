from django.shortcuts import render
from .forms import ProblemaGraficoForm
from .grafico_solver import resolver_problema
from .models import ProblemaGrafico
import json

def grafico_solver_view(request):
    resultado = None
    vertices = None
    optimo = None

    if request.method == 'POST':
        form = ProblemaGraficoForm(request.POST)
        if form.is_valid():
            problema = form.save(commit=False)

            if request.user.is_authenticated:
                problema.user = request.user  

            try:
                restricciones = json.loads(form.cleaned_data['restricciones'])
                funcion_objetivo = json.loads(form.cleaned_data['funcion_objetivo'])

                vertices, optimo = resolver_problema(restricciones, funcion_objetivo)

                problema.solucion_optima = optimo
                problema.vertices_factibles = vertices
                problema.save()

                resultado = {
                    'problema': problema,
                    'vertices': vertices,
                    'optimo': optimo,
                }
            except Exception as e:
                form.add_error(None, f"Error al resolver: {e}")
    else:
        form = ProblemaGraficoForm()

    context = {
        'form': form,
        'resultado': resultado,
        'vertices': json.dumps(vertices) if vertices is not None else "[]",
        'optimo': json.dumps(optimo) if optimo is not None else "{}"
    }
    return render(request, 'grafico_solver.html', context)
