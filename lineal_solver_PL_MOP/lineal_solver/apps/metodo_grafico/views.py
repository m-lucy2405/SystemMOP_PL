from django.shortcuts import render
from .forms import GraphicalInputForm

def graphical_solver(request):
    if request.method == 'POST':
        form = GraphicalInputForm(request.POST)
        if form.is_valid():
            # Procesar los datos del formulario
            # Implementar la lógica para resolver el problema gráficamente
            context = {
                'result': 'Result of the graphical solution',  # Marcador de posición para el resultado real
                'form': form
            }
            return render(request, 'metodo_grafico/result.html', context)
    else:
        form = GraphicalInputForm()

    return render(request, 'metodo_grafico/graphical_solver.html', {'form': form})