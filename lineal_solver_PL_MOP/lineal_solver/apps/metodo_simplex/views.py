from django.shortcuts import render
from django.http import HttpResponse
from .forms import SimplexForm
from .simplex_solver import solve_simplex

def simplex_view(request):
    """
    Vista para resolver problemas de programación lineal usando el método símplex.
    """
    if request.method == 'POST':
        form = SimplexForm(request.POST)
        if form.is_valid():
            # Extraer datos del formulario
            objective_function = form.cleaned_data['objective_function']
            constraints = form.cleaned_data['constraints']
            # Resolver el problema de programación lineal
            solution = solve_simplex(objective_function, constraints)
            return render(request, 'metodo_simplex/result.html', {'solution': solution})
    else:
        form = SimplexForm()
    
    return render(request, 'metodo_simplex/index.html', {'form': form})