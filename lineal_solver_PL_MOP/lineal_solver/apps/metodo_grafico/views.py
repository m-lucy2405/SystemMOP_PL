from django.shortcuts import render
import json
from .grafico_solver import resolver_problema
from .models import ProblemaGrafico

def grafico_solver_view(request):
    resultado = None
    vertices = None
    optimo = None

    if request.method == 'POST':
        try:
            n = int(request.POST.get('n', 2))
            m = int(request.POST.get('m', 2))
            optim = request.POST.get('optim')

            # Obtener coeficientes de la función objetivo
            obj = [float(request.POST.get(f'obj{i+1}', 0)) for i in range(n)]

            # Obtener restricciones
            cons = []
            types = []
            rhs = []
            for j in range(1, m+1):
                cons.append([float(request.POST.get(f'cons{j}_{i+1}', 0)) for i in range(n)])
                types.append(request.POST.get(f'type{j}'))
                rhs.append(float(request.POST.get(f'rhs{j}', 0)))

            restricciones = []
            for i in range(m):
                restricciones.append({
                    "lhs": cons[i],
                    "sign": types[i],
                    "rhs": rhs[i]
                })

            # Resolver el problema gráfico
            vertices, optimo = resolver_problema(restricciones, obj)

            resultado = {
                'vertices': vertices,
                'optimo': optimo,
            }

            # Guardar en la base de datos solo si el usuario está autenticado
            if request.user.is_authenticated:
                ProblemaGrafico.objects.create(
                    user=request.user,
                    optim=optim,
                    n=n,
                    m=m,
                    obj=json.dumps(obj),
                    cons=json.dumps(cons),
                    types=json.dumps(types),
                    rhs=json.dumps(rhs),
                    resultado=json.dumps(resultado)
                )

        except Exception as e:
            resultado = {'error': str(e)}

    context = {
        'resultado': resultado,
        'vertices': json.dumps(vertices) if vertices is not None else "[]",
        'optimo': json.dumps(optimo) if optimo is not None else "{}"
    }
    return render(request, 'grafico_solver.html', context)
