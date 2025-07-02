from django.shortcuts import render
import json
from .grafico_solver import resolver_problema
from .models import ProblemaGrafico

def grafico_solver_view(request):
    resultado = None
    vertices = None
    optimo = None
    paso_a_paso = None

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

            # --- PASO A PASO ---
            paso_a_paso = []
            # 1. Planteamiento
            paso_a_paso.append({
                "titulo": "Planteamiento del problema",
                "contenido": {
                    "Función objetivo": f"{'Maximizar' if optim == 'max' else 'Minimizar'} \\( Z = {obj[0]}x_1 + {obj[1]}x_2 \\)",
                    "Restricciones": [
                        f"\\( {r['lhs'][0]}x_1 + {r['lhs'][1]}x_2 {r['sign']} {r['rhs']} \\)" for r in restricciones
                    ]
                }
            })
            # 2. Despeje de restricciones
            despejes = []
            for idx, r in enumerate(restricciones):
                A, B = r["lhs"]
                signo = r["sign"]
                b = r["rhs"]
                if B != 0:
                    despeje = f"\\( x_2 {signo} \\frac{{{b} - {A}x_1}}{{{B}}} \\)"
                else:
                    despeje = f"x₁ {signo} {b/A if A != 0 else 'Indefinido'}"
                despejes.append(despeje)
            paso_a_paso.append({
                "titulo": "Despeje de restricciones para graficar",
                "contenido": despejes
            })

            # 3. Intersección de restricciones y región factible
            # Aquí debes tener funciones auxiliares en grafico_solver.py
            try:
                from .grafico_solver import parse_restricciones, encontrar_intersecciones, es_factible
                A, b_vec, signs = parse_restricciones(restricciones)
                candidatas = encontrar_intersecciones(A, b_vec, signs)
                factibles = [p for p in candidatas if es_factible(p, A, b_vec, signs)]
            except Exception:
                candidatas = []
                factibles = []

            paso_a_paso.append({
                "titulo": "Intersección de restricciones (vértices candidatos)",
                "contenido": [f"\\( ({round(p[0],4)},\\ {round(p[1],4)}) \\)" for p in candidatas]
            })
            paso_a_paso.append({
                "titulo": "Vértices factibles (región factible)",
                "contenido": [f"\\( ({round(p[0],4)},\\ {round(p[1],4)}) \\)" for p in factibles]
            })

            # 4. Evaluación de la función objetivo en cada vértice
            z_values = [obj[0]*p[0] + obj[1]*p[1] for p in factibles]
            evaluaciones = [
                {"vértice": f"\\( ({round(p[0],4)},\\ {round(p[1],4)}) \\)", "Z": f"\\( Z = {round(z,4)} \\)"}
                for p, z in zip(factibles, z_values)
            ]
            paso_a_paso.append({
                "titulo": "Evaluación de la función objetivo en cada vértice factible",
                "contenido": evaluaciones
            })

            # 5. Selección del óptimo
            if z_values:
                idx_optimo = z_values.index(max(z_values)) if optim == "max" else z_values.index(min(z_values))
                optimo = {
                    "x": factibles[idx_optimo][0],
                    "y": factibles[idx_optimo][1],
                    "z": z_values[idx_optimo]
                }
                paso_a_paso.append({
                    "titulo": "Selección del óptimo",
                    "contenido": {
                        "Vértice óptimo": f"\\( ({round(optimo['x'],4)},\\ {round(optimo['y'],4)}) \\)",
                        "Valor óptimo Z": f"\\( Z = {round(optimo['z'],4)} \\)"
                    }
                })
                vertices = [{"x": p[0], "y": p[1]} for p in factibles]
            else:
                optimo = None
                vertices = []

            # Resolver el problema gráfico (para compatibilidad con tu código actual)
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
        'optimo': json.dumps(optimo) if optimo is not None else "{}",
        'paso_a_paso': paso_a_paso
    }
    return render(request, 'grafico_solver.html', context)