from django.shortcuts import render
from .models import SimplexProblem
from .simplex_es import (
    simplex_estandar,
    simplex_solve,
    plot_feasible_region,
)
import json

# ----------------------------------------------------------------------
# Vista principal
# ----------------------------------------------------------------------
def metodo_simplex_view(request):
    # ---------------- variables que envía la plantilla ----------------
    resultado = historial = grafica = None
    usa_granM = False
    error = explicacion_inf = explicacion_unb = None
    latex_sistema = latex_convertido = None

    # ==================================================================
    #   PETICIÓN POST  →  procesar formulario
    # ==================================================================
    if request.method == "POST":
        try:
            # ------------------------------------------------------------
            # 1. Leer datos del formulario
            # ------------------------------------------------------------
            n = int(request.POST["n"])
            m = int(request.POST["m"])
            optim = request.POST["optim"]          # 'min' | 'max'
            minimize = optim == "min"

            obj   = [float(request.POST[f"obj{i+1}"]) for i in range(n)]
            cons  = [[float(request.POST[f"cons{j+1}_{i+1}"]) for i in range(n)]
                     for j in range(m)]
            types = [request.POST[f"type{j+1}"] for j in range(m)]
            rhs   = [float(request.POST[f"rhs{j+1}"]) for j in range(m)]

            # ------------------------------------------------------------
            # 2. Construir LaTeX (formulación y forma convertida)
            # ------------------------------------------------------------
            latex_sistema    = construir_latex_sistema(n, m, obj, cons, types, rhs)
            latex_convertido = construir_latex_convertido(n, m, obj, cons, types, rhs)

            # ------------------------------------------------------------
            # 3. Elegir solver  (Estándar vs Gran-M)
            # ------------------------------------------------------------
            usa_granM = any(t in (">=", "=") for t in types)

            if usa_granM:
                sol, z, historial = simplex_solve(minimize, n, m, obj, cons, types, rhs)
            else:
                sol, z, historial = simplex_estandar(minimize, n, m, obj, cons, types, rhs)

            # ------------------------------------------------------------
            # 4. Interpretar el resultado del solver
            # ------------------------------------------------------------
            if sol is None:
                # El solver devolvió None → algo falló (infactible / unbounded / error)
                if historial and isinstance(historial[-1], dict):
                    ultimo = historial[-1]

                    # -------- caso no acotado --------------------------------
                    if ultimo.get("unbounded"):
                        error = "El problema no está acotado"
                        explicacion_unb = (
                            "La función objetivo puede crecer indefinidamente porque "
                            "alguna variable puede aumentar sin violar las restricciones."
                        )

                    # -------- caso infactible --------------------------------
                    elif ultimo.get("infeasible") or ultimo.get("explicacion_inf"):
                        error = "No existe solución factible"
                        # Usa la explicación generada por el solver si está disponible;
                        # de lo contrario, muestra un texto genérico.
                        explicacion_inf = ultimo.get(
                            "explicacion_inf",
                            "Las restricciones son incompatibles: "
                            "al finalizar la Fase I quedó al menos una variable "
                            "artificial con valor positivo."
                        )

                    # -------- otro error genérico ----------------------------
                    else:
                        error = ultimo.get("error", "No se encontró solución factible")
                else:
                    error = "No se encontró solución factible"
            else:
                resultado = {"sol": sol, "z": z}

            # ------------------------------------------------------------
            # 5. Preparar historial para la tabla de la plantilla
            # ------------------------------------------------------------
            if historial:
                for paso in historial:
                    if "basis" in paso and "tabla" in paso and "headers" in paso:
                        # Emparejamos cada fila de la tabla con su variable básica
                        paso["base_filas"] = list(zip(paso["basis"], paso["tabla"][:-1]))
                        # La fila Z-Cj (última) y los headers ya vienen listos

            # ------------------------------------------------------------
            # 6. Generar gráfica (sólo si n == 2 y hubo solución)
            # ------------------------------------------------------------
            if n == 2 and resultado and not error:
                try:
                    grafica = plot_feasible_region(obj, cons, types, rhs, resultado["sol"])
                except Exception as e:
                    print(f"Error generando gráfica: {e}")

            # ------------------------------------------------------------
            # 7. Guardar en BD si todo fue bien
            # ------------------------------------------------------------
            if resultado and not error:
                SimplexProblem.objects.create(
                    optim=optim,
                    n=n,
                    m=m,
                    obj=",".join(map(str, obj)),
                    cons="\n".join(",".join(map(str, fila)) for fila in cons),
                    types=",".join(types),
                    rhs=",".join(map(str, rhs)),
                    resultado=json.dumps(resultado),
                )

        # ------------------- manejo de excepciones ----------------------
        except ValueError as ve:
            error = f"Datos numéricos inválidos: {ve}"
        except KeyError as ke:
            error = f"Faltan campos en el formulario: {ke}"
        except Exception as e:
            error = f"Error inesperado: {str(e)}"

    # ==================================================================
    #   GET o POST con errores  →  renderizar plantilla
    # ==================================================================
    return render(
        request,
        "simplex_form.html",
        {
            "resultado":        resultado,
            "historial":        historial,
            "usa_granM":        usa_granM,
            "grafica":          grafica,
            "error":            error,
            "explicacion_inf":  explicacion_inf,
            "explicacion_unb":  explicacion_unb,
            "latex_sistema":    latex_sistema,
            "latex_convertido": latex_convertido,
        },
    )

# ----------------------------------------------------------------------
# Utilidades LaTeX  (sin cambios)
# ----------------------------------------------------------------------
def construir_latex_sistema(n, m, obj, cons, types, rhs):
    partes = []
    for i in range(m):
        lhs = " + ".join(f"{cons[i][j]}x_{{{j+1}}}" for j in range(n))
        partes.append(f"{lhs} {types[i]} {rhs[i]}")
    sistema = r"\left\{\begin{array}{l}" + r"\\ ".join(partes) + r"\end{array}\right."
    obj_expr = " + ".join(f"{obj[j]}x_{{{j+1}}}" for j in range(n))
    return (
        r"\begin{aligned}"
        r"\text{Optimizar } & Z = " + obj_expr + r" \\ "
        r"\text{Sujeto a:} & " + sistema + r"\end{aligned}"
    )

def construir_latex_convertido(n, m, obj, cons, types, rhs):
    ecuaciones = []
    s = e = a = 1
    for i in range(m):
        lhs = [f"{cons[i][j]}x_{{{j+1}}}" for j in range(n)]
        if types[i] == "<=":
            lhs.append(f"s_{{{s}}}"); s += 1
        elif types[i] == ">=":
            lhs.append(f"-e_{{{e}}} + a_{{{a}}}"); e += 1; a += 1
        else:  # "="
            lhs.append(f"a_{{{a}}}"); a += 1
        ecuaciones.append(" + ".join(lhs).replace("+ -", "- ") + f" = {rhs[i]}")
    sistema = r"\left\{\begin{array}{l}" + r"\\ ".join(ecuaciones) + r"\end{array}\right."
    obj_expr = " + ".join(f"{obj[j]}x_{{{j+1}}}" for j in range(n))
    return (
        r"\begin{aligned}"
        r"\text{Optimizar } & Z = " + obj_expr + r" \\ "
        r"\text{Sujeto a:} & " + sistema + r"\end{aligned}"
    )