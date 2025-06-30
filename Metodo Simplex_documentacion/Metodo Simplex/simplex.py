from typing import List, Tuple
import pandas as pd

# VARIABLE GLOBAL PARA DEFINIR DECIMALES
decimal_places = 2  # <--- cambia esto a 3, 4, etc., según lo que necesites

def format_value(value: float) -> str:
    return f"{value:.{decimal_places}f}"

def prepare_initial_table(obj: List[float], constraints: List[List[float]]) -> Tuple[List[List[float]], List[str]]:
    num_vars = len(obj)
    num_constraints = len(constraints)

    matrix = []
    for i, row in enumerate(constraints):
        vars_part = row[:-1]
        rhs = row[-1]
        slack = [0] * num_constraints
        slack[i] = 1
        matrix.append(vars_part + slack + [rhs])

    z_row = [-c for c in obj] + [0] * (num_constraints + 1)
    matrix.append(z_row)

    var_names = [f"x{i+1}" for i in range(num_vars)]
    return matrix, var_names

def generate_html_simplex_report(matrix: List[List[float]], var_names: List[str], obj: List[float], constraints: List[List[float]]) -> str:
    num_constraints = len(matrix) - 1
    num_vars = len(var_names)

    slack_vars = [f"s{i+1}" for i in range(num_constraints)]
    all_vars = var_names + slack_vars + ["RHS"]
    table = pd.DataFrame(matrix, columns=all_vars)

    basis = slack_vars.copy()

    html_output = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #003366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #999; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .pivot {{ background-color: #ffcccc; font-weight: bold; }}
            .iteration-title {{ background-color: #003366; color: white; padding: 8px; }}
        </style>
    </head>
    <body>
        <h1>Método Simplex - Informe Paso a Paso</h1>
        <p><strong>Decimales mostrados:</strong> {decimal_places}</p>
    """

    # MODELO MATEMÁTICO
    obj_expr = " + ".join(f"{format_value(c)}·{v}" for c, v in zip(obj, var_names))
    constraints_expr = []
    for row in constraints:
        lhs = " + ".join(f"{format_value(val)}·{var}" for val, var in zip(row[:-1], var_names))
        rhs = format_value(row[-1])
        constraints_expr.append(f"{lhs} ≤ {rhs}")

    # Restricciones con variables de holgura
    transformed_expr = []
    for i, row in enumerate(constraints):
        lhs = " + ".join(f"{format_value(val)}·{var}" for val, var in zip(row[:-1], var_names))
        lhs += f" + 1·s{i+1}"
        rhs = format_value(row[-1])
        transformed_expr.append(f"{lhs} = {rhs}")

    html_output += "<h2>Modelo Matemático</h2>"
    html_output += f"<p><strong>Función objetivo:</strong> Max Z = {obj_expr}</p>"
    html_output += "<p><strong>Sujeto a:</strong></p><ul>"
    for cons in constraints_expr:
        html_output += f"<li>{cons}</li>"
    html_output += "</ul>"

    html_output += "<h2>Modelo con variables de holgura</h2><ul>"
    for t in transformed_expr:
        html_output += f"<li>{t}</li>"
    html_output += "</ul>"

    # Iteraciones
    iteration = 0
    while True:
        html_output += f'<div class="iteration-title">Iteración {iteration}</div>'
        html_output += "<table><thead><tr><th>Base</th>"
        for col in table.columns:
            html_output += f"<th>{col}</th>"
        html_output += "</tr></thead><tbody>"

        for i in range(len(table)):
            html_output += f"<tr><td>{basis[i] if i < len(basis) else 'Z'}</td>"
            for col in table.columns:
                val = table.at[i, col]
                html_output += f"<td>{format_value(val)}</td>"
            html_output += "</tr>"
        html_output += "</tbody></table>"

        last_row = table.iloc[-1, :-1]
        pivot_col_name = last_row.idxmin()
        if table.at[len(table) - 1, pivot_col_name] >= 0:
            html_output += "<p><strong>Solución óptima encontrada.</strong></p>"
            break

        ratios = []
        for i in range(len(table) - 1):
            col_val = table.at[i, pivot_col_name]
            if col_val > 0:
                ratios.append(table.at[i, "RHS"] / col_val)
            else:
                ratios.append(float('inf'))

        pivot_row = ratios.index(min(ratios))
        pivot_element = table.at[pivot_row, pivot_col_name]

        entering = pivot_col_name
        leaving = basis[pivot_row]

        html_output += f"<p>Variable que entra: <strong>{entering}</strong> | Variable que sale: <strong>{leaving}</strong></p>"
        html_output += f"<p>Elemento pivote: <strong>{format_value(pivot_element)}</strong> en fila {pivot_row + 1}, columna {pivot_col_name}</p>"

        old_pivot_row = table.iloc[pivot_row].copy()

        html_output += f"<h3>Normalización de la fila pivote F{pivot_row+1}:</h3><ul>"
        new_pivot_row = []
        for col in table.columns:
            original_val = table.at[pivot_row, col]
            new_val = original_val / pivot_element
            new_pivot_row.append(new_val)
            html_output += f"<li>{col}: {format_value(original_val)} / {format_value(pivot_element)} = {format_value(new_val)}</li>"
        table.iloc[pivot_row] = new_pivot_row
        html_output += "</ul>"

        html_output += f"<h3>Actualización de las otras filas:</h3>"
        for i in range(len(table)):
            if i != pivot_row:
                factor = table.at[i, pivot_col_name]
                html_output += f"<h4>F{i+1} = F{i+1} - ({format_value(factor)}) × F{pivot_row+1}</h4><ul>"
                original_row = table.iloc[i].copy()
                new_row = []
                for col in table.columns:
                    old_val = original_row[col]
                    pivot_val = table.at[pivot_row, col]
                    result = old_val - factor * pivot_val
                    new_row.append(result)
                    html_output += f"<li>{col}: {format_value(old_val)} - ({format_value(factor)} × {format_value(pivot_val)}) = {format_value(result)}</li>"
                table.iloc[i] = new_row
                html_output += "</ul>"

        basis[pivot_row] = entering
        iteration += 1

    # RESUMEN FINAL
    html_output += "<h2>Solución Óptima</h2><ul>"

    solution = {var: 0.0 for var in var_names}
    for i, var in enumerate(basis):
        if var in solution:
            solution[var] = table.at[i, "RHS"]

    for var in var_names:
        html_output += f"<li>{var} = <strong>{format_value(solution[var])}</strong></li>"

    z_value = table.at[len(table) - 1, "RHS"]
    html_output += f"<li><strong>Valor máximo de la función objetivo Z = {format_value(z_value)}</strong></li>"
    html_output += "</ul>"

    html_output += "</body></html>"
    return html_output

# === EJEMPLO DE USO ===
objetivo = [7, 10]
restricciones = [
    [4, 5, 200],
    [6, 3, 240],
]

matrix, var_names = prepare_initial_table(objetivo, restricciones)
html_report = generate_html_simplex_report(matrix, var_names, objetivo, restricciones)

with open("simplex_report_ejemplo.html", "w", encoding="utf-8") as f:
    f.write(html_report)

print("Reporte generado: simplex_report_ejemplo.html")
