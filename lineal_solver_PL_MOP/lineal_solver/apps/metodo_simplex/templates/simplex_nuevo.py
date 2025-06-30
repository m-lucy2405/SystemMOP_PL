# ------------------------------------------------------------
# Ejemplo resuelto del Método Simplex paso a paso
# ------------------------------------------------------------
# Problema:
# Max Z = 4x1 + 6x2
# sujeto a:
#   2x1 + 1x2 <= 6
#   1x1 + 2x2 <= 8
#   x1, x2 >= 0
# ------------------------------------------------------------

from tabulate import tabulate

# Tabla inicial
tabla = [
    [2, 1, 1, 0, 6],  # Fila 1 (s1)
    [1, 2, 0, 1, 8],  # Fila 2 (s2)
    [-4, -6, 0, 0, 0] # Fila Z
]
base = ["s1", "s2"]

def mostrar_tabla(tabla, base, titulo):
    headers = ["x1", "x2", "s1", "s2", "RHS"]
    filas = []
    for i in range(len(base)):
        filas.append([base[i]] + tabla[i])
    filas.append(["Z"] + tabla[-1])
    print(f"\n===== {titulo} =====\n")
    print(tabulate(filas, headers=["Base"] + headers, floatfmt=".2f", tablefmt="grid"))

def iteracion_simplex(tabla, base):
    z_fila = tabla[-1][:-1]
    col_entrante = z_fila.index(min(z_fila))

    razones = []
    for i in range(len(base)):
        valor = tabla[i][col_entrante]
        if valor > 0:
            razones.append(tabla[i][-1] / valor)
        else:
            razones.append(float("inf"))

    fila_saliente = razones.index(min(razones))
    pivote = tabla[fila_saliente][col_entrante]

    # Normalizar fila pivote
    tabla[fila_saliente] = [x / pivote for x in tabla[fila_saliente]]

    # Hacer ceros en la columna pivote
    for i in range(len(tabla)):
        if i != fila_saliente:
            mult = tabla[i][col_entrante]
            tabla[i] = [tabla[i][j] - mult * tabla[fila_saliente][j] for j in range(len(tabla[0]))]

    base[fila_saliente] = f"x{col_entrante+1}"

# Mostrar tabla inicial
mostrar_tabla(tabla, base, "Tabla Inicial")

# Iteración 1
print("\n>> Iteracion 1: entra x2 (mayor negativo), sale s2")
iteracion_simplex(tabla, base)
mostrar_tabla(tabla, base, "Despues de Iteracion 1")

# Iteración 2
print("\n>> Iteracion 2: entra x1, sale s1")
iteracion_simplex(tabla, base)
mostrar_tabla(tabla, base, "Despues de Iteracion 2")

# Mostrar solución óptima
solucion = {"x1": 0, "x2": 0}
for i, var in enumerate(base):
    if var.startswith("x"):
        idx = int(var[1:]) - 1
        solucion[f"x{idx+1}"] = tabla[i][-1]

z_optimo = tabla[-1][-1]

print("\nSolucion optima:")
for var, val in solucion.items():
    print(f"  {var} = {val}")
print(f"  Z = {z_optimo}")