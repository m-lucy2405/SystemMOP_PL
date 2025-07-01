# -----------------------------------------------------------------------------
# BLOQUE 1: IMPORTACIONES Y CONFIGURACIÓN INICIAL
# -----------------------------------------------------------------------------

# Esta línea especifica que el script puede ser ejecutado directamente como programa
#!/usr/bin/env python3

# Permite usar anotaciones de tipo más avanzadas con compatibilidad hacia versiones anteriores
from __future__ import annotations

# Importa el tipo Fraction para trabajar con números racionales exactos (evita errores de punto flotante)
from fractions import Fraction

# Librerías estándar para entrada/salida en memoria y codificación de imágenes en base64
import io, base64

# Importación de tipos para anotaciones de parámetros y retorno de funciones
from typing import List, Tuple, Sequence, Dict

# Importa numpy para cálculos numéricos con arrays y matrices
import numpy as np

# Importa matplotlib para generar gráficas (usado en la representación gráfica de la región factible)
import matplotlib.pyplot as plt

# Define una constante muy grande que representa el valor de "M" en el método de la Gran M
# Esta constante se usa para penalizar artificiales simbólicamente cuando se convierte a valor numérico
BIG_M_NUMERIC = 1_000_000

# Límite de iteraciones que puede realizar el algoritmo (para evitar bucles infinitos)
MAX_ITERS    = 1000

# -----------------------------------------------------------------------------
# BLOQUE 2: CLASE MixedValue – Soporte para expresiones con 'M'
# -----------------------------------------------------------------------------

# Esta clase representa una expresión del tipo: a + bM,
# donde:
#   - a: parte numérica real
#   - b: coeficiente de una constante simbólica M (muy grande)
# Se utiliza en el método de la Gran M para manejar variables artificiales simbólicamente
class MixedValue:

    def __init__(self, a: int | Fraction = 0, b: int | Fraction = 0):
        # Inicializa un MixedValue con parte real 'a' y parte simbólica 'b'
        # Se convierte a fracción para evitar errores de redondeo decimal
        self.a = Fraction(a).limit_denominator()
        self.b = Fraction(b).limit_denominator()
    def __add__(self, o):
        # Suma de dos MixedValue: (a1 + b1M) + (a2 + b2M) = (a1+a2) + (b1+b2)M
        return MixedValue(self.a+o.a, self.b+o.b)

    def __sub__(self, o):
        # Resta de dos MixedValue: (a1 + b1M) - (a2 + b2M)
        return MixedValue(self.a-o.a, self.b-o.b)

    def __mul__(self, o):
        # Multiplicación de MixedValue con otro MixedValue o escalar:
        if isinstance(o, MixedValue):
            # Solo multiplicamos la parte real y cruzamos los términos de M:
            # (a + bM)(c + dM) = ac + (ad + bc)M + bdM^2 (se ignora término M^2)
            return MixedValue(self.a*o.a, self.a*o.b + self.b*o.a)
        # Multiplicación por escalar (número real): escalar * (a + bM)
        return MixedValue(self.a*o, self.b*o)

    __rmul__ = __mul__ # Permite multiplicación conmutativa: escalar * MixedValue

    def __truediv__(self, o):
        # División entre MixedValue y escalar o MixedValue sin término M
        if isinstance(o, MixedValue):
            # Solo se permite dividir si la parte simbólica de o es cero
            if o.b != 0: raise ZeroDivisionError
            return MixedValue(self.a/o.a, self.b/o.a)
        return MixedValue(self.a/o, self.b/o)

    def is_positive(self):
        # Verifica si el MixedValue es positivo: primero se evalúa el signo de M
        return (self.b>0) or (self.b==0 and self.a>0)

    def is_negative(self):
        # Verifica si el MixedValue es negativo: se prioriza el signo de M
        return (self.b<0) or (self.b==0 and self.a<0)

    def __lt__(self, o):
        # Compara si este MixedValue es menor que otro (se evalúa b antes que a)
        if self.b!=o.b: return self.b<o.b
        return self.a<o.a

    def __eq__(self, o):
        # Verifica si dos MixedValue son exactamente iguales (ambas partes iguales)
        return isinstance(o, MixedValue) and self.a==o.a and self.b==o.b

    def __float__(self):
        # Convierte el valor simbólico a un número real aproximado usando BIG_M_NUMERIC
        return float(self.a + self.b*BIG_M_NUMERIC)

    def __str__(self):
        # Retorna una representación legible del MixedValue como string:
        # Por ejemplo: "3M+2", "-M+1", "5", "M", etc.
        parts=[]
        if self.b!=0:
            parts.append(f"{self.b}M" if abs(self.b)!=1 else ("M" if self.b>0 else "-M"))
        if self.a!=0:
            sign = "" if not parts else ("+" if self.a>0 else "")
            parts.append(f"{sign}{self.a}")
        return "".join(parts) or "0" # Retorna "0" si ambos son cero

# # -----------------------------------------------------------------------------
# BLOQUE 3: FUNCIÓN build_bigM_tableau – Construcción del tableau inicial
# -----------------------------------------------------------------------------

# Esta función construye el tableau (matriz aumentada) inicial para resolver
# un problema de programación lineal con restricciones de cualquier tipo (≤, ≥, =)
# usando el método de la Gran M. Se agregan variables de holgura, exceso y artificiales
# según sea necesario, y se arma también la fila Z–Cj para la fase I.
def build_bigM_tableau(minimize, n, m, obj, cons, types, rhs):
    s_count = sum(1 for t in types if t == "<=")
    e_count = sum(1 for t in types if t == ">=")
    a_count = sum(1 for t in types if t in (">=", "="))

    total = n + s_count + e_count + a_count + 1
    pos_s = n
    pos_e = n + s_count
    pos_a = n + s_count + e_count
    pos_b = total - 1

    headers = (
        [f"x{i+1}" for i in range(n)] 
        + [f"s{i+1}" for i in range(s_count)]
        + [f"e{i+1}" for i in range(e_count)]
        + [f"a{i+1}" for i in range(a_count)]
        + ["b"]
    )

    tableau = []
    basis = []
    si = ei = ai = 0

    for i, t in enumerate(types):
        if rhs[i] < 0:
            cons[i] = [-c for c in cons[i]]
            rhs[i] = -rhs[i]
            t = {"<=": ">=", ">=": "<=", "=": "="}[t]
            types[i] = t

        row = [MixedValue(0, 0) for _ in range(total)]
        for j in range(n):
            row[j] = MixedValue(cons[i][j], 0)
        row[pos_b] = MixedValue(rhs[i], 0)

        if t == "<=":
            row[pos_s + si] = MixedValue(1, 0)
            basis.append(f"s{si+1}")
            si += 1
        elif t == ">=":
            row[pos_e + ei] = MixedValue(-1, 0)
            row[pos_a + ai] = MixedValue(1, 0)
            basis.append(f"a{ai+1}")
            ei += 1
            ai += 1
        else:  # "="
            row[pos_a + ai] = MixedValue(1, 0)
            basis.append(f"a{ai+1}")
            ai += 1
            
        tableau.append(row)

    M = 10**6
    Cj = [MixedValue(c, 0) for c in obj] + [MixedValue(0, 0)]*(s_count + e_count) + [MixedValue(0, M if minimize else -M)]*a_count + [MixedValue(0, 0)]
    Cb = [MixedValue(0, M if minimize else -M) if b.startswith("a") else MixedValue(0, 0) for b in basis]

    Zj = [MixedValue(0, 0) for _ in range(total)]
    for j in range(total):
        for i in range(m):
            Zj[j] += Cb[i] * tableau[i][j]

    Zc = [Zj[j] - Cj[j] for j in range(total)]
    tableau.append(Zc)

    return tableau, basis, headers

def find_pivot(tableau, basis, headers, minimize):
    m = len(tableau) - 1
    z_row = tableau[-1]
    
    def is_improving(val):
        if minimize:
            return val.is_positive()
        return val.is_negative()

    candidates = []
    for j in range(len(z_row)-1):
        if headers[j] not in basis and is_improving(z_row[j]):
            priority = (z_row[j].b, abs(z_row[j].a), -j)
            candidates.append((priority, j))
    
    if not candidates:
        return None, None
    
    pc = min(candidates, key=lambda x: x[0])[1]

    ratios = []
    for i in range(m):
        a_ij = tableau[i][pc]
        if a_ij.is_positive():
            try:
                ratio = tableau[i][-1] / a_ij
                ratios.append((float(ratio), i))
            except ZeroDivisionError:
                continue
    
    if not ratios:
        raise Exception("El problema es no acotado")
    
    pr = min(ratios, key=lambda x: (x[0], x[1]))[1]
    return pr, pc


# -----------------------------------------------------------------------------
# BLOQUE 4: FUNCIÓN find_pivot – Selección del elemento pivote (Gran M)
# -----------------------------------------------------------------------------

# Esta función se utiliza durante las iteraciones del método Símplex (fase I o II)
# para seleccionar la posición del elemento pivote. El pivote es el valor central
# alrededor del cual se harán transformaciones para acercarse a la solución óptima.
def find_pivot(tableau, basis, headers, minimize):
    """
    Encuentra el elemento pivote para la iteración actual del método Simplex.
    
    Args:
        tableau: Matriz completa del tableau
        basis: Lista de variables básicas actuales
        headers: Nombres de las columnas
        minimize: Booleano que indica si es problema de minimización
        
    Returns:
        Tuple (pr, pc): Fila y columna del pivote, o (None, None) si es óptimo
        
    Raises:
        Exception: Si el problema es no acotado
    """
    m = len(tableau) - 1  # Número de restricciones (excluyendo fila Z)
    z_row = tableau[-1]   # Fila Z - Cj
    
    # ─── [MEJORA 1] Selección más robusta de columna pivote ───
    def is_improving(val):
        """Determina si un valor Z-Cj mejora la solución"""
        if minimize:
            return val.is_positive()  # Minimización: busca valores positivos
        return val.is_negative()      # Maximización: busca valores negativos

    # Generamos todas las columnas candidatas con sus valores
    candidates = []
    for j in range(len(z_row)-1):  # Excluimos la columna RHS
        if headers[j] not in basis and is_improving(z_row[j]):
            # [MEJORA 2] Usamos tupla (prioridad, valor, j) para ordenación
            priority = (
                z_row[j].b,  # Primero componente M (para Gran M)
                abs(z_row[j].a),  # Luego magnitud del coeficiente
                -j  # Finalmente índice (para desempate)
            )
            candidates.append((priority, j))
    
    if not candidates:
        return None, None  # Solución óptima alcanzada
    
    # [MEJORA 3] Selección por mayor mejora potencial
    pc = min(candidates, key=lambda x: x[0])[1]

    # ─── [MEJORA 4] Selección de fila pivote con razones exactas ───
    ratios = []
    for i in range(m):
        a_ij = tableau[i][pc]
        if a_ij.is_positive():  # Solo consideramos elementos positivos
            try:
                ratio = tableau[i][-1] / a_ij
                # [MEJORA 5] Usamos tupla (ratio, i) para ordenación estable
                ratios.append((float(ratio), i))
            except ZeroDivisionError:
                continue
    
    if not ratios:
        raise Exception("El problema es no acotado")
    
    # Seleccionamos la fila con menor ratio (usamos índice como desempate)
    pr = min(ratios, key=lambda x: (x[0], x[1]))[1]
    
    return pr, pc

# -----------------------------------------------------------------------------
# BLOQUE 5: FUNCIÓN simplex_estandar – Método Símplex estándar (solo ≤)
# -----------------------------------------------------------------------------

# Esta función resuelve un problema de programación lineal utilizando el método
# Símplex estándar, que solo permite restricciones del tipo ≤.
# No utiliza variables artificiales ni la técnica de la Gran M.
def simplex_estandar(
    minimize: bool, # True si se desea minimizar, False si se desea maximizar
    n: int, # Cantidad de variables reales (x1, x2, ...)
    m: int, # Cantidad de restricciones
    obj: Sequence[float], # Lista de coeficientes de la función objetivo
    cons: Sequence[Sequence[float]], # Matriz de coeficientes de restricciones (m x n)
    types: Sequence[str], # Tipos de restricciones (solo se acepta '<=')
    rhs: Sequence[float], # Lado derecho de las restricciones (vector b)
) -> Tuple[dict,float,List[dict]]: # Retorna solución, valor óptimo y historial

    from fractions import Fraction # Se usa para evitar errores de redondeo

    # ─── Validaciones iniciales ───
    if any(t!="<=" for t in types):
        return None,None,[{"error":"Estándar sólo ≤"}]
    if any(b<0 for b in rhs):
        return None,None,[{"error":"RHS<0 no soportado"}]

    # ─── Construcción del tableau ───
    total = n+m
    headers = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["b"]
    tableau = []
    basis = [f"s{i+1}" for i in range(m)]

    # Construir restricciones
    for i in range(m):
        row = [Fraction(0)]*(total+1)
        for j in range(n): 
            row[j] = Fraction(cons[i][j]).limit_denominator()
        row[n+i] = Fraction(1)
        row[-1] = Fraction(rhs[i]).limit_denominator()
        tableau.append(row)

    # Fila Z – Cj inicial CORREGIDA (clave para el resultado correcto)
    sign = -1 if not minimize else 1  # Maximización: signo negativo
    zrow = [Fraction(obj[j]).limit_denominator() * sign for j in range(n)] + [Fraction(0)]*(m+1)
    tableau.append(zrow)

    historial=[{
        "tabla": [[str(c) for c in row] for row in tableau],
        "basis": basis.copy(),
        "headers": headers.copy(),
        "pivote": None,
        "operaciones_filas":[]
    }]

    extra = False

    # ─── Ciclo principal de iteraciones ───
    while True:
        # Buscar posición del pivote
        pr, pc = find_pivot_frac(tableau, minimize, basis, headers)

        if pr is None:
            if extra: break
            Zc = tableau[-1][:-1]
            pc = next((j for j,v in enumerate(Zc) if v==0 and any(tableau[i][j]>0 for i in range(m))), None)
            if pc is None: break
            pr = min((tableau[i][-1]/tableau[i][pc],i) for i in range(m) if tableau[i][pc]>0)[1]
            extra = True

        # Pivoteo
        ops = []
        piv = tableau[pr][pc]
        old = tableau[pr].copy()
        new = [v/piv for v in old]
        labels = headers+["b"]

        # Formateo mejorado de las operaciones
        def format_val(val):
            f = Fraction(val).limit_denominator()
            return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else f"{f.numerator}"

        divs = [f"{labels[j]}: {format_val(old[j])}÷{format_val(piv)}={format_val(new[j])}" 
               for j in range(len(old))]
        ops.append(f"Fila{pr+1}Norm:\n" + "\n".join(divs))
        tableau[pr] = new

        for i in range(len(tableau)):
            if i == pr: continue
            fct = tableau[i][pc]
            old_i = tableau[i].copy()
            new_i = [old_i[j]-fct*tableau[pr][j] for j in range(len(old_i))]
            subs = [f"{labels[j]}: {format_val(old_i[j])}−{format_val(fct)}×{format_val(tableau[pr][j])}={format_val(new_i[j])}"
                   for j in range(len(old_i))]
            ops.append(f"Fila{i+1}Act:\n" + "\n".join(subs))
            tableau[i] = new_i

        # Actualizar base
        leaving = basis[pr]
        entering = headers[pc]
        basis[pr] = entering

        historial[-1]["pivote"] = {"fila":pr, "col":pc, "entra":entering, "sale":leaving}
        historial[-1]["operaciones_filas"] = ops
        historial.append({
            "tabla": [[str(c) for c in row] for row in tableau],
            "basis": basis.copy(),
            "headers": headers.copy(),
            "pivote": None,
            "operaciones_filas": []
        })

    # ─── Obtener solución final ───
    sol = {f"x{i+1}": float(tableau[basis.index(f"x{i+1}")][-1]) 
          if f"x{i+1}" in basis else 0.0 for i in range(n)}
    
    z = float(tableau[-1][-1])
    if minimize:
        z = -z

    return sol, z, historial

# -----------------------------------------------------------------------------
# BLOQUE 6: FUNCIÓN find_pivot_frac – Selección de pivote con fracciones
# -----------------------------------------------------------------------------

# Esta función es utilizada específicamente por simplex_estandar, donde todos los valores
# del tableau son del tipo Fraction (racionales exactos).
# Selecciona la fila y columna del pivote según la lógica del método Símplex.
def find_pivot_frac(tableau, minimize, basis, headers):
    """
    Encuentra el pivote para el método simplex (maneja fracciones y problemas no acotados).
    
    Args:
        tableau: Matriz del tableau (incluyendo la fila Z).
        minimize: Booleano (True si es minimización, False para maximización).
        basis: Conjunto de variables básicas actuales (ej: {'s₁', 's₂'}).
        headers: Lista de nombres de columnas (ej: ['x₁', 'x₂', 's₁', 'Solución']).
    
    Returns:
        (row_pivot, col_pivot): Índices del pivote (fila, columna).
    
    Raises:
        Exception: Si el problema es no acotado.
    """
    m = len(tableau) - 1  # Número de restricciones (filas sin contar Z)
    z_row = tableau[-1]   # Fila Z (última fila del tableau)

    # Paso 1: Selección de columna pivote (variable entrante)
    if minimize:
        # Minimización: elige el coeficiente MÁS POSITIVO en Z (para reducir Z)
        col_pivot = next(
            (j for j in range(len(z_row)-1) if z_row[j] > 0 and headers[j] not in basis),
            None
        )
    else:
        # Maximización: elige el coeficiente MÁS NEGATIVO en Z (para aumentar Z)
        col_pivot = next(
            (j for j in range(len(z_row)-1) if z_row[j] < 0 and headers[j] not in basis),
            None
        )

    # Si no hay coeficientes válidos, solución óptima alcanzada
    if col_pivot is None:
        return None, None

    # Paso 2: Selección de fila pivote (razón mínima positiva)
    ratios = []
    for i in range(m):
        a_ij = tableau[i][col_pivot]
        if a_ij > 0:  # Solo considerar valores positivos
            ratio = tableau[i][-1] / a_ij  # b_i / a_ij
            ratios.append((ratio, i))

    if not ratios:
        raise Exception("Problema no acotado: no hay ratios válidos")

    # Elegir la fila con la razón mínima
    row_pivot = min(ratios, key=lambda x: x[0])[1]
    return row_pivot, col_pivot
# -----------------------------------------------------------------------------
# BLOQUE 7: FUNCIÓN simplex_solve – Método de la Gran M completo (Fase I y II)
# -----------------------------------------------------------------------------

# Esta función resuelve un problema de programación lineal usando el método Símplex
# con la técnica de la Gran M. Admite restricciones de tipo <=, >= y =.
# Ejecuta dos fases:
#   - Fase I: Elimina variables artificiales.
#   - Fase II: Optimiza el problema original con las variables válidas.
def simplex_solve(minimize, n, m, obj, cons, types, rhs):
    tableau, basis, headers = build_bigM_tableau(minimize, n, m, obj, cons, types, rhs)
    historial = []
    it = 0

    while it < MAX_ITERS:
        it += 1
        snap = {
            "tabla": [[str(v) for v in row] for row in tableau],
            "basis": basis.copy(),
            "headers": headers.copy(),
            "pivote": None,
            "operaciones_filas": []
        }
        historial.append(snap)

        art = any(b.startswith("a") for b in basis)
        zrow = tableau[-1]
        better = (lambda v: v.is_positive()) if minimize else (lambda v: v.is_negative())
        
        if not any(better(zrow[j]) and headers[j] not in basis for j in range(len(zrow)-1)):
            if art and any(basis[i].startswith("a") and tableau[i][-1].a != 0 for i in range(m)):
                snap.update({"error":"No existe solución factible","infeasible":True})
                return None, None, historial
            break

        pr = pc = None
        if art:
            for i,b in enumerate(basis):
                if b.startswith("a"):
                    for j in range(len(headers)-1):
                        if (not headers[j].startswith("a") and tableau[i][j].is_positive() and headers[j] not in basis):
                            pr, pc = i, j
                            break
                    if pr is not None: break

        if pr is None:
            pr, pc = find_pivot(tableau, basis, headers, minimize)
            if pr is None:
                break

        piv = tableau[pr][pc]
        old = tableau[pr].copy()
        new = [v / piv for v in old]
        labels = headers + ["b"]
        ops = [f"Fila {pr+1} normalizada: " + "; ".join(f"{labels[j]}: {float(old[j]):.4f} ÷ {float(piv):.4f} = {float(new[j]):.4f}" for j in range(len(old)))]
        tableau[pr] = new
        
        for i in range(len(tableau)):
            if i == pr: continue
            fct = tableau[i][pc]
            old_i = tableau[i].copy()
            new_i = [old_i[j] - fct * tableau[pr][j] for j in range(len(old_i))]
            ops.append(f"Fila {i+1} actualizada: " + "; ".join(f"{labels[j]}: {float(old_i[j]):.4f} − {float(fct):.4f}×{float(tableau[pr][j]):.4f} = {float(new_i[j]):.4f}" for j in range(len(old_i))))
            tableau[i] = new_i

        leaving, entering = basis[pr], headers[pc]
        basis[pr] = entering
        snap["pivote"] = {"fila": pr, "col": pc, "entra": entering, "sale": leaving}
        snap["operaciones_filas"] = ops

    if any(b.startswith("a") for b in basis):
        fase1_val = tableau[-1][-1]
        if fase1_val.a != 0 or fase1_val.b != 0:
            historial[-1].update({"error": "No existe solución factible", "infeasible": True})
            return None, None, historial

    arti_cols = [i for i,h in enumerate(headers) if h.startswith("a")]
    for c in sorted(arti_cols, reverse=True):
        for row in tableau:
            del row[c]
        del headers[c]

    Cj = [MixedValue(c,0) for c in obj] + [MixedValue(0,0)]*(len(headers)-n-1) + [MixedValue(0,0)]
    Cb = [MixedValue(obj[int(b[1:])-1],0) if b.startswith("x") else MixedValue(0,0) for b in basis]

    Zj = []
    for j in range(len(headers)):
        s = MixedValue(0,0)
        for i in range(len(basis)):
            s += Cb[i] * tableau[i][j]
        Zj.append(s)
    Zc = [Zj[j] - Cj[j] for j in range(len(headers))]
    tableau[-1] = Zc

    while True:
        pr, pc = find_pivot(tableau, basis, headers, minimize)
        if pr is None:
            break

        piv = tableau[pr][pc]
        old = tableau[pr].copy()
        new = [v / piv for v in old]
        labels = headers.copy()
        ops = [f"(FII) F{pr+1}Norm: " + "; ".join(f"{labels[j]}: {float(old[j]):.4f} ÷ {float(piv):.4f} = {float(new[j]):.4f}" for j in range(len(old)))]
        tableau[pr] = new
        
        for i in range(len(tableau)):
            if i == pr: continue
            fct = tableau[i][pc]
            old_i = tableau[i].copy()
            new_i = [old_i[j] - fct * tableau[pr][j] for j in range(len(old_i))]
            ops.append(f"(FII) F{i+1}Act: " + "; ".join(f"{labels[j]}: {float(old_i[j]):.4f} − {float(fct):.4f}×{float(tableau[pr][j]):.4f} = {float(new_i[j]):.4f}" for j in range(len(old_i))))
            tableau[i] = new_i

        leaving, entering = basis[pr], headers[pc]
        basis[pr] = entering
        historial.append({
            "tabla": [[str(v) for v in row] for row in tableau],
            "basis": basis.copy(),
            "headers": headers.copy(),
            "pivote": {"fila": pr, "col": pc, "entra": entering, "sale": leaving},
            "operaciones_filas": ops
        })

    sol = {f"x{i+1}": 0.0 for i in range(n)}
    for i,b in enumerate(basis):
        if b.startswith("x"):
            sol[b] = float(tableau[i][-1])
    z = float(tableau[-1][-1])
    if minimize:
        z = -z
    return sol, z, historial





# -----------------------------------------------------------------------------
# BLOQUE 8: FUNCIÓN plot_feasible_region – Graficar región factible (solo n=2)
# -----------------------------------------------------------------------------

# Esta función genera una imagen (codificada en base64) que representa la región factible
# del problema de programación lineal cuando hay exactamente 2 variables (x1 y x2).
# También muestra las restricciones como líneas y el punto óptimo encontrado.
def plot_feasible_region(obj, cons, types, rhs, sol):
    if len(obj)!=2: return None # Solo se puede graficar en 2 dimensiones

    # Se generan valores para el eje x1 desde 0 hasta un 50% más que el óptimo    
    x1=np.linspace(0,sol["x1"]*1.5+1,400)
    x2=np.linspace(0,sol["x2"]*1.5+1,400)

    # Se crea una malla 2D de valores de x1 y x2
    X1,X2=np.meshgrid(x1,x2)

    # Se inicializa la región factible como verdadera (todo es permitido al principio)
    feas=np.ones_like(X1,bool)

    # Se aplica cada restricción a la malla para limitar la región factible
    for (c,t,b) in zip(cons,types,rhs):
        A,B=c; # Coeficientes de la restricción A*x1 + B*x2

        # Se aplica la condición correspondiente sobre la malla
        if t=="<=": 
            feas&=(A*X1+B*X2<=b+1e-6) # margen de tolerancia numérica
        elif t==">=": 
            feas&=(A*X1+B*X2>=b-1e-6)
        else: # igualdad
            feas&=(abs(A*X1+B*X2-b)<=1e-6)

    # Se crea una figura y se rellena la región factible
    plt.figure()
    plt.contourf(X1,X2,feas,levels=[-1,0,1],alpha=0.5)

    # Se dibujan todas las restricciones como líneas
    for (c,t,b) in zip(cons,types,rhs):
        A,B=c
        if B!=0: 
            # Se despeja x2 y se grafica en función de x1
            plt.plot(x1,(b-A*x1)/B,label=f"{A}x1+{B}x2 {t} {b}")
        else: 
            # Caso especial: restricción vertical (solo x1 involucrada)
            plt.axvline(x=b/A,label=f"{A}x1 {t} {b}")
    
    # Se marca el punto óptimo encontrado
    plt.plot(sol["x1"],sol["x2"],"ro",label="Óptimo")

    # Se ajusta el tamaño de la gráfica
    plt.xlim(0,max(x1)); plt.ylim(0,max(x2))
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()

    # Se guarda la figura como imagen PNG en memoria (sin crear archivo físico)
    buf=io.BytesIO(); plt.savefig(buf,format="png"); plt.close()
    buf.seek(0)

    # Se codifica la imagen como texto en base64 para incrustar en HTML o JSON
    return base64.b64encode(buf.read()).decode("utf-8")
