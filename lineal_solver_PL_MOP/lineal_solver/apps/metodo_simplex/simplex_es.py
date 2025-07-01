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
    # Cuenta cuántas restricciones son de cada tipo:
    # - s_count: cantidad de ≤ (requieren variable de holgura s)
    # - e_count: cantidad de ≥ (requieren variable de exceso e)
    # - a_count: cantidad de ≥ o = (requieren variable artificial a)
    s_count = sum(1 for t in types if t == "<=")
    e_count = sum(1 for t in types if t == ">=")
    a_count = sum(1 for t in types if t in (">=", "="))

    # total columnas = variables reales + s + e + a + columna b (RHS)
    total = n + s_count + e_count + a_count + 1

    # Posiciones donde empiezan cada bloque de variables
    pos_s = n
    pos_e = n + s_count
    pos_a = n + s_count + e_count
    pos_b = total - 1  # última columna

    # Construye la cabecera del tableau (etiquetas de cada columna)
    headers = (
        [f"x{i+1}" for i in range(n)] # Variables reales
        + [f"s{i+1}" for i in range(s_count)] # Variables de holgura
        + [f"e{i+1}" for i in range(e_count)] # Variables de exceso
        + [f"a{i+1}" for i in range(a_count)] # Variables artificiales
        + ["b"] # Columna del lado derecho (RHS)
    )

    tableau = [] # matriz de coeficientes fila por fila
    basis = [] # variables básicas iniciales

    # Índices para numerar s, e, a (ya que no se agregan siempre)
    si = ei = ai = 0

    for i, t in enumerate(types): # Por cada restricción...
        # ─── Normalizar si el RHS es negativo ───
        # Si el lado derecho de la desigualdad es negativo, se invierte:
        # - todos los coeficientes
        # - el signo de la desigualdad
        if rhs[i] < 0:
            # invertimos coeficientes y RHS
            cons[i] = [-c for c in cons[i]]
            rhs[i] = -rhs[i]
            # invertimos sentido de la desigualdad
            t = {"<=": ">=", ">=": "<=", "=": "="}[t]
            types[i] = t # se actualiza el tipo en la lista original

        # Se crea una fila completa con ceros (a + bM)
        row = [MixedValue(0, 0) for _ in range(total)]

        # Se insertan los coeficientes reales (x1, x2, ...) en la fila
        for j in range(n):
            row[j] = MixedValue(cons[i][j], 0)
        # Se pone el valor del lado derecho (RHS) en la última columna
        row[pos_b] = MixedValue(rhs[i], 0)

        # añadimos holgura/exceso/artificial según tipo
        if t == "<=":
            # Variable de holgura (s): se suma +1 para convertir en igualdad
            row[pos_s + si] = MixedValue(1, 0)
            basis.append(f"s{si+1}") # se agrega a la base
            si += 1

        elif t == ">=":
            # Variable de exceso (e): se resta 1
            row[pos_e + ei] = MixedValue(-1, 0)
            # Variable artificial (a): se suma 1
            row[pos_a + ai] = MixedValue(1, 0)
            basis.append(f"a{ai+1}") # se agrega a la base
            ei += 1
            ai += 1

        else:  # "="
            # Solo se agrega variable artificial
            row[pos_a + ai] = MixedValue(1, 0)
            basis.append(f"a{ai+1}")
            ai += 1
        # Se agrega la fila construida al tableau
        tableau.append(row)

    # ─── Construcción de la fila Z – Cj para Fase I ───

    # Signo depende de si es minimización o maximización
    sign = 1 if minimize else -1

    # Vector de coeficientes de la función objetivo extendida a todas las columnas
    Cj = (
        [MixedValue(c, 0) for c in obj] # coeficientes reales
        + [MixedValue(0, 0)] * (s_count + e_count)
        + [MixedValue(0, sign)] * a_count  # penaliza artificiales con M
        + [MixedValue(0, 0)] # RHS

    )

    # Coeficientes básicos (Cb): 0 para s, e; M para a
    Cb = [
        MixedValue(0, sign) if b.startswith("a") else MixedValue(0, 0)
        for b in basis
    ]

    # Cálculo de Zj: sumatoria de Cb[i] * columna_j en cada fila i
    Zj = []
    for j in range(total):
        s = MixedValue(0, 0)
        for i in range(m):
            s += Cb[i] * tableau[i][j]
        Zj.append(s)

    # Se calcula Z – Cj para cada columna
    Zc = [Zj[j] - Cj[j] for j in range(total)]

    # Se agrega la fila Z – Cj al final del tableau
    tableau.append(Zc)

    return tableau, basis, headers # Retorna el tableau armado, las variables básicas y los encabezados


# -----------------------------------------------------------------------------
# BLOQUE 4: FUNCIÓN find_pivot – Selección del elemento pivote (Gran M)
# -----------------------------------------------------------------------------

# Esta función se utiliza durante las iteraciones del método Símplex (fase I o II)
# para seleccionar la posición del elemento pivote. El pivote es el valor central
# alrededor del cual se harán transformaciones para acercarse a la solución óptima.
def find_pivot(tableau, basis, headers, minimize):
    m = len(tableau)-1 # Cantidad de filas de restricciones (sin contar fila Z – Cj)
    Zc = tableau[-1] # Fila Z – Cj (última del tableau), indica dirección de mejora

    # Función que define el criterio para mejorar la función objetivo:
    # - Si estamos minimizando: mejoramos si Z – Cj es positivo (aún se puede reducir Z)
    # - Si estamos maximizando: mejoramos si Z – Cj es negativo (aún se puede aumentar Z)
    better = (lambda v:v.is_positive()) if minimize else (lambda v:v.is_negative())

    # Se seleccionan las columnas candidatas para entrar a la base:
    # - Deben tener coeficiente Z – Cj que mejora el objetivo
    # - No deben estar en la base actual (evita retrocesos o ciclos)
    cand = [
        j for j,v in enumerate(Zc[:-1]) if better(v) and headers[j] not in basis
    ]

    # Si no hay ninguna columna que cumpla con mejorar el objetivo, se alcanzó el óptimo
    if not cand:
        return None,None
    # Se elige la columna con menor índice entre las candidatas (regla arbitraria, puede cambiarse)
    pc = min(cand)

    # Ahora buscamos la fila que limita el avance en la dirección de la columna elegida
    # Se usa el método de la razón mínima: b_i / a_ij
    ratios=[]
    for i in range(m): # Se recorren las filas de restricciones
        a = tableau[i][pc] # Coeficiente en la columna pivote
        if a.is_positive(): # Solo consideramos filas con coeficiente positivo (para mantener factibilidad)
            ratios.append((tableau[i][-1]/a,i))  # Guardamos la tupla (valor, índice_fila)

    # Si no hay ninguna razón válida (todas ≤ 0), entonces la solución no está acotada (es infinita)
    if not ratios: raise Exception("No acotado")

    # Se selecciona la fila que tenga la razón mínima: limita el avance y mantiene factibilidad
    pr = min(ratios)[1]
    return pr,pc # Devuelve posición de la fila y columna del pivote

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

    # El método solo admite restricciones del tipo ≤ (menor o igual)
    if any(t!="<=" for t in types):
        return None,None,[{"error":"Estándar sólo ≤"}]

    # No se permiten lados derechos negativos (en este método)
    if any(b<0 for b in rhs):
        return None,None,[{"error":"RHS<0 no soportado"}]

    # ─── Construcción del tableau ───

    # Cantidad total de columnas: n variables reales + m variables de holgura
    total = n+m

    # Encabezados de columnas: x1, x2, ..., s1, s2, ..., b
    headers = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["b"]

    tableau=[] # matriz del tableau

    for i in range(m): # Para cada restricción
        row=[Fraction(0)]*(total+1) # inicializa fila con ceros

        for j in range(n): 
            row[j]=Fraction(cons[i][j]).limit_denominator() # coeficientes reales
        row[n+i]=Fraction(1) # variable de holgura correspondiente
        row[-1] =Fraction(rhs[i]).limit_denominator() # lado derecho
        tableau.append(row)

    # Fila Z – Cj inicial (fila objetivo)
    # - Si se minimiza, se cambia el signo de los coeficientes para unificar lógica
    zrow=[Fraction(obj[j]).limit_denominator()*(1 if minimize else -1) for j in range(n)]
    zrow += [Fraction(0)]*(m+1)
    tableau.append(zrow)

    # ─── Variables básicas iniciales: las variables de holgura ───
    basis=[f"s{i+1}" for i in range(m)]

    # ─── Historial para registrar cada iteración ───
    historial=[{
        "tabla":   [[str(c) for c in row] for row in tableau], # valores como string
        "basis":   basis.copy(),
        "headers": headers.copy(),
        "pivote":  None,
        "operaciones_filas":[]
    }]

    extra=False # bandera para saber si se está en una iteración extra por empate

    # ─── Ciclo principal de iteraciones del método Símplex ───
    while True:
        # Buscar posición del pivote
        pr,pc=find_pivot_frac(tableau,minimize,basis,headers)

        if pr is None: # no hay más mejoras posibles
            if extra: break
            # Buscar columna con Zc = 0 que aún permita mejorar (soluciones múltiples)
            Zc=tableau[-1][:-1]
            pc=next((j for j,v in enumerate(Zc)
                     if v==0 and any(tableau[i][j]>0 for i in range(m))),None)
            if pc is None: break

            # Seleccionar fila con razón mínima como pivote
            pr=min((tableau[i][-1]/tableau[i][pc],i) for i in range(m) if tableau[i][pc]>0)[1]
            extra=True

        # ─── Pivoteo ───

        ops=[] # operaciones realizadas en esta iteración
        # Normalización de la fila pivote
        piv=tableau[pr][pc]
        old=tableau[pr].copy()
        new=[v/piv for v in old]
        labels=headers+["b"]

        divs=[f"{labels[j]}: {float(old[j]):.4f}÷{float(piv):.4f}={float(new[j]):.4f}"
              for j in range(len(old))]
        ops.append(f"Fila{pr+1}Norm:\n "+"\n".join(divs))
        tableau[pr]=new

        ops.append(f"Fila{pr+1}Norm: "+"; ".join(divs))
        tableau[pr]=new # se reemplaza la fila pivote

        # Reducción gaussiana: eliminar variable en otras filas
        for i in range(len(tableau)):
            if i==pr: continue
            fct=tableau[i][pc]; old=tableau[i].copy()
            nxt=[old[j]-fct*tableau[pr][j] for j in range(len(old))]
            subs=[f"{labels[j]}: {float(old[j]):.4f}−{float(fct):.4f}×{float(tableau[pr][j]):.4f}"
                  f"={float(nxt[j]):.4f}" for j in range(len(old))]
            ops.append(f"Fila{i+1}Act:\n "+"\n".join(subs))
            tableau[i]=nxt

        # Actualizar base: entra una nueva variable, sale la anterior
        leaving=basis[pr]
        entering=headers[pc]
        basis[pr]=entering

        # Guardar estado en el historial
        historial[-1]["pivote"]={"fila":pr,"col":pc,"entra":entering,"sale":leaving}
        historial[-1]["operaciones_filas"]=ops

        # Se guarda una nueva copia del tableau para la siguiente iteración
        historial.append({
            "tabla":   [[str(c) for c in row] for row in tableau],
            "basis":   basis.copy(),
            "headers": headers.copy(),
            "pivote":  None,
            "operaciones_filas":[]
        })

    # ─── Obtener solución final ───
    # Se extrae el valor de cada variable real (x1, x2, ...) si está en la base
    sol={f"x{i+1}":float(tableau[basis.index(f"x{i+1}")][-1])
         if f"x{i+1}" in basis else 0.0 for i in range(n)}

    # El valor óptimo está en la última celda de la fila Z – Cj
    z=float(tableau[-1][-1])
    if minimize: 
        z=-z # se corrige el signo para devolver el valor real

    # solución óptima, valor de Z, historial de pasos
    return sol,z,historial

# -----------------------------------------------------------------------------
# BLOQUE 6: FUNCIÓN find_pivot_frac – Selección de pivote con fracciones
# -----------------------------------------------------------------------------

# Esta función es utilizada específicamente por simplex_estandar, donde todos los valores
# del tableau son del tipo Fraction (racionales exactos).
# Selecciona la fila y columna del pivote según la lógica del método Símplex.
def find_pivot_frac(tableau, minimize, basis, headers):
    m=len(tableau)-1 # Cantidad de restricciones (filas sin contar Z – Cj)
    Zc=tableau[-1] # Última fila del tableau: Z – Cj

    # Función de comparación según tipo de optimización:
    # - Si se minimiza: buscamos entradas positivas
    # - Si se maximiza: buscamos entradas negativas
    better=(lambda v:v>0) if minimize else (lambda v:v<0)

    # Se selecciona la primera columna que cumple la condición de mejora
    pc=next((j for j,v in enumerate(Zc[:-1]) if better(v)),None)

    # Si no se encuentra ninguna columna candidata, se alcanzó el óptimo
    if pc is None: return None,None

    # Para la columna seleccionada, se buscan las razones b_i / a_ij para cada fila válida
    # Solo se consideran filas con a_ij > 0 para evitar violar restricciones
    ratios=[(tableau[i][-1]/tableau[i][pc],i) for i in range(m) if tableau[i][pc]>0]

    # Si ninguna fila es válida (todas ≤ 0), el problema es no acotado
    if not ratios: raise Exception("no acotada")

    # Se elige la fila con la menor razón b_i / a_ij
    return min(ratios)[1],pc # Devuelve (fila pivote, columna pivote)

# -----------------------------------------------------------------------------
# BLOQUE 7: FUNCIÓN simplex_solve – Método de la Gran M completo (Fase I y II)
# -----------------------------------------------------------------------------

# Esta función resuelve un problema de programación lineal usando el método Símplex
# con la técnica de la Gran M. Admite restricciones de tipo <=, >= y =.
# Ejecuta dos fases:
#   - Fase I: Elimina variables artificiales.
#   - Fase II: Optimiza el problema original con las variables válidas.
def simplex_solve(
    minimize: bool, # True para minimizar, False para maximizar
    n: int, # Cantidad de variables reales (x1, x2, ...)
    m: int, # Cantidad de restricciones
    obj: Sequence[float], # Coeficientes de la función objetivo
    cons: Sequence[Sequence[float]], # Coeficientes de las restricciones (matriz A)
    types: Sequence[str], # Tipos de restricciones: "<=", ">=", "="
    rhs: Sequence[float], # Lado derecho de las restricciones (vector b)
) -> Tuple[Dict[str, float], float, List[Dict]]:
     # ─── FASE I: construir el tableau extendido con variables artificiales ───
    tableau, basis, headers = build_bigM_tableau(minimize, n, m, obj, cons, types, rhs)
    historial: List[Dict] = [] # Para guardar todas las iteraciones
    it = 0 # contador de iteraciones

    # --- Fase I: eliminar artificiales --------------------------
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

        # Verifica si aún hay variables artificiales en la base
        art = any(b.startswith("a") for b in basis)

        # Verifica si ya se alcanzó un óptimo en Fase I
        zrow = tableau[-1]
        better = (lambda v: v.is_positive()) if minimize else (lambda v: v.is_negative())
        if not any(better(zrow[j]) and headers[j] not in basis
                   for j in range(len(zrow)-1)):
            # si hay artificial con b≠0 → infactible
            if art and any(basis[i].startswith("a") and tableau[i][-1].a != 0
                           for i in range(m)):
                snap.update({"error":"No existe solución factible","infeasible":True})
                return None, None, historial
            break # fin de la Fase I

        # ─── Selección del pivote ───
        pr = pc = None

        # Si hay artificiales en la base, priorizamos eliminarlas
        if art:
            for i,b in enumerate(basis):
                if b.startswith("a"):
                    for j in range(len(headers)-1):
                        if (not headers[j].startswith("a")
                            and tableau[i][j].is_positive()
                            and headers[j] not in basis):
                            pr, pc = i, j
                            break
                    if pr is not None: break

        # Si no se encontró pivote preferente, usar método clásico
        if pr is None:
            pr, pc = find_pivot(tableau, basis, headers, minimize)
            if pr is None:
                break # óptimo alcanzado

        # ─── Pivoteo: normalización y eliminación ───
        piv = tableau[pr][pc]
        old = tableau[pr].copy()
        new = [v / piv for v in old]
        labels = headers + ["b"]
        ops = [f"Fila {pr+1} normalizada: " +
               "; ".join(f"{labels[j]}: {float(old[j]):.4f} ÷ {float(piv):.4f} = {float(new[j]):.4f}"
                         for j in range(len(old)))]
        tableau[pr] = new
        for i in range(len(tableau)):
            if i == pr: continue
            fct   = tableau[i][pc]
            old_i = tableau[i].copy()
            new_i = [old_i[j] - fct * tableau[pr][j] for j in range(len(old_i))]
            ops.append(f"Fila {i+1} actualizada: " +
                       "; ".join(f"{labels[j]}: {float(old_i[j]):.4f} − {float(fct):.4f}×{float(tableau[pr][j]):.4f} = {float(new_i[j]):.4f}"
                                 for j in range(len(old_i))))
            tableau[i] = new_i

        # Actualización de la base
        leaving, entering = basis[pr], headers[pc]
        basis[pr] = entering
        snap["pivote"] = {"fila": pr, "col": pc, "entra": entering, "sale": leaving}
        snap["operaciones_filas"] = ops

    # --- Comprobación de infactibilidad tras Fase I (solo si quedan a) ---
    if any(b.startswith("a") for b in basis):
        fase1_val = tableau[-1][-1]
        if fase1_val.a != 0 or fase1_val.b != 0:
            historial[-1].update({
                "error":      "No existe solución factible",
                "infeasible": True
            })
            return None, None, historial

    # ─── FASE II: eliminar columnas artificiales y reconstruir Z – Cj ───

    # Se eliminan columnas correspondientes a variables artificiales
    arti_cols = [i for i,h in enumerate(headers) if h.startswith("a")]
    for c in sorted(arti_cols, reverse=True):
        for row in tableau:
            del row[c]
        del headers[c]

    # reconstruir fila Z–Cj con coeficientes originales
    Cj = [MixedValue(c,0) for c in obj] \
         + [MixedValue(0,0)]*(len(headers)-n-1) \
         + [MixedValue(0,0)]
    Cb = [MixedValue(obj[int(b[1:])-1],0) if b.startswith("x") else MixedValue(0,0)
          for b in basis]

    # Se recalcula Zj y Z – Cj para comenzar la Fase II
    Zj = []
    for j in range(len(headers)):
        s = MixedValue(0,0)
        for i in range(len(basis)):
            s += Cb[i] * tableau[i][j]
        Zj.append(s)
    Zc = [Zj[j] - Cj[j] for j in range(len(headers))]
    tableau[-1] = Zc

    # ─── Iteraciones Fase II para encontrar solución óptima ───
    while True:
        pr, pc = find_pivot(tableau, basis, headers, minimize)
        if pr is None:
            break

        # pivoteo Fase II
        piv = tableau[pr][pc]
        old = tableau[pr].copy()
        new = [v / piv for v in old]
        labels = headers.copy()
        ops = [f"(FII) F{pr+1}Norm: " +
               "; ".join(f"{labels[j]}: {float(old[j]):.4f} ÷ {float(piv):.4f} = {float(new[j]):.4f}"
                         for j in range(len(old)))]
        tableau[pr] = new
        for i in range(len(tableau)):
            if i == pr: continue
            fct   = tableau[i][pc]
            old_i = tableau[i].copy()
            new_i = [old_i[j] - fct * tableau[pr][j] for j in range(len(old_i))]
            ops.append(f"(FII) F{i+1}Act: " +
                       "; ".join(f"{labels[j]}: {float(old_i[j]):.4f} − {float(fct):.4f}×{float(tableau[pr][j]):.4f} = {float(new_i[j]):.4f}"
                                 for j in range(len(old_i))))
            tableau[i] = new_i

        # Actualización de la base
        leaving, entering = basis[pr], headers[pc]
        basis[pr] = entering
        historial.append({
            "tabla":   [[str(v) for v in row] for row in tableau],
            "basis":   basis.copy(),
            "headers": headers.copy(),
            "pivote":  {"fila": pr, "col": pc, "entra": entering, "sale": leaving},
            "operaciones_filas": ops
        })

    # ───  Extraer solución final ───
    sol = {f"x{i+1}": 0.0 for i in range(n)} # inicializa todas las x en 0
    for i,b in enumerate(basis):
        if b.startswith("x"):
            sol[b] = float(tableau[i][-1])
    z = float(tableau[-1][-1]) # valor óptimo de la función objetivo
    if minimize:
        z = -z # se corrige el signo para reflejar correctamente Z
    return sol, z, historial # retorna solución, valor óptimo y pasos





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
