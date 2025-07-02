from __future__ import annotations
from fractions import Fraction
import io
import base64
from typing import List, Tuple, Sequence, Dict, Union
import numpy as np
import matplotlib.pyplot as plt

# Constantes
BIG_M_NUMERIC = Fraction(10**6)  # Valor grande para M como fracción
MAX_ITERS = 1000

class MixedValue:
    def __init__(self, a: int | Fraction = 0, b: int | Fraction = 0):
        self.a = Fraction(a).limit_denominator()
        self.b = Fraction(b).limit_denominator()
    
    def __add__(self, o):
        return MixedValue(self.a + o.a, self.b + o.b)
    
    def __sub__(self, o):
        return MixedValue(self.a - o.a, self.b - o.b)
    
    def __mul__(self, o):
        if isinstance(o, MixedValue):
            return MixedValue(self.a * o.a, self.a * o.b + self.b * o.a)
        return MixedValue(self.a * o, self.b * o)
    
    __rmul__ = __mul__
    
    def __truediv__(self, o):
        if isinstance(o, MixedValue):
            if o.b != 0: raise ZeroDivisionError
            return MixedValue(self.a / o.a, self.b / o.a)
        return MixedValue(self.a / o, self.b / o)
    
    def is_positive(self):
        return (self.b > 0) or (self.b == 0 and self.a > 0)
    
    def is_negative(self):
        return (self.b < 0) or (self.b == 0 and self.a < 0)
    
    def __lt__(self, o):
        if self.b != o.b: return self.b < o.b
        return self.a < o.a
    
    def __eq__(self, o):
        return isinstance(o, MixedValue) and self.a == o.a and self.b == o.b
    
    def __float__(self):
        return float(self.a + self.b * BIG_M_NUMERIC)
    
    def __str__(self):
        parts = []
        if self.b != 0:
            parts.append(f"{self.b}M" if abs(self.b) != 1 else ("M" if self.b > 0 else "-M"))
        if self.a != 0:
            sign = "" if not parts else ("+" if self.a > 0 else "")
            parts.append(f"{sign}{self.a}")
        return "".join(parts) or "0"

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
        [f"x{i+1}" for i in range(n)] +
        [f"s{i+1}" for i in range(s_count)] +
        [f"e{i+1}" for i in range(e_count)] +
        [f"a{i+1}" for i in range(a_count)] +
        ["b"]
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
            row[j] = MixedValue(Fraction(cons[i][j]), 0)
        row[pos_b] = MixedValue(Fraction(rhs[i]), 0)

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

    M = BIG_M_NUMERIC
    Cj = [MixedValue(Fraction(c), 0) for c in obj]
    Cj += [MixedValue(0, 0)] * (s_count + e_count)
    Cj += [MixedValue(0, M if minimize else -M)] * a_count
    Cj += [MixedValue(0, 0)]

    Cb = []
    for b in basis:
        if b.startswith("a"):
            Cb.append(MixedValue(0, M if minimize else -M))
        else:
            Cb.append(MixedValue(0, 0))

    Zj = [MixedValue(0, 0) for _ in range(total)]
    for j in range(total):
        for i in range(m):
            Zj[j] += Cb[i] * tableau[i][j]

    Zc = [Zj[j] - Cj[j] for j in range(total)]
    tableau.append(Zc)

    return tableau, basis, headers

def simplex_solve(
    minimize: bool,
    n: int,
    m: int,
    obj: Sequence[float],
    cons: Sequence[Sequence[float]],
    types: Sequence[str],
    rhs: Sequence[float]
) -> Tuple[Dict[str, float], float, List[dict]]:
    """
    Gran M con historial paso a paso, idéntica firma que simplex_estandar.
    Devuelve (sol, z, historial), donde historial es lista de dicts:
      {
        "tabla": List[List[str]],
        "basis": List[str],
        "headers": List[str],
        "pivote": {"fila": int, "col": int, "entra": str, "sale": str} | None,
        "operaciones_filas": List[str]
      }
    """
    # Copias de entrada para no mutar originales
    cons = [row[:] for row in cons]
    types = list(types)
    rhs   = list(rhs)

    # 1) Construir tableau inicial
    tableau, basis, headers = build_bigM_tableau(minimize, n, m, obj, cons, types, rhs)
    historial: List[dict] = []
    it = 0

    def snapshot(piv=None, ops=None):
        return {
            "tabla": [[str(v) for v in row] for row in tableau],
            "basis": basis.copy(),
            "headers": headers.copy(),
            "pivote": piv,
            "operaciones_filas": ops or []
        }

    # registro inicial (antes de Fase I)
    historial.append(snapshot())

    # ─── Fase I: eliminar artificiales ───
    while any(b.startswith("a") for b in basis) and it < MAX_ITERS:
        it += 1
        art_row = art_col = None
        for i, b in enumerate(basis):
            if b.startswith("a"):
                art_row = i
                for j in range(len(headers)-1):
                    if not headers[j].startswith("a") and tableau[i][j].a != 0:
                        art_col = j
                        break
                if art_col is not None:
                    break
        if art_row is None or art_col is None:
            break

        # pivoteo
        piv = tableau[art_row][art_col]
        old_row = tableau[art_row].copy()
        tableau[art_row] = [v / piv for v in old_row]
        ops = [
            f"Fila{art_row+1}Norm: " +
            ", ".join(f"{old_row[j]}÷{piv}={tableau[art_row][j]}" for j in range(len(old_row)))
        ]
        for i in range(len(tableau)):
            if i == art_row: continue
            fct = tableau[i][art_col]
            old_i = tableau[i].copy()
            tableau[i] = [old_i[j] - fct * tableau[art_row][j] for j in range(len(old_i))]
            ops.append(
                f"Fila{i+1}Act: " +
                ", ".join(f"{old_i[j]}−{fct}×{tableau[art_row][j]}={tableau[i][j]}"
                          for j in range(len(old_i)))
            )

        leaving = basis[art_row]
        entering = headers[art_col]
        basis[art_row] = entering
        pivote_info = {"fila": art_row, "col": art_col, "entra": entering, "sale": leaving}
        historial.append(snapshot(piv=pivote_info, ops=ops))

    # si quedan artificiales => infactible
    if any(b.startswith("a") for b in basis):
        return None, None, [{"error": "No existe solución factible", "infeasible": True}]

    # 2) Eliminar columnas artificiales
    arti_cols = [i for i, h in enumerate(headers) if h.startswith("a")]
    for c in sorted(arti_cols, reverse=True):
        for row in tableau:
            del row[c]
        del headers[c]
    # snapshot tras Fase I
    historial.append(snapshot())

    # 3) Fase II: optimizar con objetivo original
    Cj = [MixedValue(Fraction(c), 0) for c in obj] \
         + [MixedValue(0, 0)] * (len(headers) - n) \
         + [MixedValue(0, 0)]
    while it < MAX_ITERS:
        it += 1
        # recalcular Zj–Cj
        Zj = [MixedValue(0, 0) for _ in headers]
        for j in range(len(headers)):
            for i, b in enumerate(basis):
                if b.startswith("x"):
                    idx = int(b[1:]) - 1
                    Zj[j] += MixedValue(Fraction(obj[idx]), 0) * tableau[i][j]
        Zc = [Zj[j] - Cj[j] for j in range(len(headers))]
        tableau[-1] = Zc

        # escoger pivote
        pr = pc = None
        if minimize:
            maxv = MixedValue(0, 0)
            for j in range(len(Zc)-1):
                if headers[j] not in basis and Zc[j] > maxv:
                    maxv, pc = Zc[j], j
        else:
            minv = MixedValue(0, 0)
            for j in range(len(Zc)-1):
                if headers[j] not in basis and Zc[j] < minv:
                    minv, pc = Zc[j], j
        if pc is None:
            break

        # ratios
        ratios = [
            (float(tableau[i][-1] / tableau[i][pc]), i)
            for i in range(len(basis))
            if tableau[i][pc].a > 0
        ]
        if not ratios:
            raise Exception("El problema es no acotado")
        pr = min(ratios, key=lambda x: (x[0], x[1]))[1]

        # pivoteo Fase II
        piv = tableau[pr][pc]
        old_row = tableau[pr].copy()
        tableau[pr] = [v / piv for v in old_row]
        ops = [
            f"Fila{pr+1}Norm: " +
            ", ".join(f"{old_row[j]}÷{piv}={tableau[pr][j]}" for j in range(len(old_row)))
        ]
        for i in range(len(tableau)):
            if i == pr: continue
            fct = tableau[i][pc]
            old_i = tableau[i].copy()
            tableau[i] = [old_i[j] - fct * tableau[pr][j] for j in range(len(old_i))]
            ops.append(
                f"Fila{i+1}Act: " +
                ", ".join(f"{old_i[j]}−{fct}×{tableau[pr][j]}={tableau[i][j]}"
                          for j in range(len(old_i)))
            )

        leaving = basis[pr]
        entering = headers[pc]
        basis[pr] = entering
        pivote_info = {"fila": pr, "col": pc, "entra": entering, "sale": leaving}
        historial.append(snapshot(piv=pivote_info, ops=ops))

    # 4) Extraer solución final
    sol = {f"x{i+1}": 0.0 for i in range(n)}
    for i, b in enumerate(basis):
        if b.startswith("x"):
            sol[b] = float(tableau[i][-1])
    z = float(tableau[-1][-1])
    if minimize:
        z = abs(z)

    return sol, z, historial



def find_pivot(tableau, basis, headers, minimize):
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

def simplex_estandar(
    minimize: bool,
    n: int,
    m: int,
    obj: Sequence[float],
    cons: Sequence[Sequence[float]],
    types: Sequence[str],
    rhs: Sequence[float],
) -> Tuple[Dict[str, float], float, List[Dict]]:

    # 1) Validaciones
    if any(t != "<=" for t in types):
        return None, None, [{"error": "Estándar sólo ≤"}]
    if any(b < 0 for b in rhs):
        return None, None, [{"error": "RHS < 0 no soportado"}]

    # 2) Encabezados y base
    headers = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["b"]
    basis   = [f"s{i+1}" for i in range(m)]

    # 3) Construcción del tableau
    tableau: List[List[Fraction]] = []
    for i in range(m):
        row = [Fraction(cons[i][j]).limit_denominator() for j in range(n)]
        row += [Fraction(1) if i == j else Fraction(0) for j in range(m)]
        row.append(Fraction(rhs[i]).limit_denominator())
        tableau.append(row)

    # 4) Fila Z–Cj inicial (siempre –Cj)
    zrow = [-Fraction(obj[j]).limit_denominator() for j in range(n)]
    zrow += [Fraction(0)] * (m + 1)
    tableau.append(zrow)

    # 5) Histórico
    historial: List[Dict] = [{
        "tabla":           [[str(c) for c in row] for row in tableau],
        "headers":         headers.copy(),
        "basis":           basis.copy(),
        "pivote":          None,
        "operaciones_filas": []
    }]

    # 6) Funciones de pivoteo (solo x1…xn)
    def find_pivot_min() -> Tuple[int,int]:
        z = tableau[-1][:-1]
        col = best = None
        for j in range(n):
            v = z[j]
            if v < 0 and headers[j] not in basis and (best is None or v > best):
                best, col = v, j
        if col is None:
            return None, None
        ratios = [(tableau[i][-1] / tableau[i][col], i)
                  for i in range(m) if tableau[i][col] > 0]
        if not ratios:
            raise Exception("Problema no acotado")
        return min(ratios, key=lambda x: x[0])[1], col

    def find_pivot_max() -> Tuple[int,int]:
        z = tableau[-1][:-1]
        col = best = None
        for j in range(n):
            v = z[j]
            if v < 0 and headers[j] not in basis and (best is None or v < best):
                best, col = v, j
        if col is None:
            return None, None
        ratios = [(tableau[i][-1] / tableau[i][col], i)
                  for i in range(m) if tableau[i][col] > 0]
        if not ratios:
            raise Exception("Problema no acotado")
        return min(ratios, key=lambda x: x[0])[1], col

    # 7) Bucle principal
    while True:
        pr, pc = (find_pivot_min() if minimize else find_pivot_max())
        if pr is None:
            break

        ops = []
        piv = tableau[pr][pc]
        old = tableau[pr].copy()
        norm = [v / piv for v in old]
        ops.append(
            f"Fila{pr+1}Norm:\n" +
            "\n".join(f"{headers[j]}: {old[j]}/{piv} = {norm[j]}"
                      for j in range(len(old)))
        )
        tableau[pr] = norm

        for i in range(len(tableau)):
            if i == pr: continue
            f = tableau[i][pc]
            old_i = tableau[i].copy()
            new_i = [old_i[j] - f*tableau[pr][j] for j in range(len(old_i))]
            ops.append(
                f"Fila{i+1}Act:\n" +
                "\n".join(f"{headers[j]}: {old_i[j]} - {f}×{tableau[pr][j]} = {new_i[j]}"
                          for j in range(len(old_i)))
            )
            tableau[i] = new_i

        leaving  = basis[pr]
        entering = headers[pc]
        basis[pr] = entering
        historial[-1]["pivote"] = {"fila": pr, "col": pc,
                                   "entra": entering, "sale": leaving}
        historial[-1]["operaciones_filas"] = ops
        historial.append({
            "tabla":           [[str(c) for c in row] for row in tableau],
            "headers":         headers.copy(),
            "basis":           basis.copy(),
            "pivote":          None,
            "operaciones_filas": []
        })

    # 8) Solución
    sol = {
        f"x{i+1}": float(tableau[basis.index(f"x{i+1}")][-1])
        if f"x{i+1}" in basis else 0.0
        for i in range(n)
    }
    z_val = float(tableau[-1][-1])
    # CORRECCIÓN FINAL: Z siempre positivo
    if minimize:
        Z = z_val
    else:
        Z = -z_val * -1

    return sol, Z, historial



















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
