#!/usr/bin/env python3
"""Método Símplex – versión Estándar y Gran M con historial detallado.

Exporta:
    * simplex_estandar   – para problemas con todas las restricciones "≤".
    * simplex_solve      – para problemas con restricciones mixtas (≤, ≥, =).
    * plot_feasible_region – gráfica 2‑D de la región factible (sólo n = 2).

Cada función devuelve un historial listo para renderizar en una plantilla
Django. Las celdas del tableau se convierten en cadenas para facilitar la
serialización.
"""
from __future__ import annotations

from fractions import Fraction
import io
import base64
from typing import List, Tuple, Sequence, Dict

import numpy as np
import matplotlib.pyplot as plt

BIG_M_NUMERIC = 1_000_000  # 1 e6 para convertir MixedValue → float

# -----------------------------------------------------------------------------
# MixedValue:  a + b·M  (M ≈ 1e6)
# -----------------------------------------------------------------------------
class MixedValue:
    """Representa números de la forma  a + b·M  usados en el método de la Gran M."""

    def __init__(self, a: int | Fraction = 0, b: int | Fraction = 0):
        self.a = Fraction(a).limit_denominator()
        self.b = Fraction(b).limit_denominator()

    # ---------------- operaciones básicas ----------------
    def __add__(self, other: "MixedValue") -> "MixedValue":
        return MixedValue(self.a + other.a, self.b + other.b)

    def __sub__(self, other: "MixedValue") -> "MixedValue":
        return MixedValue(self.a - other.a, self.b - other.b)

    def __mul__(self, other):
        if isinstance(other, MixedValue):  # ignoramos término M²
            return MixedValue(self.a * other.a, self.a * other.b + self.b * other.a)
        return MixedValue(self.a * Fraction(other), self.b * Fraction(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, MixedValue):
            if other.b != 0:
                raise ZeroDivisionError("División por expresión con M no soportada")
            return MixedValue(self.a / other.a, self.b / other.a)
        return MixedValue(self.a / Fraction(other), self.b / Fraction(other))

    # ---------------- utilidades ----------------
    def is_positive(self) -> bool:
        if self.b > 0:
            return True
        if self.b < 0:
            return False
        return self.a > 0

    def is_negative(self) -> bool:
        if self.b < 0:
            return True
        if self.b > 0:
            return False
        return self.a < 0

    # ---------------- comparaciones (para min/max de ratios) ----------------
    def __lt__(self, other: "MixedValue") -> bool:
        if self.b != other.b:
            return self.b < other.b
        return self.a < other.a

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MixedValue):
            return NotImplemented
        return self.a == other.a and self.b == other.b

    # ---------------- casts y print ----------------
    def __float__(self):
        return float(self.a + self.b * BIG_M_NUMERIC)

    def __str__(self):
        parts: List[str] = []
        if self.b != 0:
            if self.b == 1:
                parts.append("M")
            elif self.b == -1:
                parts.append("-M")
            else:
                parts.append(f"{self.b}M")
        if self.a != 0:
            sign = "" if not parts else ("+" if self.a > 0 else "")
            parts.append(f"{sign}{self.a}")
        return "".join(parts) if parts else "0"

    # Para que Fraction(self) funcione (debug opcional)
    def as_fraction(self) -> Fraction:
        return self.a + self.b * BIG_M_NUMERIC


# -----------------------------------------------------------------------------
# 1.  Tableau inicial (Gran M)
# -----------------------------------------------------------------------------

def build_bigM_tableau(
    minimize: bool,
    n: int,
    m: int,
    obj: Sequence[float],
    cons: Sequence[Sequence[float]],
    types: Sequence[str],
    rhs: Sequence[float],
) -> Tuple[List[List[MixedValue]], List[str], List[str]]:
    """Construye el tableau extendido para Gran M y devuelve (tableau, basis, headers)."""

    # Contadores de variables auxiliares
    s_count = sum(1 for t in types if t == "<=")      # holgura
    e_count = sum(1 for t in types if t == ">=")      # exceso
    a_count = sum(1 for t in types if t in (">=", "="))  # artificial

    total_cols = n + s_count + e_count + a_count + 1  # + 1 RHS

    # Posiciones de los bloques
    pos_s = n
    pos_e = pos_s + s_count
    pos_a = pos_e + e_count
    pos_b = total_cols - 1

    headers = (
        [f"x{i+1}" for i in range(n)]
        + [f"s{i+1}" for i in range(s_count)]
        + [f"e{i+1}" for i in range(e_count)]
        + [f"a{i+1}" for i in range(a_count)]
        + ["b"]
    )

    tableau: List[List[MixedValue]] = []
    basis: List[str] = []
    si = ei = ai = 0

    for i in range(m):
        row = [MixedValue(0, 0) for _ in range(total_cols)]
        # coeficientes origen
        for j in range(n):
            row[j] = MixedValue(cons[i][j], 0)
        # RHS
        row[pos_b] = MixedValue(rhs[i], 0)
        # tipo de restricción
        if types[i] == "<=":
            row[pos_s + si] = MixedValue(1, 0)
            basis.append(f"s{si+1}")
            si += 1
        elif types[i] == ">=":
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

    # ---------------- Fila Z–Cj ----------------
    sign = 1 if minimize else -1  # +M para min, -M para max

    Cj = (
        [MixedValue(c, 0) for c in obj]
        + [MixedValue(0, 0) for _ in range(s_count + e_count)]
        + [MixedValue(0, sign) for _ in range(a_count)]
        + [MixedValue(0, 0)]
    )

    Cb = [MixedValue(0, sign) if b.startswith("a") else MixedValue(0, 0) for b in basis]

    Zj = []
    for j in range(total_cols):
        s = MixedValue(0, 0)
        for i in range(m):
            s += Cb[i] * tableau[i][j]
        Zj.append(s)

    Zc = [Zj[j] - Cj[j] for j in range(total_cols)]
    tableau.append(Zc)

    return tableau, basis, headers


# -----------------------------------------------------------------------------
# 2.  Impresión opcional para consola (debug)
# -----------------------------------------------------------------------------

def print_tableau(tableau: List[List[MixedValue]], basis: List[str], headers: List[str]):
    m = len(tableau) - 1
    print("     " + "  ".join(f"{h:>8}" for h in headers))
    for i in range(m):
        print(f"{basis[i]:>4}", end=" ")
        print("  ".join(f"{str(v):>8}" for v in tableau[i]))
    print("Z–Cj", end=" ")
    print("  ".join(f"{str(v):>8}" for v in tableau[-1]))
    print()


# -----------------------------------------------------------------------------
# 3.  Pivote: selección y operación
# -----------------------------------------------------------------------------

def find_pivot(
    tableau: List[List[MixedValue]],
    basis: List[str],
    headers: List[str],
    minimize: bool
) -> Tuple[int | None, int | None]:
    """
    Selecciona columna y fila pivote.
    Devuelve (pr, pc) o (None, None) si ya no hay mejora real.
    · Ignora columnas cuya variable ya está en la base.
    · Aplica Bland (elige el índice más pequeño) para evitar ciclos.
    """
    m = len(tableau) - 1          # número de filas de restricciones
    Zc = tableau[-1]              # última fila: Z – Cj

    # 1️⃣ columnas candidatas con coeficiente que mejora y NO están en la base
    better = (lambda v: v.is_positive()) if minimize else (lambda v: v.is_negative())
    improving = [
        j for j, v in enumerate(Zc[:-1])
        if better(v) and headers[j] not in basis
    ]
    if not improving:
        return None, None         # óptimo (o fase I limpia)

    # 2️⃣ elige columna con Bland
    pc = min(improving)

    # 3️⃣ prueba de razón  (b / a_·pc)  ≥ 0  y coeficiente positivo
    ratios: List[Tuple[MixedValue, int]] = []
    for i in range(m):
        coeff = tableau[i][pc]
        if coeff.is_positive():
            ratios.append((tableau[i][-1] / coeff, i))

    if not ratios:
        raise Exception("Problema no acotado (todas las razones negativas).")

    # 4️⃣ fila pivote — Bland sobre las mínimas razones
    pr = min(ratios)[1]
    return pr, pc





def find_pivot_frac(tableau: List[List[Fraction]],
                    minimize: bool) -> Tuple[int | None, int | None]:
    m = len(tableau) - 1
    Zc = tableau[-1]
    better = (lambda v: v > 0) if minimize else (lambda v: v < 0)
    pc = next((j for j, v in enumerate(Zc[:-1]) if better(v)), None)
    if pc is None:
        return None, None
    ratios = [(tableau[i][-1] / tableau[i][pc], i)
              for i in range(m) if tableau[i][pc] > 0]
    if not ratios:
        raise Exception("no acotada")
    pr = min(ratios)[1]
    return pr, pc



def pivot(tableau: List[List[MixedValue]], basis: List[str], headers: List[str], pr: int, pc: int) -> Tuple[str, str]:
    # normalizar fila pivote
    pe = tableau[pr][pc]
    tableau[pr] = [v / pe for v in tableau[pr]]

    # eliminar en resto de filas
    for i in range(len(tableau)):
        if i == pr:
            continue
        factor = tableau[i][pc]
        tableau[i] = [tableau[i][j] - factor * tableau[pr][j] for j in range(len(tableau[0]))]

    leaving = basis[pr]
    entering = headers[pc]
    basis[pr] = entering
    return leaving, entering


# -----------------------------------------------------------------------------
# 4.  Método Símplex Estándar (todas ≤)
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------
# Método Símplex Estándar  (≤)  – con paso extra si Z-Cj = 0
# ------------------------------------------------------------------
def simplex_estandar(
    minimize: bool,
    n: int,
    m: int,
    obj: Sequence[float],
    cons: Sequence[Sequence[float]],
    types: Sequence[str],
    rhs: Sequence[float],
):
    # --- validaciones ------------------------------------------------
    if any(t != "<=" for t in types):
        return None, None, [{"error": "El método estándar sólo admite restricciones ≤."}]
    if any(b < 0 for b in rhs):
        return None, None, [{"error": "RHS negativo no soportado por el estándar."}]

    # --- tableau inicial --------------------------------------------
    total_vars = n + m
    headers = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["b"]

    tableau: List[List[Fraction]] = []
    for i in range(m):
        row = [Fraction(0) for _ in range(total_vars + 1)]
        for j in range(n):
            row[j] = Fraction(cons[i][j]).limit_denominator()
        row[n + i] = Fraction(1)                              # holgura
        row[-1] = Fraction(rhs[i]).limit_denominator()
        tableau.append(row)

    z_row = [Fraction(obj[j]).limit_denominator() * (1 if minimize else -1)
             for j in range(n)]
    z_row += [Fraction(0) for _ in range(m + 1)]
    tableau.append(z_row)

    basis = [f"s{i+1}" for i in range(m)]
    historial = [{
        "tabla":   [[str(c) for c in row] for row in tableau],
        "basis":   basis.copy(),
        "headers": headers.copy(),
        "pivote":  None,
    }]

    # --- iteraciones -------------------------------------------------
    extra_done = False   # ← sólo permitimos una pivoteada extra

    while True:
        pr, pc = find_pivot_frac(tableau, minimize)

        if pr is None:                          # óptimo (no mejora real)
            if extra_done:                      # ya hicimos la extra → salir
                break

            # intentar una pivoteada por Z–Cj = 0
            Zc = tableau[-1][:-1]
            pc = next(
                (j for j, v in enumerate(Zc)
                 if v == 0 and any(tableau[i][j] > 0 for i in range(m))),
                None,
            )
            if pc is None:
                break                           # no hay alternativa
            ratios = [(tableau[i][-1] / tableau[i][pc], i)
                      for i in range(m) if tableau[i][pc] > 0]
            pr = min(ratios)[1]
            extra_done = True                   # marcamos que ya hicimos la extra

        # ----- guardar info de pivote antes de modificar -------------
        historial[-1]["pivote"] = {
            "fila": pr, "col": pc,
            "entra": headers[pc],
            "sale": basis[pr],
        }

        # ----- operación de pivote -----------------------------------
        piv = tableau[pr][pc]
        tableau[pr] = [v / piv for v in tableau[pr]]
        for i, row in enumerate(tableau):
            if i == pr:
                continue
            factor = row[pc]
            tableau[i] = [row[j] - factor * tableau[pr][j]
                          for j in range(total_vars + 1)]
        basis[pr] = headers[pc]

        # ----- snapshot ----------------------------------------------
        historial.append({
            "tabla":   [[str(c) for c in row] for row in tableau],
            "basis":   basis.copy(),
            "headers": headers.copy(),
            "pivote":  None,
        })

    # --- solución final ----------------------------------------------
    sol = {
        f"x{i+1}": float(tableau[basis.index(f"x{i+1}")][-1])
        if f"x{i+1}" in basis else 0.0
        for i in range(n)
    }
    zval = float(tableau[-1][-1])
    if minimize:
        zval = -zval
    return sol, zval, historial

# ----- sustituye tu función simplex_solve por esta -----
MAX_ITERS = 1000   #  ya sea aquí o arriba en el archivo (solo una vez)

# Asegúrate de definir MAX_ITERS solo una vez en el archivo
MAX_ITERS = 1000

def simplex_solve(
    minimize: bool,
    n: int,
    m: int,
    obj: Sequence[float],
    cons: Sequence[Sequence[float]],
    types: Sequence[str],
    rhs: Sequence[float],
):
    # Construir tableau inicial con Gran M
    tableau, basis, headers = build_bigM_tableau(minimize, n, m, obj, cons, types, rhs)
    historial = []
    iter_count = 0
    
    # Fase I: Eliminar variables artificiales
    while iter_count <= MAX_ITERS:
        iter_count += 1
        
        # Snapshot actual
        current_tableau = [[str(v) for v in row] for row in tableau]
        snap = {
            "tabla": current_tableau,
            "basis": basis.copy(),
            "headers": headers.copy(),
            "pivote": None,
        }
        historial.append(snap)

        # Verificar variables artificiales en base
        artificial_in_base = any(b.startswith('a') for b in basis)
        
        # Verificar optimalidad
        optimal = True
        z_row = tableau[-1]
        for j in range(len(headers)-1):
            if headers[j] in basis:
                continue
            if minimize and z_row[j].is_positive():
                optimal = False
                break
            if not minimize and z_row[j].is_negative():
                optimal = False
                break

        # Condiciones de parada
        if optimal:
            if artificial_in_base:
                # Verificar si variables artificiales son cero
                for i, b in enumerate(basis):
                    if b.startswith('a') and not tableau[i][-1].a == 0:
                        historial.append({
                            "error": "No existe solución factible",
                            "explicacion_inf": "Variables artificiales no pueden eliminarse",
                            "infeasible": True
                        })
                        return None, None, historial
            break

        # Selección de pivote - priorizar eliminar artificiales
        pr, pc = None, None
        if artificial_in_base:
            for i, b in enumerate(basis):
                if b.startswith('a'):
                    for j in range(len(headers)-1):
                        if (not headers[j].startswith('a') and 
                            tableau[i][j].is_positive() and 
                            headers[j] not in basis):
                            pr, pc = i, j
                            break
                    if pr is not None:
                        break
        
        # Si no hay artificiales para eliminar, pivote normal
        if pr is None:
            pr, pc = find_pivot(tableau, basis, headers, minimize)
            if pr is None:
                break

        # Operación de pivote
        pivot_val = tableau[pr][pc]
        tableau[pr] = [v / pivot_val for v in tableau[pr]]
        
        for i in range(len(tableau)):
            if i != pr:
                factor = tableau[i][pc]
                tableau[i] = [tableau[i][j] - factor * tableau[pr][j] 
                            for j in range(len(tableau[0]))]
        
        basis[pr] = headers[pc]
        snap["pivote"] = {"fila": pr, "col": pc, "entra": headers[pc], "sale": basis[pr]}

    # Verificar factibilidad final
    if any(b.startswith('a') for b in basis):
        historial.append({
            "error": "No existe solución factible",
            "explicacion_inf": "Variables artificiales permanecen en la base",
            "infeasible": True
        })
        return None, None, historial

    # Extraer solución
    sol = {f"x{i+1}": 0.0 for i in range(n)}
    for i, b in enumerate(basis):
        if b.startswith("x"):
            idx = int(b[1:])-1
            if idx < n:
                sol[b] = float(tableau[i][-1])

    Z = float(tableau[-1][-1])
    if minimize:
        Z = -Z

    return sol, Z, historial
# ----- fin de la función simplex_solve -----



# -----------------------------------------------------------------------------
# 6.  Gráfica de la región factible (solo n = 2)
# -----------------------------------------------------------------------------

def plot_feasible_region(
    obj: Sequence[float],
    cons: Sequence[Sequence[float]],
    types: Sequence[str],
    rhs: Sequence[float],
    sol: Dict[str, float],
) -> str | None:
    if len(obj) != 2:
        return None

    x1 = np.linspace(0, sol["x1"] * 1.5 + 1, 400)
    x2 = np.linspace(0, sol["x2"] * 1.5 + 1, 400)
    X1, X2 = np.meshgrid(x1, x2)
    feas = np.ones_like(X1, dtype=bool)

    for (c, ty, b) in zip(cons, types, rhs):
        A, B = c
        if ty == "<=":
            feas &= (A * X1 + B * X2 <= b + 1e-6)
        elif ty == ">=":
            feas &= (A * X1 + B * X2 >= b - 1e-6)
        else:
            feas &= (np.abs(A * X1 + B * X2 - b) <= 1e-6)

    plt.figure()
    plt.contourf(X1, X2, feas, levels=[-1, 0, 1], colors=["#ccccff"], alpha=0.5)

    for (c, ty, b) in zip(cons, types, rhs):
        A, B = c
        if B != 0:
            plt.plot(x1, (b - A * x1) / B, label=f"{A}x1+{B}x2 {ty} {b}")
        else:
            plt.axvline(x=b / A, label=f"{A}x1 {ty} {b}")

    plt.plot(sol["x1"], sol["x2"], "ro", label="Óptimo")
    plt.xlim(0, max(x1))
    plt.ylim(0, max(x2))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Región factible y punto óptimo")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
