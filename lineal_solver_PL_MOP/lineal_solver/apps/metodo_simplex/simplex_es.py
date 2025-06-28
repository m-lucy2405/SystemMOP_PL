#!/usr/bin/env python3
from __future__ import annotations
from fractions import Fraction
import io, base64
from typing import List, Tuple, Sequence, Dict

import numpy as np
import matplotlib.pyplot as plt

BIG_M_NUMERIC = 1_000_000
MAX_ITERS    = 1000

# -----------------------------------------------------------------------------
# MixedValue: soporte para coeficientes con M
# -----------------------------------------------------------------------------
class MixedValue:
    def __init__(self, a: int | Fraction = 0, b: int | Fraction = 0):
        self.a = Fraction(a).limit_denominator()
        self.b = Fraction(b).limit_denominator()
    def __add__(self, o): return MixedValue(self.a+o.a, self.b+o.b)
    def __sub__(self, o): return MixedValue(self.a-o.a, self.b-o.b)
    def __mul__(self, o):
        if isinstance(o, MixedValue):
            return MixedValue(self.a*o.a, self.a*o.b + self.b*o.a)
        return MixedValue(self.a*o, self.b*o)
    __rmul__ = __mul__
    def __truediv__(self, o):
        if isinstance(o, MixedValue):
            if o.b != 0: raise ZeroDivisionError
            return MixedValue(self.a/o.a, self.b/o.a)
        return MixedValue(self.a/o, self.b/o)
    def is_positive(self):
        return (self.b>0) or (self.b==0 and self.a>0)
    def is_negative(self):
        return (self.b<0) or (self.b==0 and self.a<0)
    def __lt__(self, o):
        if self.b!=o.b: return self.b<o.b
        return self.a<o.a
    def __eq__(self, o):
        return isinstance(o, MixedValue) and self.a==o.a and self.b==o.b
    def __float__(self): return float(self.a + self.b*BIG_M_NUMERIC)
    def __str__(self):
        parts=[]
        if self.b!=0:
            parts.append(f"{self.b}M" if abs(self.b)!=1 else ("M" if self.b>0 else "-M"))
        if self.a!=0:
            sign = "" if not parts else ("+" if self.a>0 else "")
            parts.append(f"{sign}{self.a}")
        return "".join(parts) or "0"

# -----------------------------------------------------------------------------
# Construcción del tableau Gran M
# -----------------------------------------------------------------------------
def build_bigM_tableau(minimize, n, m, obj, cons, types, rhs):
    s_count = sum(1 for t in types if t == "<=")
    e_count = sum(1 for t in types if t == ">=")
    a_count = sum(1 for t in types if t in (">=", "="))
    total = n + s_count + e_count + a_count + 1
    pos_s, pos_e, pos_a = n, n + s_count, n + s_count + e_count
    pos_b = total - 1

    headers = (
        [f"x{i+1}" for i in range(n)]
        + [f"s{i+1}" for i in range(s_count)]
        + [f"e{i+1}" for i in range(e_count)]
        + [f"a{i+1}" for i in range(a_count)]
        + ["b"]
    )

    tableau, basis = [], []
    si = ei = ai = 0
    for i, t in enumerate(types):
        # ─── Normalizar si rhs negativo ───
        if rhs[i] < 0:
            # invertimos coeficientes y RHS
            cons[i] = [-c for c in cons[i]]
            rhs[i] = -rhs[i]
            # invertimos sentido de la desigualdad
            t = {"<=": ">=", ">=": "<=", "=": "="}[t]
            types[i] = t

        # construimos la fila
        row = [MixedValue(0, 0) for _ in range(total)]
        for j in range(n):
            row[j] = MixedValue(cons[i][j], 0)
        row[pos_b] = MixedValue(rhs[i], 0)

        # añadimos holgura/exceso/artificial según tipo
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

    # construimos la fila Z–Cj para Fase I
    sign = 1 if minimize else -1
    Cj = (
        [MixedValue(c, 0) for c in obj]
        + [MixedValue(0, 0)] * (s_count + e_count)
        + [MixedValue(0, sign)] * a_count
        + [MixedValue(0, 0)]
    )
    Cb = [
        MixedValue(0, sign) if b.startswith("a") else MixedValue(0, 0)
        for b in basis
    ]

    Zj = []
    for j in range(total):
        s = MixedValue(0, 0)
        for i in range(m):
            s += Cb[i] * tableau[i][j]
        Zj.append(s)
    Zc = [Zj[j] - Cj[j] for j in range(total)]
    tableau.append(Zc)

    return tableau, basis, headers


# -----------------------------------------------------------------------------
# Selección de pivote Gran M
# -----------------------------------------------------------------------------
def find_pivot(tableau, basis, headers, minimize):
    m = len(tableau)-1
    Zc = tableau[-1]
    better = (lambda v:v.is_positive()) if minimize else (lambda v:v.is_negative())
    cand = [j for j,v in enumerate(Zc[:-1]) if better(v) and headers[j] not in basis]
    if not cand: return None,None
    pc = min(cand)
    ratios=[]
    for i in range(m):
        a = tableau[i][pc]
        if a.is_positive():
            ratios.append((tableau[i][-1]/a,i))
    if not ratios: raise Exception("No acotado")
    pr = min(ratios)[1]
    return pr,pc

# -----------------------------------------------------------------------------
# Método Estándar (≤)
# -----------------------------------------------------------------------------
def simplex_estandar(
    minimize: bool, n: int, m: int,
    obj: Sequence[float],
    cons: Sequence[Sequence[float]],
    types: Sequence[str],
    rhs: Sequence[float],
) -> Tuple[dict,float,List[dict]]:
    from fractions import Fraction
    # validaciones
    if any(t!="<=" for t in types):
        return None,None,[{"error":"Estándar sólo ≤"}]
    if any(b<0 for b in rhs):
        return None,None,[{"error":"RHS<0 no soportado"}]

    # construir tableau
    total = n+m
    headers = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["b"]
    tableau=[]
    for i in range(m):
        row=[Fraction(0)]*(total+1)
        for j in range(n): row[j]=Fraction(cons[i][j]).limit_denominator()
        row[n+i]=Fraction(1)
        row[-1] =Fraction(rhs[i]).limit_denominator()
        tableau.append(row)
    zrow=[Fraction(obj[j]).limit_denominator()*(1 if minimize else -1) for j in range(n)]
    zrow += [Fraction(0)]*(m+1)
    tableau.append(zrow)

    # inicial
    basis=[f"s{i+1}" for i in range(m)]
    historial=[{
        "tabla":   [[str(c) for c in row] for row in tableau],
        "basis":   basis.copy(),
        "headers": headers.copy(),
        "pivote":  None,
        "operaciones_filas":[]
    }]

    extra=False
    while True:
        # pivot
        pr,pc=find_pivot_frac(tableau,minimize,basis,headers)
        if pr is None:
            if extra: break
            # extra
            Zc=tableau[-1][:-1]
            pc=next((j for j,v in enumerate(Zc)
                     if v==0 and any(tableau[i][j]>0 for i in range(m))),None)
            if pc is None: break
            pr=min((tableau[i][-1]/tableau[i][pc],i) for i in range(m) if tableau[i][pc]>0)[1]
            extra=True

        ops=[]
        # normalizar elemento a elemento
        piv=tableau[pr][pc]
        old=tableau[pr].copy()
        new=[v/piv for v in old]
        labels=headers+["b"]
        divs=[f"{labels[j]}: {float(old[j]):.4f}÷{float(piv):.4f}={float(new[j]):.4f}"
              for j in range(len(old))]
        ops.append(f"Fila{pr+1}Norm: "+"; ".join(divs))
        tableau[pr]=new

        # eliminar
        for i in range(len(tableau)):
            if i==pr: continue
            fct=tableau[i][pc]; old=tableau[i].copy()
            nxt=[old[j]-fct*tableau[pr][j] for j in range(len(old))]
            subs=[f"{labels[j]}: {float(old[j]):.4f}−{float(fct):.4f}×{float(tableau[pr][j]):.4f}"
                  f"={float(nxt[j]):.4f}" for j in range(len(old))]
            ops.append(f"Fila{i+1}Act: "+"; ".join(subs))
            tableau[i]=nxt

        leaving=basis[pr]; entering=headers[pc]; basis[pr]=entering
        historial[-1]["pivote"]={"fila":pr,"col":pc,"entra":entering,"sale":leaving}
        historial[-1]["operaciones_filas"]=ops
        historial.append({
            "tabla":   [[str(c) for c in row] for row in tableau],
            "basis":   basis.copy(),
            "headers": headers.copy(),
            "pivote":  None,
            "operaciones_filas":[]
        })

    # solución
    sol={f"x{i+1}":float(tableau[basis.index(f"x{i+1}")][-1])
         if f"x{i+1}" in basis else 0.0 for i in range(n)}
    z=float(tableau[-1][-1])
    if minimize: z=-z
    return sol,z,historial

# -----------------------------------------------------------------------------
# Fase I & II Gran M
# -----------------------------------------------------------------------------
def find_pivot_frac(tableau, minimize, basis, headers):
    m=len(tableau)-1; Zc=tableau[-1]; better=(lambda v:v>0) if minimize else (lambda v:v<0)
    pc=next((j for j,v in enumerate(Zc[:-1]) if better(v)),None)
    if pc is None: return None,None
    ratios=[(tableau[i][-1]/tableau[i][pc],i) for i in range(m) if tableau[i][pc]>0]
    if not ratios: raise Exception("no acotada")
    return min(ratios)[1],pc

def simplex_solve(
    minimize: bool,
    n: int,
    m: int,
    obj: Sequence[float],
    cons: Sequence[Sequence[float]],
    types: Sequence[str],
    rhs: Sequence[float],
) -> Tuple[Dict[str, float], float, List[Dict]]:
    # --- Construir tableau Gran M inicial (Fase I) ------------------
    tableau, basis, headers = build_bigM_tableau(minimize, n, m, obj, cons, types, rhs)
    historial: List[Dict] = []
    it = 0

    # --- 1️⃣ Fase I: eliminar artificiales --------------------------
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

        # ¿Quedan artificiales en la base?
        art = any(b.startswith("a") for b in basis)
        # ¿Óptimo en esta Fase I?
        zrow = tableau[-1]
        better = (lambda v: v.is_positive()) if minimize else (lambda v: v.is_negative())
        if not any(better(zrow[j]) and headers[j] not in basis
                   for j in range(len(zrow)-1)):
            # si hay artificial con b≠0 → infactible
            if art and any(basis[i].startswith("a") and tableau[i][-1].a != 0
                           for i in range(m)):
                snap.update({"error":"No existe solución factible","infeasible":True})
                return None, None, historial
            break

        # elegir pivote, priorizando eliminar artificiales
        pr = pc = None
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
        if pr is None:
            pr, pc = find_pivot(tableau, basis, headers, minimize)
            if pr is None:
                break

        # pivoteo Fase I
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

        leaving, entering = basis[pr], headers[pc]
        basis[pr] = entering
        snap["pivote"] = {"fila": pr, "col": pc, "entra": entering, "sale": leaving}
        snap["operaciones_filas"] = ops

    # --- 🚨 Comprobación de infactibilidad tras Fase I (solo si quedan a) ---
    if any(b.startswith("a") for b in basis):
        fase1_val = tableau[-1][-1]
        if fase1_val.a != 0 or fase1_val.b != 0:
            historial[-1].update({
                "error":      "No existe solución factible",
                "infeasible": True
            })
            return None, None, historial

    # --- 2️⃣ Fase II: reconstruir Z–Cj original y optimizar --------
    # eliminar columnas de artificiales
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

    Zj = []
    for j in range(len(headers)):
        s = MixedValue(0,0)
        for i in range(len(basis)):
            s += Cb[i] * tableau[i][j]
        Zj.append(s)
    Zc = [Zj[j] - Cj[j] for j in range(len(headers))]
    tableau[-1] = Zc

    # iterar Fase II
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

        leaving, entering = basis[pr], headers[pc]
        basis[pr] = entering
        historial.append({
            "tabla":   [[str(v) for v in row] for row in tableau],
            "basis":   basis.copy(),
            "headers": headers.copy(),
            "pivote":  {"fila": pr, "col": pc, "entra": entering, "sale": leaving},
            "operaciones_filas": ops
        })

    # --- 3️⃣ extraer solución final -----------------------------------
    sol = {f"x{i+1}": 0.0 for i in range(n)}
    for i,b in enumerate(basis):
        if b.startswith("x"):
            sol[b] = float(tableau[i][-1])
    z = float(tableau[-1][-1])
    if minimize:
        z = -z
    return sol, z, historial





# -----------------------------------------------------------------------------
# Graficar región (n=2)
# -----------------------------------------------------------------------------
def plot_feasible_region(obj, cons, types, rhs, sol):
    if len(obj)!=2: return None
    x1=np.linspace(0,sol["x1"]*1.5+1,400)
    x2=np.linspace(0,sol["x2"]*1.5+1,400)
    X1,X2=np.meshgrid(x1,x2)
    feas=np.ones_like(X1,bool)
    for (c,t,b) in zip(cons,types,rhs):
        A,B=c; 
        if t=="<=": feas&=(A*X1+B*X2<=b+1e-6)
        elif t==">=": feas&=(A*X1+B*X2>=b-1e-6)
        else: feas&=(abs(A*X1+B*X2-b)<=1e-6)
    plt.figure()
    plt.contourf(X1,X2,feas,levels=[-1,0,1],alpha=0.5)
    for (c,t,b) in zip(cons,types,rhs):
        A,B=c
        if B!=0: plt.plot(x1,(b-A*x1)/B,label=f"{A}x1+{B}x2 {t} {b}")
        else: plt.axvline(x=b/A,label=f"{A}x1 {t} {b}")
    plt.plot(sol["x1"],sol["x2"],"ro",label="Óptimo")
    plt.xlim(0,max(x1)); plt.ylim(0,max(x2))
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
    buf=io.BytesIO(); plt.savefig(buf,format="png"); plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
