import numpy as np
from scipy.optimize import linprog
from itertools import combinations

def parse_restricciones(restricciones):
    A = []
    b = []
    signs = []
    for r in restricciones:
        A.append(r['lhs'])
        b.append(r['rhs'])
        signs.append(r['sign'])
    return np.array(A), np.array(b), signs

def encontrar_intersecciones(A, b, signs):
    intersecciones = []

    for pair in combinations(range(len(A)), 2):
        A_sub = A[list(pair), :]
        b_sub = b[list(pair)]

        try:
            punto = np.linalg.solve(A_sub, b_sub)
            if np.all(np.isfinite(punto)):
                intersecciones.append(punto.tolist())
        except np.linalg.LinAlgError:
            continue

    return intersecciones

def es_factible(punto, A, b, signs):
    for i in range(len(A)):
        lhs = np.dot(A[i], punto)
        if signs[i] == "<=" and lhs > b[i] + 1e-6:
            return False
        if signs[i] == ">=" and lhs < b[i] - 1e-6:
            return False
    return True

def resolver_problema(restricciones, funcion_objetivo):
    """
    restricciones: [{"lhs":[..], "sign":"<=", "rhs":..}, ...]
    funcion_objetivo: [c1, c2]
    """

    A, b, signs = parse_restricciones(restricciones)

    # Encontrar intersecciones de restricciones
    candidatas = encontrar_intersecciones(A, b, signs)

    # Filtrar factibles
    factibles = [p for p in candidatas if es_factible(p, A, b, signs)]

    if not factibles:
        raise ValueError("No se encontró región factible.")

    # Evaluar la función objetivo
    z_values = [funcion_objetivo[0]*p[0] + funcion_objetivo[1]*p[1] for p in factibles]

    idx_optimo = np.argmax(z_values)
    optimo = {
        "x": factibles[idx_optimo][0],
        "y": factibles[idx_optimo][1],
        "z": z_values[idx_optimo]
    }

    # Resultado de vértices
    vertices = [{"x": p[0], "y": p[1]} for p in factibles]

    return vertices, optimo
