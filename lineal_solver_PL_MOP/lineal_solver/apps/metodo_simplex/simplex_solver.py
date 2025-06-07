# simplex_solver.py

def simplex_solver(c, A, b):
    """
    Resuelve un problema de programación lineal mediante el método símplex.

    Parámetros:
    c (lista): Coeficientes de la función objetivo.
    A (lista de listas): Coeficientes de las restricciones.
    b (lista): Valores del lado derecho de las restricciones.

    Devuelve:
    dict: Un diccionario que contiene la solución óptima y el valor óptimo.
    """
    from scipy.optimize import linprog

    # Resolver el problema de programación lineal
    res = linprog(c, A_ub=A, b_ub=b, method='highs')

    if res.success:
        return {
            'valor_optimo': res.fun,
            'solucion_optima': res.x
        }
    else:
        return {
            'valor_optimo': None,
            'solucion_optima': None,
            'mensaje': res.message
        }