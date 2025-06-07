# File: /lineal_solver/lineal_solver/apps/grafico/grafico_solver.py

# This file contains the logic for solving linear programming problems graphically.

def plot_graphic_solution(constraints, objective_function):
    """
    Plots the graphical solution of a linear programming problem.

    Parameters:
    constraints (list): A list of constraint equations.
    objective_function (tuple): The coefficients of the objective function.

    Returns:
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a grid of points
    x = np.linspace(0, 10, 200)
    y = np.linspace(0, 10, 200)
    X, Y = np.meshgrid(x, y)

    # Plot constraints
    for constraint in constraints:
        plt.plot(x, eval(constraint), label=f'Constraint: {constraint}')

    # Plot objective function
    plt.plot(x, (objective_function[0] * x + objective_function[1]), label='Objective Function', color='red')

    # Set limits and labels
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Graphical Solution of Linear Programming Problem')
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.show()