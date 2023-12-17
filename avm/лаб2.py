import numpy as np
import matplotlib.pyplot as plt

def df(x):
    a1 = 1.38
    a2 = -0.52 * 2  # Удваиваем коэффициент, так как производная от x^2 это 2x
    a3 = -7.43
    return a1 + a2*x + a3*np.cos(x)


def f(x):
    a0 = 0.41
    a1 = 1.38
    a2 = -0.52
    a3 = -7.43
    return a0 + a1*x + a2*x**2 + a3*np.sin(x)

x_vals = np.linspace(-10, 10, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title("Graph of f(x)")
plt.legend()
plt.show()


import pandas as pd


# Метод половинного деления:
def bisection_method(f, a, b, tol):
    steps = 0
    if f(a) * f(b) > 0:
        return None, steps
    while (b - a) / 2.0 > tol:
        steps += 1
        midpoint = (a + b) / 2.0
        if f(midpoint) == 0:
            return midpoint, steps
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2.0, steps


# Метод касательных:
def newton_method(f, df, x0, tol):
    steps = 0
    while abs(f(x0)) > tol:
        steps += 1
        x0 = x0 - f(x0) / df(x0)
    return x0, steps


tolerances = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# Для корректного определения интервала локализации, используйте графический метод.
a, b = -10, 10  # Это просто примерные значения. Уточните их на основе вашего графика.
initial_guess = 0  # Начальное приближение для метода Ньютона

results_bisection = {'Tolerance': [], 'Root': [], 'Steps': []}
results_newton = {'Tolerance': [], 'Root': [], 'Steps': []}

for tol in tolerances:
    root_bisection, step_bisection = bisection_method(f, a, b, tol)
    root_newton, step_newton = newton_method(f, df, initial_guess, tol)

    results_bisection['Tolerance'].append(tol)
    results_bisection['Root'].append(root_bisection)
    results_bisection['Steps'].append(step_bisection)

    results_newton['Tolerance'].append(tol)
    results_newton['Root'].append(root_newton)
    results_newton['Steps'].append(step_newton)

# Представляем результаты в виде таблицы
df_bisection = pd.DataFrame(results_bisection)
df_newton = pd.DataFrame(results_newton)

print("Results using Bisection Method:")
print(df_bisection)

print("\nResults using Newton's Method:")
print(df_newton)

# График зависимости количества итераций от точности :
plt.figure(figsize=(10, 6))
plt.plot(tolerances, results_bisection['Steps'], 'o-', label='Bisection method ')
plt.plot(tolerances, results_newton['Steps'], 's-', label="Newton's method ")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tolerance')
plt.ylabel('Number of Steps')
plt.title('Steps vs Tolerance for ')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

