import numpy as np

def f(x):
    a0 = 0.41
    a1 = 1.38
    a2 = -0.52
    a3 = -7.43
    return a0 + a1*x + a2*x**2 + a3*np.sin(x)

def df(x):
    a1 = 1.38
    a2 = -0.52 * 2
    a3 = -7.43
    return a1 + a2*x + a3*np.cos(x)

import matplotlib.pyplot as plt

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

def newton_method(f, df, x0, tol):
    steps = 0
    while abs(f(x0)) > tol:
        steps += 1
        x0 = x0 - f(x0)/df(x0)
    return x0, steps


tolerances = [0.1, 0.01, 0.001, 0.0001, 0.00001]

roots_bisection = []
steps_bisection = []

roots_newton = []
steps_newton = []

for tol in tolerances:
    root_bisection, step_bisection = bisection_method(f, -10, 10, tol)
    root_newton, step_newton = newton_method(f, df, 0, tol)  # начальное приближение x0 = 0

    roots_bisection.append(root_bisection)
    steps_bisection.append(step_bisection)

    roots_newton.append(root_newton)
    steps_newton.append(step_newton)

plt.figure(figsize=(10, 6))
plt.plot(tolerances, steps_bisection, 'o-', label='Bisection method')
plt.plot(tolerances, steps_newton, 's-', label="Newton's method")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tolerance')
plt.ylabel('Number of Steps')
plt.title('Steps vs Tolerance')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

