import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def divided_differences(x, y):
    n = len(x)
    f = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        f[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            f[i][j] = (f[i + 1][j - 1] - f[i][j - 1]) / (x[i + j] - x[i])
    return f

def newton_interpolation(x_data, y_data, x):
    f = divided_differences(x_data, y_data)
    n = len(x_data) - 1
    P = f[0][0]
    prod = 1.0
    for i in range(1, n + 1):
        prod *= (x - x_data[i - 1])
        P += (prod * f[0][i])
    return P

def lagrange_interpolation(x, y, xi):
    n = len(x)
    total_sum = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term = term * (xi - x[j]) / (x[i] - x[j])
        total_sum += term
    return total_sum

def finite_differences(x, y):
    n = len(x)
    table = np.zeros((n, n + 1))
    table[:, 0] = x
    table[:, 1] = y
    for j in range(2, n + 1):
        for i in range(n - j + 1):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    return table[:, :n]

def mse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def plot_interpolation(x, y, x_used, y_used, x_vals, y_vals, method_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Экспериментальные точки')
    plt.scatter(x_used, y_used, color='red', label=f'Точки для {method_name}')
    plt.plot(x_vals, y_vals, color='green', label=f'Интерполирование {method_name}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Интерполирование по формуле {method_name}')
    plt.show()

# Исходные экспериментальные точки
x = [-1.90, -1.50, -1.10, -0.70, -0.30, 0.10, 0.50, 0.90, 1.30, 1.70, 2.10, 2.50, 2.90, 3.30]
y = [-13.05, -10.36, -8.12, -6.14, -4.49, -3.13, -2.07, -1.18, -0.56, -0.13, 0.05, 0.17, 0.10, -0.17]

# Newton Interpolation
newton_points = [1, 6, 8, 13]
x_used_newton = [x[i - 1] for i in newton_points]
y_used_newton = [y[i - 1] for i in newton_points]
x_vals_newton = np.linspace(min(x), max(x), 400)
y_vals_newton = [newton_interpolation(x_used_newton, y_used_newton, xi) for xi in x_vals_newton]

# Lagrange Interpolation
lagrange_points = [1, 7, 13]
x_used_lagrange = [x[i - 1] for i in lagrange_points]
y_used_lagrange = [y[i - 1] for i in lagrange_points]
x_vals_lagrange = np.linspace(min(x), max(x), 400)
y_vals_lagrange = [lagrange_interpolation(x_used_lagrange, y_used_lagrange, xi) for xi in x_vals_lagrange]

# Finite Differences
differences_table = finite_differences(x, y)
df = pd.DataFrame(differences_table)
print(df)

# Calculate MSE for Newton Interpolation
y_newton_all = [newton_interpolation(x_used_newton, y_used_newton, xi) for xi in x]
mse_all_newton = mse(y, y_newton_all)
x_not_used_newton = [x[i] for i in range(len(x)) if i + 1 not in newton_points]
y_not_used_newton = [y[i] for i in range(len(y)) if i + 1 not in newton_points]
y_newton_not_used = [newton_interpolation(x_used_newton, y_used_newton, xi) for xi in x_not_used_newton]
mse_not_used_newton = mse(y_not_used_newton, y_newton_not_used)
print(f"СКО для всех точек (Newton): {mse_all_newton}")
print(f"СКО для точек, не использованных при интерполировании (Newton): {mse_not_used_newton}")

# Plot Interpolations
plot_interpolation(x, y, x_used_newton, y_used_newton, x_vals_newton, y_vals_newton, 'Ньютона')
plot_interpolation(x, y, x_used_lagrange, y_used_lagrange, x_vals_lagrange, y_vals_lagrange, 'Лагранжа')
plt.title('Экспериментальные данные')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
