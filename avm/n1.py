import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Исходные экспериментальные точки
x = [-1.90, -1.50, -1.10, -0.70, -0.30, 0.10, 0.50, 0.90, 1.30, 1.70, 2.10, 2.50, 2.90, 3.30]
y = [-13.05, -10.36, -8.12, -6.14, -4.49, -3.13, -2.07, -1.18, -0.56, -0.13, 0.05, 0.17, 0.10, -0.17]


def divided_differences(x, y):
    n = len(x)
    f = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        f[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            f[i][j] = (f[i+1][j-1] - f[i][j-1]) / (x[i+j] - x[i])
    return f


def newton_interpolation(x_data, y_data, x):
    f = divided_differences(x_data, y_data)
    n = len(x_data) - 1
    P = f[0][0]
    prod = 1.0
    for i in range(1, n+1):
        prod *= (x - x_data[i-1])
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


lagrange_points = [1, 7, 13]

x_used = [x[i-1] for i in lagrange_points]
y_used = [y[i-1] for i in lagrange_points]

x_vals = np.linspace(min(x), max(x), 400)
y_vals = [lagrange_interpolation(x_used, y_used, xi) for xi in x_vals]


def finite_differences(x, y):
    # Количество точек
    n = len(x)

    # Создание таблицы с x и y
    table = np.zeros((n, n + 1))
    table[:, 0] = x
    table[:, 1] = y

    # Заполнение таблицы конечными разностями
    for j in range(2, n + 1):
        for i in range(n - j + 1):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    return table[:, :n]


# Вычисление конечных разностей
differences_table = finite_differences(x, y)

# Вывод таблицы с помощью pandas
df = pd.DataFrame(differences_table)
print(df)


def mse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


# Точки, использованные для интерполирования по формуле Ньютона
newton_points = [1, 6, 8, 13]

x_used = [x[i-1] for i in newton_points]
y_used = [y[i-1] for i in newton_points]

# Вычисляем интерполированные значения на всем диапазоне
x_vals = np.linspace(min(x), max(x), 400)
y_vals = [newton_interpolation(x_used, y_used, xi) for xi in x_vals]

y_newton_all = [newton_interpolation(x_used, y_used, xi) for xi in x]


# Вычисляем СКО для всех точек
mse_all = mse(y, y_newton_all)

# Убираем точки, использованные для интерполирования по Ньютона
x_not_used = [x[i] for i in range(len(x)) if i+1 not in newton_points]
y_not_used = [y[i] for i in range(len(y)) if i+1 not in newton_points]

# Вычисляем значения интерполирующего полинома Ньютона для точек, не использованных при интерполировании
y_newton_not_used = [newton_interpolation(x_used, y_used, xi) for xi in x_not_used]

# Вычисляем СКО для точек, не использованных при интерполировании
mse_not_used = mse(y_not_used, y_newton_not_used)

print(f"СКО для всех точек: {mse_all}")
print(f"СКО для точек, не использованных при интерполировании: {mse_not_used}")

# Построение графика Нутика
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.scatter(x_used, y_used, color='red', label='Точки для Ньютона')
plt.plot(x_vals, y_vals, color='green', label='Интерполирование Ньютона')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Интерполирование по формуле Ньютона')
plt.show()


# Построение графика Лагрика
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', marker='o', label='Экспериментальные точки')
plt.title('Экспериментальные данные')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.scatter(x_used, y_used, color='red', label='Точки для Лагранжа')
plt.plot(x_vals, y_vals, color='green', label='Интерполирование Лагранжа')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Интерполирование по формуле Лагранжа')
plt.show()
