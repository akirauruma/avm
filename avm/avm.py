import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Определение функции оптимизации
def optimization_function(x):
    return -0.38 + 0.99 * x - 3.21 * np.sin(1.15 * x)

# Определение интервала локализации
interval_bounds = (-13.0, 10.0)

# Решение задачи оптимизации с использованием метода локализации экстремума
result = minimize_scalar(optimization_function, bounds=interval_bounds, method='bounded')

# Вывод результата
print(f'Глобальный минимум находится в точке x = {result.x:.3f}, значение функции R(x) = {result.fun:.3f}')

# Построение графика функции
x_values = np.linspace(interval_bounds[0], interval_bounds[1], 1000)
y_values = optimization_function(x_values)

plt.plot(x_values, y_values, label='R(x)')
plt.scatter(result.x, result.fun, color='red', label='Глобальный минимум')
plt.title('График оптимизируемой функции и глобального минимума')
plt.xlabel('x')
plt.ylabel('R(x)')
plt.legend()
plt.show()
