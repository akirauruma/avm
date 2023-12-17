import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Заданные значения
delta_x = 0.5
x_values = np.arange(-2, 2 + delta_x, delta_x)


# Определение функции для дифференциального уравнения
def differential_equation(x, y):
    return 1 - x - y


# Решение дифференциального уравнения с использованием solve_ivp
solution = solve_ivp(differential_equation, [-2, 2], [0], t_eval=x_values)


# Построение графика
plt.plot(solution.t, solution.y[0], label="Численное решение", marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Решение дифференциального уравнения методом Эйлера с улучшением')
plt.legend()
plt.show()
