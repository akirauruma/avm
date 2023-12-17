import numpy as np
import matplotlib.pyplot as plt

# Заданные параметры
a = 4.6
b = -1.5
x0 = 4.0  # Начальное значение x
y0 = 0.0  # Начальное значение y
h = 0.2   # Шаг


# Функция правой части дифференциального уравнения
def f(x, y):
    return 4.6 * y - 1.5 * x


# Метод Эйлера-Коши
def euler_coshi(x0, y0, h, a, b):
    x_values = [x0]
    y_values = [y0]

    while x0 < b:
        y0 += h * f(x0, y0)
        x0 += h

        x_values.append(x0)
        y_values.append(y0)

    return np.array(x_values), np.array(y_values)


# Вычисление решения
x_values, y_values = euler_coshi(x0, y0, h, a, b)

# Вывод результатов
print("x values:", x_values)
print("y values:", y_values)

# График решения
plt.plot(x_values, y_values, label='Numerical solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution of the Differential Equation')
plt.legend()
plt.show()
