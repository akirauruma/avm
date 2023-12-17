import numpy as np
import matplotlib.pyplot as plt


# Определите подынтегральную функцию
def your_function(x):
    a = -1.27
    b = 4.57
    c = 0.14
    d = 0.35
    return a * np.exp(-d * (x - c)**2) + b


# Задайте пределы интегрирования
x1 = -3.6
x2 = 0.5

# Создайте массив значений x
x_values = np.linspace(x1, x2, 30)

# Вычислите значения подынтегральной функции для каждого x
y_values = your_function(x_values)

# Постройте график
plt.plot(x_values, y_values, label='Подынтегральная функция')
plt.xlabel('x')
plt.ylabel('y')
plt.title('График подынтегральной функции')
plt.legend()
plt.show()
