import numpy as np
from scipy.optimize import curve_fit

# Заданные экспериментальные данные
i_data = np.array([1, 2, 3, 4])
x_data = np.array([-3.6, -2.1, -0.8, 1.6])
y_data = np.array([18.5, 6.5, -0.7, -12.8])


# Аппроксимация полиномом первой степени (y = ax + b)
def linear_fit(x, a, b):
    return a * x + b


# Аппроксимация полиномом второй степени (y = ax^2 + bx + c)
def quadratic_fit(x, a, b, c):
    return a * x**2 + b * x + c


sum_x = np.sum(x_data)
sum_x_squared = np.sum(x_data**2)
sum_x_cubed = np.sum(x_data**3)
sum_x_to_4th = np.sum(x_data**4)

sum_y = np.sum(y_data)
sum_xy = np.sum(x_data * y_data)
sum_x_squared_y = np.sum((x_data**2) * y_data)

# Вывод результатов
print("Сумма x:", sum_x)
print("Сумма x^2:", sum_x_squared)
print("Сумма x^3:", sum_x_cubed)
print("Сумма x^4:", sum_x_to_4th)
print("Сумма y:", sum_y)
print("Сумма xy:", sum_xy)
print("Сумма x^2y:", sum_x_squared_y)


# МНК для полинома первой степени
params_linear, covariance_linear = curve_fit(linear_fit, x_data, y_data)
residuals_linear = y_data - linear_fit(x_data, *params_linear)
mse_linear = np.mean(residuals_linear**2)

# МНК для полинома второй степени
params_quadratic, covariance_quadratic = curve_fit(quadratic_fit, x_data, y_data)
residuals_quadratic = y_data - quadratic_fit(x_data, *params_quadratic)
mse_quadratic = np.mean(residuals_quadratic**2)

# Вывод результатов
print("Параметры полинома первой степени (ax + b):", params_linear)
print("Критерий МНК для полинома первой степени:", mse_linear)

print("\nПараметры полинома второй степени (ax^2 + bx + c):", params_quadratic)
print("Критерий МНК для полинома второй степени:", mse_quadratic)
