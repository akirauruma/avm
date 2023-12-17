import numpy as np
from scipy.interpolate import interp1d

# Заданные экспериментальные точки
i = np.array([1, 2, 3, 4, 5, 6, 7])
x = np.array([-2.4, -1.4, -0.4, 0.6, 1.6, 2.6, 3.6])
y = np.array([1.60, 1.97, 1.21, 1.14, 3.54, 10.22, 22.99])

# Создание кусочно-линейной интерполяции
interp_function = interp1d(x, y, kind='linear', fill_value="extrapolate")

# Значения интерполированной функции в указанных точках
x_interp = np.array([-3.0, -2.3, -0.6, 0.1, 3.7])
y_interp = interp_function(x_interp)

# Вывод результатов
for xi, yi in zip(x_interp, y_interp):
    print(f'При x = {xi}, интерполированное значение y = {yi}')
