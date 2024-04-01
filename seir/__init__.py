import numpy as np
import matplotlib.pyplot as plt

# Параметры модели
beta = 0.3  # Контактная скорость
sigma = 0.1  # Скорость инкубации
gamma = 0.05  # Скорость выздоровления

# Начальные условия
S0 = 990
E0 = 10
I0 = 0
R0 = 0
N = S0 + E0 + I0 + R0  # Общая численность населения

# Временные параметры
t_end = 100
dt = 0.1
steps = int(t_end / dt)

# Инициализация массивов
S = np.zeros(steps)
E = np.zeros(steps)
I = np.zeros(steps)
R = np.zeros(steps)
t = np.linspace(0, t_end, steps)

# Установка начальных значений
S[0] = S0
E[0] = E0
I[0] = I0
R[0] = R0

# Решение модели методом конечных разностей
for step in range(1, steps):
    S[step] = S[step - 1] - (beta * S[step - 1] * I[step - 1] / N) * dt
    E[step] = E[step - 1] + (beta * S[step - 1] * I[step - 1] / N - sigma * E[step - 1]) * dt
    I[step] = I[step - 1] + (sigma * E[step - 1] - gamma * I[step - 1]) * dt
    R[step] = R[step - 1] + (gamma * I[step - 1]) * dt

# Визуализация результатов
plt.plot(t, S, label='Восприимчивые')
plt.plot(t, E, label='Подверженные')
plt.plot(t, I, label='Инфицированные')
plt.plot(t, R, label='Выздоровевшие')
plt.xlabel('Время')
plt.ylabel('Численность населения')
plt.legend()
plt.show()