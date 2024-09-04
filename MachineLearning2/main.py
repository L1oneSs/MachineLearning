import random

import numpy as np
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

best_w = []
best_b = []
best_a = []


def create_basis_functions_mas():
    mas = []
    func_mas = [
        np.cos, np.sin, np.tan, np.exp, np.sqrt, np.cosh,
        np.sinh, np.tanh
    ]

    for i in range(1, 101):
        mas.append(i)
    mas.extend(func_mas)
    return mas


def create_basis_matrix(x_loc, functions):
    basis_matrix = np.zeros((len(x_loc), len(functions)))
    for i in range(len(functions)):
        if callable(functions[i]):
            basis_matrix[:, i] = functions[i](x_loc)
        else:
            basis_matrix[:, i] = x_loc ** functions[i]
    return basis_matrix


def calc_w(a, F_loc, t_loc):
    I = np.eye(np.shape(np.transpose(F_loc) @ F_loc)[0])
    return np.linalg.inv(np.transpose(F_loc) @ F_loc + a * np.transpose(I)) @ np.transpose(F_loc) @ t_loc


def calc_E(w_loc, F_loc, t_loc):
    return (1 / len(F_loc)) * np.sum((t_loc - F_loc @ w_loc.T) ** 2)


def regression_output(x_loc, functions, w_loc):
    F_loc = create_basis_matrix(x_loc, functions)
    y_loc = np.zeros_like(x_loc)
    for j in range(len(w_loc)):
        y_loc += w_loc[j] * F_loc[:, j]
    return y_loc


alphas = [0, 10 ** (-50), 10 ** (-40), 10 ** (-30), 10 ** (-20), 10 ** (-10), 0.00001,
          0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50]

indices = np.random.permutation(N)

# Разделение индексов на train, validation, test
train_idx = indices[:int(0.6 * N)]
validation_idx = indices[int(0.6 * N):int(0.8 * N)]
test_idx = indices[int(0.8 * N):]

x_train, x_validation, x_test = x[train_idx], x[validation_idx], x[test_idx]
t_train, t_validation, t_test = t[train_idx], t[validation_idx], t[test_idx]

epochs = 20
E_min = np.finfo(float).max

for i in range(0, epochs):
    basis_init = create_basis_functions_mas()
    # Генерация массива базисных функций
    b = [0]
    num_functions = random.randint(25, 35)
    while len(b) < num_functions:
        function = random.choice(basis_init)
        if function not in b:
            b.append(function)
            basis_init.remove(function)
    # Создание матрицы плана

    F_train = create_basis_matrix(x_train, b)

    for alpha in alphas:
        w = calc_w(alpha, F_train, t_train)
        F_validation = create_basis_matrix(x_validation, b)
        E = calc_E(w, F_validation, t_validation)
        print(E, "<>", E_min)
        if E < E_min:
            E_min = E
            best_w = w
            best_b = b
            best_a = alpha

# Ошибка для тестовой части
F = create_basis_matrix(x_test, best_b)
E_test = calc_E(best_w, F, t_test)

# Регрессия
y = regression_output(x, best_b, best_w)

print(f"Значение ошибки на тестовой части: {E_test}\n")
print(f"Значение лучшего коэффициента регуляризации: {best_a}\n")
funcs = []
for func in best_b:
    if callable(func):
        if func.__name__ == '1':
            funcs.append('1')
        else:
            funcs.append(func.__name__ + '(x)')
    else:
        if func == 1:
            funcs.append('1')
        else:
            funcs.append(f'x^{func}')
print(f"Лучший набор базисных функций: {funcs}")

plt.plot(x, z, color='blue', label='True Function z(x)')
plt.scatter(x, t, color='skyblue', label='Data Points t(x)')
plt.plot(x, y, color='r', label='Regression')
plt.legend()
plt.show()