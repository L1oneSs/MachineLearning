from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

error_graph = []
iteration_graph = []

california_housing = fetch_california_housing()

x = california_housing.data
y = california_housing.target

x_train, x_test, t_train, t_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Вычисляем среднее значение и стандартное отклонение для каждого признака
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

X_train_scaled = (x_train - mean) / std

X_test_scaled = (x_test - mean) / std

import numpy as np


def mse_loss(F_loc, t_loc, w_loc):
    if w_loc is None:
        return None
    return (1 / len(F_loc)) * np.sum((t_loc - F_loc @ w_loc.T) ** 2)


def gradient_mse(X, t, w):
    N = len(t)
    gradient = -2 / N * np.dot(X.T, (t - np.dot(X, w)))
    return gradient


def gradient_descent(X, t, initial_weights, learning_rate=0.00000001, max_iters=1000, epsilon=1e-6, alpha=1 ** (-10)):
    w = initial_weights
    iterations = 0

    while True:
        prev_w = w.copy()
        error = mse_loss(X, t, w)
        # print(error)
        gradient = -(t.T @ X).T + (w.T @ (X.T @ X)).T + alpha * w.T
        # gradient = gradient_mse(X, y, w) + 2 * alpha * w
        w -= learning_rate * gradient

        error_graph.append(error)
        iteration_graph.append(iterations)

        iterations += 1

        if iterations >= max_iters:
            break

        if np.linalg.norm(w - prev_w) < epsilon:
            break

        if np.linalg.norm(gradient) < epsilon:
            break

    return w, iterations


# Функция для создания базисных функций

def create_basis_matrix(x_loc, functions):
    if functions is None:
        functions = [0, np.cos, np.sin, np.tan]
    basis_matrix = np.zeros((len(x_loc), len(functions) * x_loc.shape[1]))
    for i, func in enumerate(functions):
        if callable(func):
            basis_matrix[:, i * x_loc.shape[1]:(i + 1) * x_loc.shape[1]] = func(x_loc)
        else:
            basis_matrix[:, i * x_loc.shape[1]:(i + 1) * x_loc.shape[1]] = x_loc ** func
    return basis_matrix


print(X_train_scaled)

X_train_basis = create_basis_matrix(X_train_scaled, None)
X_test_basis = create_basis_matrix(X_test_scaled, None)

# Начальное приближение инициализируется случайными значениями
initial_weights = np.random.normal(loc=0, scale=0.1, size=X_train_basis.shape[1])

# Применяем градиентный спуск
learned_weights, num_iterations = gradient_descent(X_train_basis, t_train, initial_weights)

print("Error on train: ", mse_loss(X_train_basis, t_train, learned_weights))
print("Error on test: ", mse_loss(X_test_basis, t_test, learned_weights))

#print("Learned weights:", learned_weights)
print("Number of iterations:", num_iterations)

import matplotlib.pyplot as plt

plt.plot(iteration_graph, error_graph)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error vs. Iteration')
plt.grid(True)
plt.show()


# <=========================================>
x_train, x_test, t_train, t_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=0.25, random_state=42)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train_scaled = (x_train - mean) / std

x_test_scaled = (x_test - mean) / std

x_val_scaled = (x_val - mean) / std

import random

alpha_values = [10 ** i for i in range(-30, -10)]
learning_rate_values = [10 ** i for i in range(-20, -5)]
loc_values = [0, 0.1, 0.5]
scale_values = [0.01, 0.1, 1]

# Параметры валидации
num_epochs = 30  # Количество эпох
best_degrees = None
best_alpha = None
best_learning_rate = None
best_loc = None
best_scale = None
best_w = None
best_model_error = float('inf')

for epoch in range(num_epochs):
    print(f"Validation. epoch={epoch}")
    # Случайным образом выбираем значения параметров
    alpha = random.choice(alpha_values)
    learning_rate = random.choice(learning_rate_values)
    loc = random.choice(loc_values)
    scale = random.choice(scale_values)

    # Создаем базисную матрицу на основе полиномов заданной степени
    degrees = []
    poly_degrees = [i for i in range(0, 20)]
    num_functions = random.randint(1, 5)
    while len(degrees) < num_functions:
        func = random.choice(poly_degrees)
        degrees.append(func)
        poly_degrees.remove(func)

    X_train_polynomial_basis = create_basis_matrix(x_train_scaled, degrees)
    X_val_polynomial_basis = create_basis_matrix(x_val_scaled, degrees)
    #X_test_polynomial_basis = create_basis_matrix(x_test_scaled, degrees)

    # Применяем градиентный спуск для получения модели с выбранными параметрами
    initial_weights = np.random.normal(loc=loc, scale=scale, size=X_train_polynomial_basis.shape[1])
    learned_weights, num_iterations = gradient_descent(X_train_polynomial_basis, t_train, initial_weights,
                                                       learning_rate=learning_rate, alpha=alpha)

    # Оцениваем качество модели на валидационных данных
    model_error = mse_loss(X_val_polynomial_basis, t_val, learned_weights)

    # Сохраняем лучшую модель
    if model_error < best_model_error:
        best_w = learned_weights
        best_model_error = model_error
        best_degrees = degrees
        best_alpha = alpha
        best_learning_rate = learning_rate
        best_loc = loc
        best_scale = scale


# Оцениваем качество моделей на тестовых данных с лучшими весами
poly_model_error = mse_loss(create_basis_matrix(x_test_scaled, best_degrees), t_test, best_w)
linear_model_error = mse_loss(x_test_scaled, t_test, best_w)

print("\nModel after validation")
if best_model_error is not None:
    print(f"model_error: {best_model_error}\n"
          f"polynomial_degrees: {best_degrees}\n"
          f"alpha: {best_alpha}\n"
          f"learning_rate: {best_learning_rate}\n"
          f"loc: {best_loc}\n"
          f"scale: {best_scale}\n")
else:
    print("Can't reach best params")

print("Standard Model")
print(f"model_error on test: {linear_model_error}\n")

print("Poly Model")
print(f"model_error on test: {poly_model_error}\n")
