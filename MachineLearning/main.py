import numpy as np
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

def create_basis_matrix(x, M):
    basis_matrix = np.zeros((len(x), M+1))
    for i in range(M+1):
        basis_matrix[:, i] = x**i
    return basis_matrix

def solve_regression(x, t, M):
    F = create_basis_matrix(x, M)
    w = np.linalg.inv(np.transpose(F) @ F) @ np.transpose(F) @ t
    return w

def regression_output(x, w):
    M = len(w) - 1
    F = create_basis_matrix(x, M)
    y = np.zeros_like(x)
    for j in range(M+1):
        y += w[j] * F[:, j]
    return y

plt.figure(figsize=(15, 10))

# График для M=1
plt.subplot(2, 2, 1)
plt.plot(x, z, color='blue', label='True Function z(x)')
plt.scatter(x, t, color='skyblue', label='Data Points t(x)')
w = solve_regression(x, t, 1)
plt.plot(x, regression_output(x, w), color='r', label='Regression M=1')
plt.legend()

# График для M=8
plt.subplot(2, 2, 2)
plt.plot(x, z, color='blue', label='True Function z(x)')
plt.scatter(x, t, color='skyblue', label='Data Points t(x)')
w = solve_regression(x, t, 8)
plt.plot(x, regression_output(x, w), color='r', label='Regression M=8')
plt.legend()

# График для M=100
plt.subplot(2, 2, 3)
plt.plot(x, z, color='blue', label='True Function z(x)')
plt.scatter(x, t, color='skyblue', label='Data Points t(x)')
w = solve_regression(x, t, 100)
plt.plot(x, regression_output(x, w), color='r', label='Regression M=100')
plt.legend()

# График зависимости ошибки от степени полинома
plt.subplot(2, 2, 4)
errors = []
for M in range(1, 101):
    F = create_basis_matrix(x, M)
    w = np.linalg.inv(np.transpose(F) @ F) @ np.transpose(F) @ t
    y_pred = F @ w
    error = 0.5 * np.sum((t - y_pred)**2)
    errors.append(error)

plt.plot(range(1, 101), errors)
plt.xlabel('Degree of Polynomial (M)')
plt.ylabel('Error')
plt.title('Error vs. Degree of Polynomial')

plt.tight_layout()
plt.show()
