import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# 1. СТВОРЕННЯ ТА ЗЧИТУВАННЯ ДАНИХ
def create_data_file(filename):

    data = [
        ['RPS', 'CPU'],
        [50, 20],
        [100, 35],
        [200, 60],
        [400, 110],
        [800, 210]
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def read_data(filename):

    x, y = [], []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['RPS']))
            y.append(float(row['CPU']))
    return x, y


# 2. МЕТОД НЬЮТОНА
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef


def newton_polynomial(coef, x_data, x):
    n = len(x_data) - 1
    p = coef[0][n]
    for k in range(1, n + 1):
        p = coef[0][n - k] + (x - x_data[n - k]) * p
    return p


# 3. ФАКТОРІАЛЬНІ МНОГОЧЛЕНИ
def finite_differences(y):
    n = len(y)
    delta = np.zeros([n, n])
    delta[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            delta[i][j] = delta[i + 1][j - 1] - delta[i][j - 1]
    return delta


def factorial_polynomial(delta, t):
    n = len(delta) - 1
    result = delta[0][0]
    t_factorial = 1

    for k in range(1, n + 1):
        t_factorial *= (t - k + 1)
        result += (delta[0][k] / math.factorial(k)) * t_factorial

    return result


# 4. КУБІЧНИЙ СПЛАЙН
def compute_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    c = np.zeros(n + 1)

    alpha[0] = 0
    beta[0] = 0
    for i in range(1, n):
        A = h[i - 1]
        B = 2 * (h[i - 1] + h[i])
        D = h[i]
        F = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        denom = A * alpha[i - 1] + B
        alpha[i] = -D / denom
        beta[i] = (F - A * beta[i - 1]) / denom

    c[n] = 0
    for i in range(n - 1, 0, -1):
        c[i] = alpha[i] * c[i + 1] + beta[i]

    a_coef = y[:-1]
    b_coef = np.zeros(n)
    d_coef = np.zeros(n)
    for i in range(n):
        d_coef[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - (h[i] * (c[i + 1] + 2 * c[i])) / 3

    return a_coef, b_coef, c[:-1], d_coef


def spline_eval(x_val, x_nodes, a_coef, b_coef, c_coef, d_coef):
    if x_val < x_nodes[0] or x_val > x_nodes[-1]:
        return 0
    idx = np.searchsorted(x_nodes, x_val) - 1
    idx = max(0, min(idx, len(a_coef) - 1))
    dx = x_val - x_nodes[idx]
    return a_coef[idx] + b_coef[idx] * dx + c_coef[idx] * dx ** 2 + d_coef[idx] * dx ** 3


# ГОЛОВНА ПРОГРАМА
if __name__ == "__main__":
    # ЧАСТИНА 1: Базове завдання

    file_name = "data.csv"
    create_data_file(file_name)  # Спочатку створюємо файл із даними
    x_data, y_data = read_data(file_name)  # Тепер зчитуємо дані з нього

    target_rps = 600

    print(" БАЗОВІ ОБЧИСЛЕННЯ ")

    # Вивід таблиці розділених різниць
    coef_matrix = divided_differences(x_data, y_data)
    print("Таблиця розділених різниць:")
    for row in coef_matrix:
        print([round(float(val), 5) for val in row if abs(val) > 0.000001])
    print("-" * 40)

    # Прогноз Ньютоном
    predicted_cpu_newton = newton_polynomial(coef_matrix, x_data, target_rps)
    print(f"Прогноз (Ньютон) при {target_rps} RPS: {predicted_cpu_newton:.2f}%")

    # Прогноз Факторіальним многочленом
    num_points = 5
    x_eq = np.linspace(min(x_data), max(x_data), num_points)
    y_eq = [newton_polynomial(coef_matrix, x_data, xi) for xi in x_eq]

    delta_matrix = finite_differences(y_eq)
    h = x_eq[1] - x_eq[0]
    t_target = (target_rps - x_eq[0]) / h

    predicted_cpu_fact = factorial_polynomial(delta_matrix, t_target)
    print(f"Прогноз (Факторіальний) при {target_rps} RPS: {predicted_cpu_fact:.2f}%\n")

    # Графік 1 (Базовий)
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    y_smooth_newton = [newton_polynomial(coef_matrix, x_data, xi) for xi in x_smooth]

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ro', label='Оригінальні точки', markersize=8)
    plt.plot(x_smooth, y_smooth_newton, 'b-', label='Крива Ньютона')
    plt.plot(target_rps, predicted_cpu_newton, 'g*', label=f'Прогноз: {predicted_cpu_newton:.1f}%', markersize=12)
    plt.title('Прогнозування навантаження на CPU')
    plt.xlabel('RPS')
    plt.ylabel('CPU (%)')
    plt.legend()
    plt.grid(True)
    plt.show()  # Програма зупиниться тут, поки ти не закриєш перше вікно з графіком

    # ЧАСТИНА 2: Дослідницька (5, 10, 20 вузлів)
    print(" ДОСЛІДНИЦЬКА ЧАСТИНА ")
    print("Побудова графіків для 5, 10 та 20 вузлів (Власний сплайн)...")

    # Перетворюємо в numpy-масиви для сплайнів
    x_arr = np.array(x_data)
    y_arr = np.array(y_data)

    # Рахуємо коефіцієнти сплайна ОДИН РАЗ на основі базових точок
    a, b, c, d = compute_spline(x_arr, y_arr)

    # Створюємо базову функцію (густу сітку) через наш сплайн
    x_dense = np.linspace(min(x_data), max(x_data), 500)
    y_true = np.array([spline_eval(xi, x_arr, a, b, c, d) for xi in x_dense])

    nodes_list = [5, 10, 20]
    colors = ['blue', 'green', 'red']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Графік інтерполяції
    ax1.plot(x_dense, y_true, 'k--', label='Базова крива (Власний Сплайн)', linewidth=2)

    for n, color in zip(nodes_list, colors):
        # Генеруємо точки x_nodes та вираховуємо їх ідеальне значення через сплайн
        x_nodes = np.linspace(min(x_data), max(x_data), n)
        y_nodes = np.array([spline_eval(xi, x_arr, a, b, c, d) for xi in x_nodes])

        # Будуємо поліном Ньютона для цих нових точок
        coef = divided_differences(x_nodes, y_nodes)
        y_poly = np.array([newton_polynomial(coef, x_nodes, xi) for xi in x_dense])

        ax1.plot(x_dense, y_poly, color=color, label=f'Ньютон (n={n})', alpha=0.8)

        # Графік похибки
        error = np.abs(y_true - y_poly)
        ax2.plot(x_dense, error, color=color, label=f'Похибка (n={n})')

    ax1.set_title('Інтерполяція при 5, 10 та 20 вузлах')
    ax1.set_ylabel('CPU (%)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-10, 250)

    ax2.set_title('Абсолютна похибка інтерполяції')
    ax2.set_xlabel('RPS')
    ax2.set_ylabel('|Похибка| (логарифмічна шкала)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.show()