import csv
import matplotlib.pyplot as plt


filename = 'data.csv'


sample_data = [
    ["Month", "Temp"],
    [1, -3.5], [2, -1.2], [3, 5.0], [4, 12.1],
    [5, 17.5], [6, 21.0], [7, 23.5], [8, 22.1],
    [9, 16.0], [10, 9.5], [11, 2.5], [12, -2.0]
]

with open(filename, 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sample_data)

# 1. Зчитування вхідних даних
x = []
y = []
with open(filename, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Пропускаємо заголовок (Month, Temp)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

n_points = len(x)

# 2. Функції МНК та розв'язку СЛАР
def form_matrix(x_vals, m):
    """Формує матрицю A розміром (m+1) x (m+1)"""
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum((xi ** (i + j)) for xi in x_vals)
    return A

def form_vector(x_vals, y_vals, m):
    """Формує вектор вільних членів b розміром (m+1)"""
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(yi * (xi ** i) for xi, yi in zip(x_vals, y_vals))
    return b

def gauss_solve(A_in, b_in):
    """Метод Гауса з вибором головного елемента по стовпцях"""
    A = [row[:] for row in A_in]
    b = b_in[:]
    n = len(b)

    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i

        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        if abs(A[k][k]) < 1e-12:
            continue

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x_sol[j] for j in range(i + 1, n))
        if A[i][i] != 0:
            x_sol[i] = (b[i] - sum_ax) / A[i][i]
    return x_sol

def polynomial(x_vals, coef):
    """Обчислення значень полінома для масиву x"""
    y_poly = []
    for xi in x_vals:
        val = sum(c * (xi ** i) for i, c in enumerate(coef))
        y_poly.append(val)
    return y_poly

def variance(y_true, y_approx):
    """Обчислення дисперсії (середнього квадрата похибки)"""
    n = len(y_true)
    return sum((yt - ya) ** 2 for yt, ya in zip(y_true, y_approx)) / n

# 3. Вибір оптимального ступеня полінома
max_degree = 10
variances = []
all_coefficients = []

for m in range(1, max_degree + 1):
    A = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve(A, b_vec)
    all_coefficients.append(coef)

    y_approx = polynomial(x, coef)
    var = variance(y, y_approx)
    variances.append(var)
    print(f"Степінь m={m}, Дисперсія: {var:.4f}")

min_variance = min(variances)
optimal_m = variances.index(min_variance) + 1
optimal_coef = all_coefficients[optimal_m - 1]

print(f"\n---> Оптимальний степінь полінома: {optimal_m}")

# 4. Прогноз та похибка для оптимального m
y_optimal_approx = polynomial(x, optimal_coef)
errors = [yt - ya for yt, ya in zip(y, y_optimal_approx)]

x_future = [13, 14, 15]
y_future = polynomial(x_future, optimal_coef)

print(f"Прогноз на 13-й місяць: {y_future[0]:.2f} °C")
print(f"Прогноз на 14-й місяць: {y_future[1]:.2f} °C")
print(f"Прогноз на 15-й місяць: {y_future[2]:.2f} °C")

# 5. Візуалізація результатів
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(range(1, max_degree + 1), variances, marker='o', color='purple')
plt.title('Залежність дисперсії від степеня апроксимуючого многочлена')
plt.xlabel('Степінь m')
plt.ylabel('Дисперсія')
plt.grid(True)
plt.xticks(range(1, max_degree + 1))

plt.subplot(3, 1, 2)
plt.scatter(x, y, color='blue', label='Фактичні дані', zorder=5)
plt.plot(x, y_optimal_approx, color='red', label=f'Апроксимація (m={optimal_m})')
plt.scatter(x_future, y_future, color='green', marker='*', s=100, label='Прогноз', zorder=5)
plt.title('Апроксимація температурних даних методом найменших квадратів')
plt.xlabel('Місяць')
plt.ylabel('Температура')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, errors, color='orange', marker='s')
plt.axhline(0, color='black', linewidth=1)
plt.title('Похибка апроксимації ε(x)')
plt.xlabel('Місяць')
plt.ylabel('Похибка')
plt.grid(True)

plt.tight_layout()
plt.show()