import csv
import numpy as np
import matplotlib.pyplot as plt


# 1. Підготовка та зчитування даних
def create_sample_csv(filename):
    #Створює CSV файл з даними для Варіанту 2, якщо його немає
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RPS', 'CPU'])
        # Дані Варіанту 2
        data = [[50, 20], [100, 35], [200, 60], [400, 110], [800, 210]]
        writer.writerows(data)


def read_data(filename):
    #Зчитує дані з CSV файлу.
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['RPS']))
            y.append(float(row['CPU']))
    return np.array(x), np.array(y)



# 2. Математичні функції (Метод Ньютона)
def divided_differences(x, y):
    #Обчислює таблицю розділених різниць та повертає коефіцієнти Ньютона
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            # Рекурентна формула для розділених різниць
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    # Коефіцієнти для інтерполяції вперед - це верхній рядок таблиці
    return coef[0, :]


def newton_polynomial(coef, x_data, x):
    #Обчислює значення інтерполяційного многочлена Ньютона у точці x
    n = len(x_data)
    p = coef[0]
    omega = 1.0

    for k in range(1, n):
        # Факторіальний множник w_k(x) = (x - x_0)(x - x_1)...(x - x_k-1)
        omega *= (x - x_data[k - 1])
        p += coef[k] * omega

    return p


def w_n(x_data, x):
    #Обчислення функції w_n(x) для оцінки похибки
    omega = 1.0
    for xi in x_data:
        omega *= (x - xi)
    return omega



# 3. Основна логіка виконання
def main():
    filename = "data_var2.csv"
    create_sample_csv(filename)

    # 1) Зчитування даних
    x_nodes, y_nodes = read_data(filename)
    print(f"Вузли інтерполяції (RPS): {x_nodes}")
    print(f"Значення функції (CPU %): {y_nodes}\n")

    # 2) Побудова таблиці розділених різниць
    coefs = divided_differences(x_nodes, y_nodes)
    print("Коефіцієнти многочлена Ньютона (розділені різниці f(x0,...,xk)):")
    for i, c in enumerate(coefs):
        print(f"  Порядок {i}: {c:.6f}")

    # 3) Оцінка CPU при 600 RPS
    rps_target = 600
    cpu_predicted = newton_polynomial(coefs, x_nodes, rps_target)
    print(f"\n---> Прогнозоване навантаження CPU для {rps_target} RPS: {cpu_predicted:.2f} % <---")

    # 4) Побудова графіків CPU = f(RPS)
    x_dense = np.linspace(min(x_nodes), max(x_nodes), 500)
    y_dense = [newton_polynomial(coefs, x_nodes, xi) for xi in x_dense]

    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, y_dense, label='Інтерполяційний многочлен Ньютона $N_n(x)$', color='blue')
    plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Експериментальні дані (вузли)')
    plt.scatter([rps_target], [cpu_predicted], color='green', marker='*', s=150, zorder=6,
                label=f'Прогноз (600 RPS, CPU={cpu_predicted:.1f}%)')

    plt.title('Прогнозування навантаження на CPU від RPS (Варіант 2)')
    plt.xlabel('RPS (запити за секунду)')
    plt.ylabel('CPU (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # 5) Дослідження впливу кількості вузлів (5, 10, 20) та похибок
    print("\n--- Дослідження впливу кількості вузлів (Аналіз похибки) ---")

    # Базова "ідеальна" функція для симуляції - наш поліном на 5 вузлах
    def true_function(x):
        return newton_polynomial(coefs, x_nodes, x)

    node_counts = [5, 10, 20]
    plt.figure(figsize=(12, 8))

    for n in node_counts:
        # Генеруємо рівномірну сітку з n вузлів на відрізку [50, 800]
        x_sim_nodes = np.linspace(min(x_nodes), max(x_nodes), n)
        y_sim_nodes = [true_function(xi) for xi in x_sim_nodes]

        # Будуємо поліном для нової сітки
        sim_coefs = divided_differences(x_sim_nodes, y_sim_nodes)

        # Обчислюємо похибку на густій сітці
        errors = [abs(true_function(xi) - newton_polynomial(sim_coefs,x_sim_nodes, xi)) for xi in x_dense]
        max_error = max(errors)

        print(f"Максимальна похибка для n={n} вузлів: {max_error:.2e}")

        plt.plot(x_dense, errors, label=f'Похибка $\epsilon(x)$ для n={n}')

    plt.title('Графік похибок інтерполяції при різній кількості вузлів')
    plt.xlabel('RPS')
    plt.ylabel('Абсолютна похибка $\epsilon(x)$')
    plt.yscale('log')  # Логарифмічна шкала для кращої видимості
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()