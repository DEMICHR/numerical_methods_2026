import numpy as np
import math
import matplotlib.pyplot as plt


# --- 1. Оголошення всіх функцій ---

def moisture_level(t):
    """Розрахунок поточної вологості ґрунту M(t)."""
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)


def deriv_analytical(t):
    """Точне (аналітичне) значення похідної M'(t)."""
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


def diff_central(f, x_val, step):
    """Формула центральної різниці для чисельного диференціювання."""
    return (f(x_val + step) - f(x_val - step)) / (2 * step)


def calculate_aitken(d1, d2, d4):
    """Обчислення за формулою Ейткена та оцінка порядку точності."""
    num_aitken = d2 ** 2 - d4 * d1
    den_aitken = 2 * d2 - (d4 + d1)

    # Запобіжник від ділення на нуль
    aitken_val = num_aitken / den_aitken if den_aitken != 0 else float('inf')

    num_order = d4 - d2
    den_order = d2 - d1
    est_p = (1 / math.log(2)) * math.log(abs(num_order / den_order)) if den_order != 0 else 0

    return aitken_val, est_p


def plot_graphs(h_grid, err_collection, best_h, min_err):
    """Побудова графіків вологості ґрунту та залежності похибки."""
    # Створюємо одне полотно з двома графіками поруч (1 рядок, 2 колонки)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Графік 1: Вологість ґрунту ---
    t_values = np.linspace(0, 10, 500)  # Час від 0 до 10 умовних одиниць
    m_values = moisture_level(t_values)

    ax1.plot(t_values, m_values, color='b', linewidth=2)
    ax1.set_title("Зміна рівня вологості ґрунту з часом", fontsize=14)
    ax1.set_xlabel("Час $t$", fontsize=12)
    ax1.set_ylabel("Вологість $M(t)$", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # --- Графік 2: Залежність похибки від кроку ---
    ax2.loglog(h_grid, err_collection, label="Залежність похибки $R(h)$", color='g', linewidth=2)
    ax2.loglog(best_h, min_err, 'mo', markersize=8,
               label=f'Оптимальний крок: $h_0 \\approx {best_h:.1e}$\nПохибка: $R_0 \\approx {min_err:.1e}$')

    ax2.set_title("Вплив розміру кроку $h$ на похибку", fontsize=14)
    ax2.set_xlabel("Крок сітки $h$ (логарифмічна шкала)", fontsize=12)
    ax2.set_ylabel("Абсолютна похибка $R$ (логарифмічна шкала)", fontsize=12)
    ax2.grid(True, which="both", ls=":", alpha=0.8)
    ax2.legend(fontsize=12)

    plt.tight_layout()  # Автоматично вирівнює відступи між графіками
    plt.savefig('moisture_and_error_plot.png', dpi=300)
    plt.show()


# --- 2. Головна логіка програми ---

def main():
    # Задаємо точку для аналізу
    target_t = 1.0
    true_deriv = deriv_analytical(target_t)
    print(f"Крок 1. Точне значення похідної в точці t0=1: {true_deriv:.8f}")

    # Етап 2: Визначення оптимального кроку
    h_grid = np.logspace(-20, 3, 1000)
    h_grid = h_grid[h_grid != 0]  # Фільтруємо нулі

    # Векторизоване обчислення (Pythonic way: без повільного for)
    approx_derivs = diff_central(moisture_level, target_t, h_grid)
    err_collection = np.abs(approx_derivs - true_deriv)

    idx_min = np.argmin(err_collection)
    best_step = h_grid[idx_min]
    lowest_err = err_collection[idx_min]

    print(f"\nКрок 2. Оптимальний крок h0: {best_step:.2e}")
    print(f"Досягнута точність R0: {lowest_err:.8e}")

    # Етап 3-6: Застосування методу Рунге-Ромберга
    base_step = 1e-3
    print(f"\nКрок 3. Приймаємо h = {base_step}")

    d1 = diff_central(moisture_level, target_t, base_step)
    d2 = diff_central(moisture_level, target_t, 2 * base_step)

    err_R1 = abs(d1 - true_deriv)
    print(f"Крок 4-5. Похідна з кроком h: {d1:.8f}, Похибка R1: {err_R1:.8e}")

    runge_romb_val = d1 + (d1 - d2) / 3
    err_R2 = abs(runge_romb_val - true_deriv)
    print(f"Крок 6. Рунге-Ромберг (уточнене): {runge_romb_val:.8f}, Похибка R2: {err_R2:.8e}")
    print(f"Характер зміни похибки: Похибка зменшилась у {err_R1 / err_R2:.2f} разів порівняно з R1.")

    # Етап 7: Застосування методу Ейткена
    d4 = diff_central(moisture_level, target_t, 4 * base_step)

    aitken_val, est_p = calculate_aitken(d1, d2, d4)
    err_R3 = abs(aitken_val - true_deriv)

    print(f"\nКрок 7. Метод Ейткена (уточнене): {aitken_val:.8f}")
    print(f"Оцінка порядку точності p: {est_p:.2f}")
    print(f"Похибка R3: {err_R3:.8e}")

    # Аналіз результатів для системи поливу
    print("\n--- Висновок щодо режиму поливу ---")
    if true_deriv < 0:
        print("Похідна від'ємна. Це означає, що швидкість зміни вологості падає (ґрунт висихає).")
        print(
            "Оскільки швидкість висихання становить близько -1.82 одиниць вологості за одиницю часу, системі автоматичного поливу слід готуватися до увімкнення, якщо поточна вологість наближається до критичного мінімуму.")
    else:
        print("Похідна додатна. Ґрунт накопичує вологу.")

    # Побудова графіків
    plot_graphs(h_grid, err_collection, best_step, lowest_err)


# --- 3. Точка входу ---
if __name__ == "__main__":
    main()