import requests
import numpy as np
import matplotlib.pyplot as plt


locations = [
    "48.164214,24.536044", "48.164983,24.534836", "48.165605,24.534068",
    "48.166228,24.532915", "48.166777,24.531927", "48.167326,24.530884",
    "48.167011,24.530061", "48.166053,24.528039", "48.166655,24.526064",
    "48.166497,24.523574", "48.166128,24.520214", "48.165416,24.517170",
    "48.164546,24.514640", "48.163412,24.512980", "48.162331,24.511715",
    "48.162015,24.509462", "48.162147,24.506932", "48.161751,24.504244",
    "48.161197,24.501793", "48.160580,24.500537"
]
url = f"https://api.open-elevation.com/api/v1/lookup?locations={'|'.join(locations)}"

try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    print("Помилка API, використовуються резервні дані висот.")
    # Резервні дані на випадок, якщо API недоступне
    results = [{'latitude': float(loc.split(',')[0]), 'longitude': float(loc.split(',')[1]), 'elevation': 1200 + i * 40}
               for i, loc in enumerate(locations)]

n_points = len(results)
print(f"Кількість вузлів: {n_points}")



def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0.0]

for i in range(1, n_points):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)



def cubic_spline_coefficients(x, y):
    n = len(x) - 1
    h = np.diff(x)

    
    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)

    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

   
    A = np.zeros(n)
    B = np.zeros(n)
    for i in range(1, n):
        denominator = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denominator
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denominator

    
    c = np.zeros(n + 1)
    for i in range(n - 1, 0, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    
    a = y[:-1]
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])

    return a, b, c[:-1], d


def evaluate_spline(x_eval, x_nodes, a, b, c, d):
    y_eval = np.zeros_like(x_eval)
    for i in range(len(x_nodes) - 1):
        mask = (x_eval >= x_nodes[i]) & (x_eval <= x_nodes[i + 1])
        dx = x_eval[mask] - x_nodes[i]
        y_eval[mask] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return y_eval



plt.figure(figsize=(12, 8))
plt.scatter(distances, elevations, color='red', zorder=5, label='Вихідні дані (API)')

nodes_variants = [10, 15, 20]
colors = ['blue', 'green', 'orange']

for nodes_count, color in zip(nodes_variants, colors):
   
    indices = np.linspace(0, n_points - 1, nodes_count, dtype=int)
    x_sub = np.array(distances)[indices]
    y_sub = np.array(elevations)[indices]

    a, b, c, d = cubic_spline_coefficients(x_sub, y_sub)

    x_dense = np.linspace(x_sub[0], x_sub[-1], 500)
    y_spline = evaluate_spline(x_dense, x_sub, a, b, c, d)

    plt.plot(x_dense, y_spline, color=color, label=f'Сплайн ({nodes_count} вузлів)')

plt.title('Профіль висоти маршруту: Інтерполяція кубічними сплайнами')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.legend()
plt.grid(True)
plt.show()


print("\n--- Характеристики маршруту ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n_points))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n_points))
print(f"Сумарний спуск (м): {total_descent:.2f}")


x_dense = np.linspace(distances[0], distances[-1], 500)
a, b, c, d = cubic_spline_coefficients(np.array(distances), np.array(elevations))
y_full = evaluate_spline(x_dense, np.array(distances), a, b, c, d)

grad_full = np.gradient(y_full, x_dense) * 100
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")


mass = 80
g = 9.81
energy = mass * g * total_ascent
print("\n--- Механічна енергія ---")
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")
