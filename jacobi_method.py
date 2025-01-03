import numpy as np
from tabulate import tabulate

# Читання розширеної матриці з файлу
def read_matrix(filename):
    with open(filename, "r") as f:
        matrix = [list(map(float, line.split())) for line in f.readlines()]
    return np.array(matrix)

# Перевірка умови діагонального домінування
def check_diagonal_dominance(A):
    for i in range(A.shape[0]):
        diag = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in range(A.shape[1] - 1) if j != i)
        if diag < off_diag_sum:
            return False
    return True

# Метод Якобі
def jacobi_method(A, b, tol=1e-4, max_iter=100):
    n = len(b)
    x = np.zeros(n)  # Початкове наближення
    history = []

    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        
        # Збереження ітерації
        history.append(x_new.copy())

        # Перевірка точності
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, history, True
        x = x_new
    
    return x, history, False

# Перевірка розв'язку
def check_solution(A, x, b):
    return np.allclose(A @ x, b, atol=1e-4)

# Виведення матриці
def display_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{elem:.2f}" for elem in row))

# Основна функція
def main():
    # Зчитуємо матрицю з файлу
    filename = "matrix_j.txt"
    matrix = read_matrix(filename)
    print("Вхідна розширена матриця:")
    display_matrix(matrix)
    A = matrix[:, :-1]  # Коефіцієнти
    b = matrix[:, -1]   # Вектор правої частини

    # Запит точності у користувача
    try:
        tol = float(input("Введіть точність (ε, наприклад 0.0001): "))
        if tol <= 0:
            raise ValueError("Точність має бути додатним числом.")
    except ValueError as e:
        print(f"Помилка введення: {e}")
        return

    # Перевірка умови діагонального домінування
    if not check_diagonal_dominance(A):
        print("Матриця не задовольняє умову діагонального домінування. Метод Якобі може не збігатися.")
        return

    print("Матриця задовольняє умову діагонального домінування. Починаємо ітерації.\n")

    # Розв'язок методом Якобі
    solution, history, converged = jacobi_method(A, b, tol=tol)
    
    # Вивід таблиці ітерацій
    headers = [f"x{i+1}" for i in range(len(solution))]
    table = [list(map(lambda x: round(x, 6), h)) for h in history]
    print("Таблиця ітерацій:")
    print(tabulate(table, headers=headers, tablefmt="grid", showindex=range(1, len(history) + 1)))

    if converged:
        print(f"\nМетод Якобі збігся за точністю {tol}. Розв'язок:")
    else:
        print("\nМетод Якобі не збігся за задану кількість ітерацій. Поточний наближення:")
    print(tabulate([solution], headers=headers, tablefmt="grid"))

    # Перевірка розв'язку
    if check_solution(A, solution, b):
        print("\nРозв'язок перевірено: він задовольняє СЛАР.")
    else:
        print("\nРозв'язок не задовольняє СЛАР.")

if __name__ == "__main__":
    main()
