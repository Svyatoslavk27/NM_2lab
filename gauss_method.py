import numpy as np
from prettytable import PrettyTable

def load_matrix(file_path):
    """Завантажує матрицю з файлу"""
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                row = list(map(float, line.split()))
                matrix.append(row)
    return np.array(matrix)

def print_matrix(matrix, title="Matrix"):
    """Виводить матрицю з вертикальною рискою, якщо це розширена матриця"""
    print(f"\n{title}:")
    rows, cols = matrix.shape
    # Якщо кількість стовпців на 1 більша за кількість рядків, це розширена матриця
    is_augmented = cols == rows + 1
    for row in matrix:
        if is_augmented:
            # Вивід із розділенням основної частини і правого вектора
            print(' '.join(f"{val: .2f}" for val in row[:-1]) + " | " + f"{row[-1]: .2f}")
        else:
            # Звичайний вивід
            print(' '.join(f"{val: .2f}" for val in row))

def print_augmented_with_inverse(augmented_matrix, inverse_matrix, title="Матриця з оберненою"):
    """Виводить розширену матрицю та обернену матрицю у форматі, схожому на об'єднані клітинки"""
    print(f"\n{title}:")

    # Формуємо заголовки
    num_cols_A = augmented_matrix.shape[1] - 1
    num_cols_inv = inverse_matrix.shape[1]
    header_A = "матриця A".ljust(10 * num_cols_A)
    header_b = "b".ljust(10)
    header_inv = "обернена матриця".ljust(10 * num_cols_inv)
    
    # Виводимо заголовок таблиці
    print(f"{header_A} {header_b} {header_inv}")

    # Формуємо рядки таблиці
    for i in range(augmented_matrix.shape[0]):
        # Формуємо рядок для матриці A
        row_A = ' '.join(f"{val:7.2f}" for val in augmented_matrix[i, :-1])
        # Формуємо рядок для b
        row_b = f"{augmented_matrix[i, -1]:7.2f}"
        # Формуємо рядок для оберненої матриці
        row_inv = ' '.join(f"{val:7.2f}" for val in inverse_matrix[i])
        # Підключаємо всі частини
        print(f"{row_A} | {row_b} | {row_inv}")

def gaussian_elimination(matrix):
    """Метод Гаусса з частковим вибором головного елемента для розв'язання системи лінійних рівнянь"""
    size = matrix.shape[0]
    inverse_matrix = np.identity(size)
    determinant = 1.0
    solutions = np.zeros(size)

    # Forward elimination (прямий хід)
    for i in range(size):
        # Частковий вибір головного елемента
        max_row = i + np.argmax(np.abs(matrix[i:, i]))
        if matrix[max_row, i] == 0:
            raise ValueError("Система рівнянь несумісна або має нескінченну кількість розв'язків.")

        # Переставляємо рядки
        if max_row != i:
            matrix[[i, max_row]] = matrix[[max_row, i]]
            inverse_matrix[[i, max_row]] = inverse_matrix[[max_row, i]]
            determinant *= -1  # Змінюємо знак детермінанта через перестановку рядків

        # Нормалізація поточного рядка
        pivot = matrix[i, i]
        determinant *= pivot
        matrix[i] /= pivot
        inverse_matrix[i] /= pivot

        print_augmented_with_inverse(matrix, inverse_matrix, f"Після нормалізації рядка {i + 1}")

        # Виключення нижче поточного рядка
        for j in range(i + 1, size):
            factor = matrix[j, i]
            matrix[j] -= matrix[i] * factor
            inverse_matrix[j] -= inverse_matrix[i] * factor

            print_augmented_with_inverse(matrix, inverse_matrix, f"Після усунення елемента {i + 1}, {j + 1}")

    # Backward substitution (зворотний хід)
    print("\nВиконується зворотне підстановлення...")
    solutions[size - 1] = matrix[size - 1, -1]
    for i in range(size - 2, -1, -1):
        solutions[i] = matrix[i, -1] - np.dot(matrix[i, i+1:size], solutions[i+1:size])

    print(f"\nРішення: {solutions}")

    # Зворотній хід для обчислення оберненої матриці
    for i in range(size - 1, -1, -1):
        for j in range(i):
            factor = matrix[j, i]
            matrix[j] -= matrix[i] * factor
            inverse_matrix[j] -= inverse_matrix[i] * factor

    return solutions, determinant, inverse_matrix

def main():
    # Завантажуємо матрицю з файлу
    matrix = load_matrix("matrix_g.txt")
    size = matrix.shape[0]
    print_matrix(matrix, "Початкова розширена матриця системи рівнянь")

    # Створюємо додаткову матрицю для вирішення
    matrix_augmented = np.hstack((matrix[:, :-1], matrix[:, -1][:, np.newaxis]))

    # Розв'язуємо систему рівнянь методом Гаусса
    solutions, determinant, inverse_matrix = gaussian_elimination(matrix_augmented)

    # Виводимо результат
    print("\nРішення системи рівнянь:")
    for i, sol in enumerate(solutions, 1):
        print(f"x{i} = {sol:.2f}")

    print(f"\nДетермінант матриці: {determinant:.2f}")

    print_matrix(inverse_matrix, "Обернена матриця")

if __name__ == "__main__":
    main()
