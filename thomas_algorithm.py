import numpy as np

# Читання матриці з файлу
def read_matrix_from_file(file_name):
    try:
        with open(file_name, 'r') as input_file:
            matrix = []
            for line in input_file:
                if line.strip():
                    row = list(map(float, line.split()))
                    matrix.append(row)
            return np.array(matrix)
    except FileNotFoundError:
        print(f"Error: Unable to open file {file_name}")
        return None

def split_matrix(matrix):
    # Витягуємо квадратну матрицю A (всі стовпці, крім останнього)
    A = matrix[:, :-1]
    # Витягуємо вектор стовпчик b (останній стовпець)
    b = matrix[:, -1]
    return A, b

# Виведення матриці
def display_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{elem:.2f}" for elem in row))

# Перевірка на тридіагональність
def is_tridiagonal_matrix(matrix):
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            if abs(i - j) > 1 and matrix[i, j] != 0:
                return False
    return True

# Перевірка на збіжність (діагональне домінування)
def check_convergence(matrix):
    size = len(matrix)
    for i in range(size):
        diag = abs(matrix[i, i])
        off_diag_sum = sum(abs(matrix[i, j]) for j in range(size) if j != i)
        if diag <= off_diag_sum:
            print(f"Warning: Row {i + 1} does not satisfy the diagonal dominance condition.")
            return False
    return True

# Розв'язання системи методом прогонки (метод Томаса)
def solve_tridiagonal(matrix, epsilon=1e-8):
    size = matrix.shape[0]

    # Перевірка розміру матриці
    if matrix.shape[1] != size + 1:
        print("Error: Input matrix must be of size n x (n+1) for tridiagonal systems.")
        return None

    A = matrix[:, :-1]
    b = matrix[:, -1]

    # Перевірка тридіагональності
    if not is_tridiagonal_matrix(A):
        print("Error: The matrix is not tridiagonal. This method only works for tridiagonal matrices.")
        return None
    else:
        print("Matrix is tridiagonal because non-zero elements exist only on the main diagonal and adjacent diagonals.")

    # Перевірка діагонального домінування
    if not check_convergence(A):
        print("Warning: The matrix might not guarantee convergence, but the method will attempt to solve it.")
    else:
        print("Matrix satisfies the diagonal dominance condition, ensuring convergence.")

    alpha = np.zeros(size)
    beta = np.zeros(size)

    print("\nПрямий хід:")
    # Ініціалізація
    if abs(A[0, 0]) < epsilon:
        print("Error: Zero or near-zero pivot at the first row. Cannot proceed.")
        return None
    alpha[0] = -A[0, 1] / A[0, 0]
    beta[0] = b[0] / A[0, 0]
    print(f"Step 1: alpha[0] = {alpha[0]:.4f}, beta[0] = {beta[0]:.4f}")

    # Прямий хід
    for i in range(1, size):
        denominator = A[i, i] + A[i, i - 1] * alpha[i - 1]
        if abs(denominator) < epsilon:
            print(f"Error: Zero or near-zero pivot detected at row {i + 1}. Cannot proceed.")
            return None
        if i < size - 1:
            alpha[i] = -A[i, i + 1] / denominator
        beta[i] = (b[i] - A[i, i - 1] * beta[i - 1]) / denominator
        print(f"Step {i + 1}: alpha[{i}] = {alpha[i]:.4f}, beta[{i}] = {beta[i]:.4f}")

    # Зворотній хід
    print("\nЗворотній хід:")
    solutions = np.zeros(size)
    solutions[-1] = beta[-1]
    print(f"Step {size}: x[{size}] = {solutions[-1]:.4f}")

    for i in range(size - 2, -1, -1):
        solutions[i] = alpha[i] * solutions[i + 1] + beta[i]
        print(f"Step {i + 1}: x[{i + 1}] = {solutions[i]:.4f}")

    return solutions


# Основна функція
if __name__ == "__main__":
    file_name = "matrix_ta.txt"
    matrix = read_matrix_from_file(file_name)

    if matrix is not None:
        print("Input Matrix:")
        display_matrix(matrix)

        # Введення точності
        while True:
            try:
                epsilon = float(input("Enter the desired precision (e.g., 1e-8): "))
                if epsilon <= 0:
                    raise ValueError("Precision must be a positive number.")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")

        solutions = solve_tridiagonal(matrix, epsilon)
        if solutions is not None:
            print("\nThe solution is:")
            for i, sol in enumerate(solutions, start=1):
                print(f"x{i} = {sol:.4f}")
            
            # Перевірка правильності розв'язку
            A, b = split_matrix(matrix)
            result = np.dot(A, solutions)  # Обчислення A * x
            residual = b - result          # Залишкова похибка

            # Округлення залишкової похибки для читабельності
            residual_rounded = np.round(residual, decimals=int(abs(np.log10(epsilon))) + 2)

            # Виведення результатів
            print("\nОбчислене Ax:")
            print(result)
            print("Залишкова похибка (b - Ax), округлена до значущих цифр:")
            print(residual_rounded)