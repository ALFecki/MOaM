import numpy as np


def modify_matrix(A, A_inv, x, i):
    n = A.shape[0]

    l = np.dot(A_inv, x)
    if l[i] == 0:
        return None

    le = np.copy(l)
    le[i] = -1

    hat_l = -1 / l[i] * le

    Q = np.eye(n)
    Q[:, i] = hat_l

    A_inv_modified = np.dot(Q, A_inv)

    return A_inv_modified


n = int(input("Введите размерность матрицы: "))
A = np.zeros((n, n))
for i in range(n):
    row = input(f"Введите элементы {i+1}-й строки через пробел: ")
    A[i] = list(map(float, row.split()))

A_inv = np.linalg.inv(A)

x_row = input("Введите элементы вектора x через пробел: ")
x = np.array(list(map(float, x_row.split())))

i = int(input("Введите индекс i: ")) - 1

print("Матрица A:")
print(A)
print("Вектор x:")
print(x)

A_inv_modified = modify_matrix(A, A_inv, x, i)
if A_inv_modified is not None:
    print("Матрица A обратима.")
    print("Обратная матрица (A)^(-1):")
    print(A_inv_modified)
else:
    print("Матрица A необратима.")
