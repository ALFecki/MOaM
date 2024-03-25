import numpy as np


def dual_simplex_method(c, A, b, B):
    m, n = A.shape

    while True:
        # Шаг 1
        AB = A[:, B]
        AB_inv = np.linalg.inv(AB)

        # Шаг 2
        cB = c[B]

        # Шаг 3
        y = np.dot(cB, AB_inv)

        # Шаг 4
        k_arr = np.zeros(n)
        k_arr_B = np.dot(AB_inv, b)

        index = 0
        for val in B:
            k_arr[val] = k_arr_B[index]
            index += 1

        # Шаг 5
        if np.all(np.array(k_arr) >= 0):
            x = np.zeros(n)
            x[B] = k_arr_B
            return x

        # Шаг 6
        k = 0
        j = np.argmin(k_arr)

        k = [val for val in k_arr if val < 0].index(k_arr[j])

        # Шаг 7
        delta_y = AB_inv[k]
        mu_j = [[np.dot(delta_y, A[:, i]), i] for i in range(n) if i not in B]

        # Шаг 8
        if np.all(np.array(mu_j[0]) >= 0):
            print("Задача несовметсна, метод завершает работу")

        # Шаг 9
        values = [val[0] for val in mu_j]
        sigmas = [
            (c[val[1]] - np.dot(A[:, val[1]], y)) / val for val in mu_j if val[0] < 0
        ]

        # Шаг 10
        values = [val[0] for val in sigmas]
        j_0 = np.argmin(np.array(values))

        # Шаг 11
        B[k] = j_0



c = np.array([-4, -3, -7, 0, 0])
A = np.array(
    [
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1],
    ]
)
b = np.array([-1, -3 / 2])
B = np.array([3, 4]) 

result = dual_simplex_method(c, A, b, B)
print(result)
