import numpy as np


def optimize_plan(A, c, B, x):
    m = len(B)

    AB = A[:, np.array(B) - 1]  # Modify this line
    print(AB)
    A_inv_B = np.linalg.inv(AB)
    print(A_inv_B)

    while True:
        print(A_inv_B)

        c_B = c[np.array(B) - 1]
        print(c_B)
        u = np.dot(c_B, A_inv_B)
        print(u)
        delta = np.dot(u, A) - c
        print(delta)
        if np.all(delta >= 0):
            return delta, B, x

        j0 = np.where(delta < 0)[0][0] + 1

        Aj0 = A[:, j0 - 1]
        z = np.dot(A_inv_B, Aj0)

        theta = np.zeros(m)
        for i in range(m):
            if z[i] > 0:
                theta[i] = x[B[i] - 1] / z[i]
            else:
                theta[i] = np.inf

        theta0 = np.min(theta)

        if np.isinf(theta0):
            return None
        min_ind = np.argmin(theta)
        j_star = B[min_ind]

        B[min_ind] = j0

        # x = np.zeros(len(c))
        x[j0 - 1] = theta0
        for i in range(m):
            if i != min_ind:
                x[B[i] - 1] = x[B[i] - 1] - theta0 * z[i]
        x[j_star - 1] = 0

        A_inv_B = np.linalg.inv(A[:, np.array(B) - 1])
        print(B)


A = np.array([[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]])
c = np.array([1, 1, 0, 0, 0])
B = [3, 4, 5]
x = [0, 0, 1, 3, 2]

result = optimize_plan(A, c, B, x)
if result is not None:
    delta, B, x = result
    print("Оптимальный план найден:")
    print("delta =", delta)
    print("B =", B)
    print("x = ", x)
else:
    print("Целевая функция неограничена.")
