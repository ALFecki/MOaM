import math
import numpy as np


def get_solution_with_branch_and_bound_method(RevA, X, Col):
    n = RevA.shape[0]
    l = np.dot(RevA, X)

    if abs(l[Col]) == 0:
        raise ValueError("Матрица необратима")

    one_divide_li = -1.0 / l[Col]
    l[Col] = -1

    l *= one_divide_li
    res = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == Col:
                res[i][j] = l[i] * RevA[i][j]
            else:
                res[i][j] = RevA[i][j] + l[i] * RevA[Col][j]

    return res


def main_phase_simplex_method(c, A, b, basis, x):
    m, n = A.shape

    AB = A[:, basis]
    A_invB = np.linalg.inv(AB)

    cB = c[basis]

    u = cB @ A_invB

    while True:
        Delta = u @ A - c

        if np.all(Delta >= 0):
            return x, basis

        j0 = np.argmin(Delta)
        Aj0 = A[:, j0]
        z = A_invB @ Aj0
        theta = [x[basis[indx]] / z[indx] if z[indx] > 0 else np.inf for indx in range(m)]

        theta0 = np.min(theta)

        if np.isinf(theta0):
            print("Целевой функционал задачи не ограничен сверху на множестве допустимых планов")
            return

        s = np.argmin(theta)

        for indx in range(m):
            x[basis[indx]] -= theta0 * z[indx]
        x[j0] = theta0
        basis[s] = j0

        A_invB = get_solution_with_branch_and_bound_method(A_invB, A[:, j0], s)

        cB = c[basis]
        u = cB @ A_invB


def first_simplex_phase(c, A, b):
    neg_b_indices = np.where(b < 0)[0]
    if neg_b_indices.size > 0:
        b[neg_b_indices] *= -1
        A[neg_b_indices] *= -1

    m, n = A.shape
    Ae = np.hstack([A, np.eye(m)])
    ec = np.hstack([np.zeros(n), -np.ones(m)])

    B = np.arange(n, n + m)
    xe = np.zeros(n + m)
    xe[B] = np.linalg.solve(Ae[:, B], b)
    xe, B = main_phase_simplex_method(ec, Ae, b, B, xe)

    tmp = xe[n:]

    if not np.all(xe[n:] == 0):
        print("Задача несовместна")
        return
    
    x = xe[:n]

    while True:

        if np.all(B < n):
            break

        k = np.argmax(B)
        jk = B[k]
        i = jk - n

        Ae_B = Ae[:, B]
        Ae_B_inv = np.linalg.inv(Ae_B)
        flg = 0
        for j in set(range(n)) - set(B):
            Aj = Ae[:, j]
            lj = Ae_B_inv @ Aj

            if lj[k] != 0:
                B[k] = j
                flg = 1
                break

        if flg == 0:
            B = np.delete(B, k, axis=0)
            A = np.delete(A, i, axis=0)
            b = np.delete(b, i, axis=0)
            Ae = np.delete(Ae, i, axis=0)
            m -= 1

    return x, B



A = np.array([[3, 2, 1, 0],
             [-3, 2, 0, 1]])
b = np.array([6, 0])
c = np.array([0, 1, 0, 0])

def main(): 
    c_str = ""
    for i, ci in enumerate(c):
        c_str += f"({ci})x{i+1} + "
    c_str = c_str[:-2] + "-> max"
    print(f"c:{c_str}")
    print(f"b:{b}")
    print(f"A:\n{A}\n")

    X, B = first_simplex_phase(c, A, b)
    print(f"Результат симплекс метода:\nx:{X}\nB:{B}")

    xi = -math.inf
    ii = -1
    for i, x in enumerate(X):
        if x != int(x):
            ii = i
            xi = x
            break

    if ii == -1:
        print("Оптимальный план первоначальной задачи")
        exit()

    print(f"Нецелая компонента плана {xi} на позиции {ii}")
    i_star = np.where(B == ii)[0][0]

    N = np.array([i for i in range(len(X)) if i not in B])
    print(f"Множество небазисных индексов: {N}")

    Ab = A[:, B]
    An = A[:, N]

    print(f"Ab:\n{Ab}")
    print(f"An:\n{An}")

    Ab_min = np.linalg.inv(Ab)
    print(f"Ab^-1:\n{Ab_min}")

    Q = np.dot(Ab_min, An)
    print(f"Q = Ab^-1 * N:\n{Q}")

    l = Q[i_star]
    print(f"l = {l}")

    res = [0] * len(X)
    il = 0
    for n in N:
        res[n] = l[il]
        il += 1

    res.append(-1)
    res = np.array(res)

    free_member = xi - int(xi)

    print("Ограничение Гомори имеет вид:")

    res_str = ""
    for i, ri in enumerate(res):
        res_str += f"({ri})x{i+1} + "
    res_str = res_str[:-2]

    print(f"{res_str} = {free_member}")


if __name__ == "__main__":
    main()