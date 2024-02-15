import numpy as np

def optimize_plan(A, c, B):
    m = len(B)
    
    # Шаг 1: Вычисление обратной матрицы
    AB = A[:, B]
    A_inv_B = np.linalg.inv(AB)
    
    while True:
        # Шаг 2: Формирование вектора cB
        c_B = c[B]
        
        # Шаг 3: Вычисление вектора потенциалов
        u = np.dot(c_B, A_inv_B)
        
        # Шаг 4: Вычисление вектора оценок
        delta = np.dot(u, A) - c
        
        # Шаг 5: Проверка условия оптимальности
        if np.all(delta >= 0):
            # Текущий план является оптимальным
            return delta, B
        
        # Шаг 6: Поиск первой отрицательной компоненты
        j0 = np.where(delta < 0)[0][0]
        
        # Шаг 7: Вычисление вектора z
        Aj0 = A[:, j0]
        z = np.dot(A_inv_B, Aj0)
        
        # Шаг 8: Вычисление вектора theta
        theta = np.zeros(m)
        for i in range(m):
            if z[i] > 0:
                theta[i] = delta[B[i]] / z[i]
            else:
                theta[i] = np.inf
        
        # Шаг 9: Вычисление theta0
        theta0 = np.min(theta)
        
        # Шаг 10: Проверка условия неограниченности
        if np.isinf(theta0):
            # Целевая функция неограничена
            return None
        
        # Шаг 11: Нахождение базисного индекса j*
        j_star = np.argmin(theta)
        
        # Шаг 12: Замена базисного индекса
        B[j_star] = j0
        
        # Шаг 13: Обновление компонентов плана x
        x = np.zeros(len(c))
        x[B] = theta0
        for i in range(m):
            if i != j_star:
                x[B[i]] = x[B[i]] - theta0 * z[i]
        
        # Обновление A_inv_B для следующей итерации
        A_inv_B = np.linalg.inv(A[:, B])
        print(B)


A = np.array([[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]])
c = np.array([1, 1, 0, 0 ,0])
B = [3, 4, 5]

result = optimize_plan(A, c, B)
if result is not None:
    delta, B = result
    print("Оптимальный план найден:")
    print("delta =", delta)
    print("B =", B)
else:
    print("Целевая функция неограничена.")