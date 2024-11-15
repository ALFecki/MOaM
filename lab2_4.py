def knapsack(volumes, values, capacity):
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for v in range(capacity + 1):
            if volumes[i - 1] <= v:
                dp[i][v] = max(dp[i - 1][v], dp[i - 1][v - volumes[i - 1]] + values[i - 1])
            else:
                dp[i][v] = dp[i - 1][v]
    
    v = capacity
    selected_items = []
    for i in range(n, 0, -1):
        if dp[i][v] != dp[i - 1][v]:
            selected_items.append(i - 1) 
            v -= volumes[i - 1]

    return dp[n][capacity], selected_items

volumes = [10, 20, 30, 40]  
values = [60, 100, 120, 240] 
capacity = 50                

max_value, items_selected = knapsack(volumes, values, capacity)

print("Максимальная ценность:", max_value)
print("Выбранные предметы (индексы):", items_selected)