from collections import deque

class BipartiteGraph:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.graph = [[] for _ in range(n)]
        self.pair_u = [-1] * n
        self.pair_v = [-1] * m
        self.dist = [-1] * n

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def bfs(self):
        queue = deque()
        for u in range(self.n):
            if self.pair_u[u] == -1:
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = float('inf')

        found_augmenting_path = False
        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if self.pair_v[v] == -1:
                    found_augmenting_path = True
                elif self.dist[self.pair_v[v]] == float('inf'):
                    self.dist[self.pair_v[v]] = self.dist[u] + 1
                    queue.append(self.pair_v[v])

        return found_augmenting_path

    def dfs(self, u):
        for v in self.graph[u]:
            if self.pair_v[v] == -1 or (self.dist[self.pair_v[v]] == self.dist[u] + 1 and self.dfs(self.pair_v[v])):
                self.pair_u[u] = v
                self.pair_v[v] = u
                return True
        self.dist[u] = float('inf')
        return False

    def hopcroft_karp(self):
        matching = 0
        while self.bfs():
            for u in range(self.n):
                if self.pair_u[u] == -1 and self.dfs(u):
                    matching += 1
        return matching

    def get_matching_edges(self):
        edges = []
        for u in range(self.n):
            if self.pair_u[u] != -1:
                edges.append((u, self.pair_u[u]))
        return edges


if __name__ == "__main__":
    
    n = 3
    m = 3 
    bg = BipartiteGraph(n, m)

    bg.add_edge(0, 0)
    bg.add_edge(1, 0)
    bg.add_edge(1, 1)
    bg.add_edge(2, 2)
    bg.add_edge(2, 1)
    bg.add_edge(2, 0)

    max_matching = bg.hopcroft_karp()
    print("Максимальное паросочетание:", max_matching)

    matching_edges = bg.get_matching_edges()
    print("Рёбра максимального паросочетания:", matching_edges)
