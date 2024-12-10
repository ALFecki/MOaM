import numpy as np
import networkx as nx

def find_maximum_graph_matching(
        V1: list[str],
        V2: list[str],
        edges: list[tuple[str, str]]
) -> tuple[set[tuple[str, str]], nx.DiGraph]:
    G = nx.DiGraph()
    G.add_nodes_from(V1, bipartite=0)
    G.add_nodes_from(V2, bipartite=1)
    G.add_edges_from(edges)

    s, t = 's', 't'
    G.add_nodes_from([s, t])
    for u in V1:
        G.add_edge(s, u)
    for v in V2:
        G.add_edge(v, t)

    while True:
        try:
            path = nx.shortest_path(G, s, t)
            path_edges = list(zip(path, path[1:]))
        except nx.NetworkXNoPath:
            break

        for edge in path_edges:
            G.remove_edge(*edge)
            if edge[0] != s and edge[1] != t:
                G.add_edge(*edge[::-1])

    M = set()
    for v in V2:
        for u in G.successors(v):
            if u in V1:
                M.add((u, v))

    return M, G


def hungarian_algorithm(C: np.ndarray) -> list[tuple[int]]:
    n, m = C.shape
    assert n == m, "The matrix must be square"

    alpha = np.zeros(n)
    beta = [C[:, j].min() for j in range(m)]

    while True:
        J_eq = [(i, j) for i in range(n) for j in range(m) if alpha[i] + beta[j] == C[i, j]]

        V1 = [f"u{i}" for i in range(n)]
        V2 = [f"v{j}" for j in range(m)]
        edges = [(f"u{i}", f"v{j}") for i, j in J_eq]

        M, G = find_maximum_graph_matching(V1, V2, edges)

        if len(M) == n:
            return [(int(edge[0][1:]), int(edge[1][1:])) for edge in M]

        I_star = set([i for i in range(n) if nx.has_path(G, 's', f"u{i}")])
        J_star = set([i for i in range(n) if nx.has_path(G, 's', f"v{i}")])

        alpha_hat = [1 if i in I_star else -1 for i in range(n)]
        beta_hat = [-1 if j in J_star else 1 for j in range(m)]

        theta = min(
            (C[i, j] - alpha[i] - beta[j]) / 2
            for i in range(n) if i in I_star
            for j in range(m) if j not in J_star
        )

        alpha_star = [alpha[i] + theta * alpha_hat[i] for i in range(n)]
        beta_star = [beta[i] + theta * beta_hat[i] for i in range(m)]

        alpha, beta = alpha_star, beta_star


if __name__ == "__main__":
    C = np.array([
        [7, 2, 1, 9, 4],
        [9, 6, 9, 5, 5],
        [3, 8, 3, 1, 8],
        [7, 9, 4, 2, 2],
        [8, 4, 7, 4, 8]
    ])

    positions = hungarian_algorithm(C)
    print(C, "\n", positions)