from collections import deque
from copy import deepcopy
from typing import Tuple, Any, Dict

import networkx as nx


def label_method(Gf: nx.DiGraph, start: str):
    Q = deque([start])
    X = set([start])
    l = {start: None}

    while Q:
        u = Q.popleft()
        for v in Gf.neighbors(u):
            if v not in X and Gf[u][v]["capacity"] > 0:
                l[v] = (u, v)
                X.add(v)
                Q.append(v)

    return X, l


def ford_fulkerson_algorithm(G: nx.DiGraph, start: str, end: str) -> tuple[int | Any, dict[Any, int]]:
    edges = deepcopy(G.edges())
    for edge in edges:
        if (edge[1], edge[0]) not in edges:
            G.add_edge(edge[1], edge[0], capacity=0)
    max_flow = 0

    f = {edge: 0 for edge in G.edges()}

    Gf = deepcopy(G)
    for u, v in Gf.edges():
        Gf[u][v]["capacity"] = G[u][v]["capacity"] - f[(u, v)] + f[(v, u)]

    while True:
        X, l = label_method(Gf, start)
        if end not in X:
            break

        path, v = [], end
        while True:
            u, v = l[v]
            path.append((u, v))
            v = u
            if v == start:
                break
        path.reverse()

        theta = min(G[u][v]["capacity"] for u, v in path)

        fP = {edge: (theta if edge in path else 0) for edge in G.edges()}

        for u, v in path:
            f[(u, v)] += fP[(u, v)]
            Gf[u][v]["capacity"] -= theta
            Gf[v][u]["capacity"] += theta
        max_flow += theta

    return max_flow, f


if __name__ == "__main__":
    G = nx.DiGraph()

    G.add_edge('s', 'a', capacity=3)
    G.add_edge('s', 'b', capacity=2)
    G.add_edge('a', 'b', capacity=2)
    G.add_edge('a', 't', capacity=1)
    G.add_edge('b', 't', capacity=2)

    result = ford_fulkerson_algorithm(G, 's', 't')
    max_flow = result[0]
    flow_dict = result[1]

    max_value = max(flow_dict.values())

    for node, flow in flow_dict.items():
        if flow == max_value:
            print(node)
    print(max_value)