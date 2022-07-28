from __future__ import annotations

from typing import Hashable

import networkx as nx

from utils import check_cycle, generate_metric_graph

class Union:
    """ class for Disjoint Set (Union Find)
            Applied to Kruskal's Algorithm for MST
    ----------

    Attributes:
    --------
    n: int that represents # of elements in Union
    p: parent array for Union

    Examples
    --------
    >>> u = Union(3)
    >>> u.union(1, 2)
    """
    def __init__(self, n: int):
        self.p = [i for i in range(n)]
        self.n = n
    def find(self, i: int):
        """ recursive find method for Union, finds root
            parent of i
        ----------

        Attributes:
        ----------
        i: the element to find root parent for


        Returns:
        --------
        the root parent of i
        """
        if(self.p[i] != i):
            self.p[i] = self.find(self.p[i])
        return self.p[i]

    def union(self, i: int, j: int):
        """ union method for Union, merges the groups
            i and j belong to if they are in different groups
        ----------

        Attributes:
        ----------
        i, j: targets to merge


        Returns:
        --------
        1 if merged groups, -1 if i and j already in same group
        """
        i = self.find(i)
        j = self.find(j)
        if(i != j):
            self.p[i] = j;
            return 1
        return -1

def MST(graph: nx.Graph):
    """ Compute MST for a nx.Graph
        ----------

        Attributes:
        ----------
        graph: nx.Graph to compute MST for


        Returns:
        --------
        a list of edges that make up the MST
        """
    n = len(list(graph.nodes))
    u = Union(n + 1)
    edges = []
    for e in list(graph.edges):
        edges.append([e[0], e[1], graph[e[0]][e[1]]['weight']])
    edges.sort(key = lambda e : e[2])
    #print(edges)
    res = []
    groups = n
    for e in edges:
        if u.union(e[0], e[1]) == 1:
            res.append(e)
            groups -= 1
        if groups == 1:
            break
    return res

adj = dict()

def hierholzer(u):
    """ computes a Eulerian tour for a graph with
        each vertex having even degree and only 
        one connected component
        ----------

        Attributes:
        ----------
        u: starting element of the eulerian path
        global adj: adjacency list storing the vertices
            and edges of the graph


        Returns:
        --------
        a list of vertices indicating the vertices visited on the
        eulerian tour, without the ending vertex
        (also the starting vertex)
        """
    global adj
    cur = [u]
    eulerian = []
    v = u
    while cur:
        if adj[v]:
            nxt = 0
            for e in adj[v].keys():
                nxt = e
                break
            cur.append(nxt)
            if(adj[v][nxt] == 1):
                del adj[v][nxt]
                del adj[nxt][v]
            else:
                adj[v][nxt] -= 1
                adj[nxt][v] -= 1
            v = nxt
        else:
            eulerian.append(cur[len(cur) - 1])
            v = cur[len(cur) - 1]
            del cur[len(cur) - 1]
    return eulerian


def christofides(graph: nx.Graph) -> list[Hashable]:
    """Approximates a solution to the metric TSP using Christofides' algorithm.
    
    Parameters
    ----------
    graph : nx.Graph
        Weighted graph where weight function is metric.

    Returns
    -------
    cycle : list[Hashable]
        A Hamiltonian cycle with total weight within 3/2 of the minimum.
        The list should begin and end with the same node, e.g. [u, v, w, u] instead of [u, v, w].

    Examples
    --------
    >>> g = generate_metric_graph(6, seed=42)
    >>> cycle = christofides(g)
    >>> check_cycle(g, cycle, optimum=18, approximation_ratio=3/2)
    True
    """
    global adj

    mst = MST(graph)
    adj = dict()
    deg = dict()
    for v in list(graph.nodes):
        adj[v] = dict()
        deg[v] = 0
    for e in mst:
        adj[e[0]][e[1]] = 1
        adj[e[1]][e[0]] = 1
        deg[e[0]] += 1
        deg[e[1]] += 1
    #print(len(mst), len(list(graph.nodes)))
    #print(mst)
    #print(deg)
    odd = set()
    start = 0
    G = nx.Graph()
    for v in list(graph.nodes):
        start = v
        if(deg[v] % 2 == 1):
            odd.add(v)
            G.add_node(v)
    for e in list(graph.edges):
        if(e[0] in odd and e[1] in odd and e[0] != e[1]):
            G.add_edge(*e)
            G[e[0]][e[1]]['weight'] = graph[e[0]][e[1]]['weight']
            G[e[1]][e[0]]['weight'] = graph[e[1]][e[0]]['weight']
    #print(len(odd))
    #print(odd)
    #print(list(G.nodes))
    #print(list(G.edges))
    #print(G[10][13]['weight'])
    min_matching = nx.min_weight_matching(G, maxcardinality = True, weight = 'weight')
    #print(min_matching)
    for e in min_matching:
        if e[1] not in adj[e[0]].keys():
            adj[e[0]][e[1]] = 1
        else:
            adj[e[0]][e[1]] += 1
        if e[0] not in adj[e[1]].keys():
            adj[e[1]][e[0]] = 1
        else:
            adj[e[1]][e[0]] += 1
    #eulerian.append(start)
    eulerian = hierholzer(start)
    visited = set()
    res = []
    for v in eulerian:
        if v not in visited:
            res.append(v)
            visited.add(v)
    res.append(res[0])
    return res


def two_opt(graph: nx.Graph) -> list[Hashable]:
    """Approximates a solution to the metric TSP using the 2-opt algorithm.

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph where weight function is metric.

    Returns
    -------
    cycle : list[Hashable]
        A Hamiltonian cycle with total weight within sqrt(|V|/2) of the minimum.
        The list should begin and end with the same node, e.g. [u, v, w, u] instead of [u, v, w].

    Examples
    --------
    >>> g = generate_metric_graph(6, seed=42)
    >>> cycle = two_opt(g)
    >>> check_cycle(g, cycle, optimum=18, approximation_ratio=3**0.5)
    True
    """
    l = []
    gl = list(graph.nodes)
    for i in range(len(gl)):
        l.append([gl[i], gl[(i + 1) % len(gl)]])
    cost = 0
    for i in range(len(l)):
        cost += graph[l[i][0]][l[i][1]]['weight']
    while(True):
        if(len(l) == 1):
            break
        m = 0
        x = -1
        y = -1
        for i in range(0, len(l) - 2):
            for j in range(i + 2, len(l)):
                s = set()
                s.add(l[i][0])
                s.add(l[i][1])
                s.add(l[j][0])
                s.add(l[j][1])
                if len(s) != 4:
                    continue
                prev = graph[l[i][0]][l[i][1]]['weight'] + graph[l[j][0]][l[j][1]]['weight']
                cur = graph[l[i][0]][l[j][0]]['weight'] + graph[l[i][1]][l[j][1]]['weight']
                if prev - cur > m:
                    m = prev - cur
                    x = i
                    y = j
        xi = x
        yi = y
        # if results can be improved, change ordering of edges
        #   and direction of edges visited.
        if(x != y):
            tp = l[x][1]
            l[x][1] = l[y][0]
            l[y][0] = tp
            x += 1
            y -= 1
            while x < y:
                tp1 = l[x][0]
                tp2 = l[x][1]
                l[x][0] = l[y][0]
                l[x][1] = l[y][1]
                l[y][0] = tp1
                l[y][1] = tp2
                tp1 = l[x][0]
                l[x][0] = l[x][1]
                l[x][1] = tp1
                tp1 = l[y][0]
                l[y][0] = l[y][1]
                l[y][1] = tp1
                y -= 1
                x += 1
            if x == y:
                tp1 = l[x][0]
                l[x][0] = l[x][1]
                l[x][1] = tp1
        tmp = 0
        for i in range(len(l)):
            tmp += graph[l[i][0]][l[i][1]]['weight']
        if(tmp < cost):
            cost = tmp
        else:
            break
    res = []
    for e in l:
        res.append(e[0])
    res.append(res[0])
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
