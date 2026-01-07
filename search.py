"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.

Extensión (Práctica):
- Añadida Ramificación y Acotación (Branch & Bound) y B&B con Subestimación (A*).
- Añadida contabilidad: nodos generados y nodos visitados.
- Mejorada compatibilidad con IDE (Pylance): abstract() en lugar de "abstract".
- Mantiene el código original y lo amplía (no se elimina funcionalidad).
"""

from utils import *
import random
import sys
import heapq  # Necesario para la cola de prioridad


# ______________________________________________________________________________
# Definición de problemas y nodos


class Problem:
    """The abstract class for a formal problem.

    You should subclass this and implement the method successor, and possibly
    __init__, goal_test, and path_cost.
    """

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal state."""
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """Given a state, return a sequence of (action, state) pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework.
        """
        abstract()  

    def goal_test(self, state):
        """Return True if the state is a goal."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from state1."""
        return c + 1

    def value(self):
        """For optimization problems, each state has a value."""
        abstract()  


class Node:
    """A node in a search tree."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def path(self):
        """Create a list of nodes from the root to this node."""
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        return result

    def expand(self, problem):
        """Return a list of nodes reachable from this node. [Fig. 3.8]"""
        return [Node(next, self, act,
                     problem.path_cost(self.path_cost, self.state, act, next))
                for (act, next) in problem.successor(self.state)]


# ______________________________________________________________________________
# Soporte adicional: PriorityQueue + contabilidad de nodos (Parte 1 y 3)


class PriorityQueue(Queue):
    """Cola de prioridad sencilla.

    Los elementos se extraen en orden creciente del valor devuelto por `key`.
    Hereda de Queue para respetar la interfaz general usada en el resto
    del código (append, pop, extend, __len__).
    """

    def __init__(self, key=lambda x: x):
        self.key = key
        self._heap = []
        self._counter = 0  # Para romper empates de forma estable

    def append(self, item):
        """Inserta un elemento en la cola de prioridad."""
        heapq.heappush(self._heap, (self.key(item), self._counter, item))
        self._counter += 1

    def extend(self, items):
        """Inserta múltiples elementos en la cola de prioridad."""
        for item in items:
            self.append(item)

    def pop(self):
        """Extrae el elemento con menor prioridad."""
        return heapq.heappop(self._heap)[2]

    def __len__(self):
        return len(self._heap)


# Métricas globales de la última búsqueda ejecutada
_last_search_metrics = {
    "algorithm": None,
    "nodes_generated": 0,
    "nodes_visited": 0,
}


def reset_search_metrics(algorithm_name):
    """Reinicia los contadores para una nueva ejecución de búsqueda."""
    global _last_search_metrics
    _last_search_metrics = {
        "algorithm": algorithm_name,
        "nodes_generated": 0,
        "nodes_visited": 0,
    }


def _count_generated(n):
    """Suma n al número de nodos generados."""
    _last_search_metrics["nodes_generated"] += n


def _count_visited():
    """Incrementa en uno el número de nodos visitados (extraídos de la frontera)."""
    _last_search_metrics["nodes_visited"] += 1


def get_last_search_metrics():
    """Devuelve una copia de las métricas de la última búsqueda."""
    return dict(_last_search_metrics)


# ______________________________________________________________________________
# Uninformed Search algorithms (modificados para incluir contabilidad)


def graph_search(problem, fringe):
    """Search through the successors of a problem to find a goal.

    The argument fringe should be an empty queue.
    If two paths reach a state, only use the best one. [Fig. 3.18]

    Versión adaptada para contar:
      - nodos generados,
      - nodos visitados.

    NOTA: Esta función no llama a `reset_search_metrics` por sí sola; se
    espera que lo hagan las funciones concretas (BFS, DFS, etc.).
    """
    closed = {}

    # Nodo inicial
    root = Node(problem.initial)
    fringe.append(root)
    _count_generated(1)

    while fringe:
        node = fringe.pop()
        _count_visited()

        if problem.goal_test(node.state):
            return node

        if node.state not in closed:
            closed[node.state] = True
            children = node.expand(problem)
            _count_generated(len(children))
            fringe.extend(children)
    return None


def breadth_first_graph_search(problem):
    """Search the shallowest nodes in the search tree first. [p 74]"""
    reset_search_metrics("breadth_first_graph_search")
    return graph_search(problem, FIFOQueue())  # FIFOQueue -> fringe


def depth_first_graph_search(problem):
    """Search the deepest nodes in the search tree first. [p 74]"""
    reset_search_metrics("depth_first_graph_search")
    return graph_search(problem, Stack())


# ______________________________________________________________________________
# Ramificación y Acotación (Branch & Bound) - Parte 1
#
# NOTA IMPORTANTE (TABLA):
# - Para el formato esperado en la tabla, normalmente se "devuelve" cuando se
#   encuentra la primera solución óptima que sale de la frontera por prioridad.
# - Para mantener el comportamiento original (seguir buscando mejoras),
#   existe el parámetro `return_on_goal`.
#
# return_on_goal=True  -> comportamiento ideal para tabla (para al encontrar objetivo prometedor)
# return_on_goal=False -> comportamiento original (sigue explorando por si hay mejor solución)


def branch_and_bound_search(problem, debug=False, debug_max_iterations=5, return_on_goal=True):
    """Estrategia de búsqueda de Ramificación y Acotación SIN heurística.

    Características:
        - La frontera es una PriorityQueue ordenada por el coste acumulado
          g(n) = node.path_cost.
        - Se mantiene la mejor solución actual (mejor coste) y se podan
          aquellos nodos cuyo coste ya es peor o igual.
        - Se guarda el mejor coste alcanzado para cada estado para evitar
          re-explorar caminos claramente peores.

    Parámetros:
        problem: instancia de Problem (p.ej. GPSProblem sobre el grafo de Rumanía)
        debug (bool): si es True, se muestran por pantalla las primeras iteraciones.
        debug_max_iterations (int): número máximo de iteraciones a imprimir.
        return_on_goal (bool):
            - True: devuelve al encontrar una solución (modo tabla / práctica).
            - False: sigue explorando para garantizar "mejor de todas" aunque cueste más.
    """
    reset_search_metrics("branch_and_bound_search")

    # Cola de prioridad ordenada por el coste acumulado g(n)
    fringe = PriorityQueue(key=lambda node: node.path_cost)
    best_solution = None
    best_cost = infinity
    closed = {}  # mejor coste conocido para cada estado

    root = Node(problem.initial)
    fringe.append(root)
    _count_generated(1)

    iteration = 0

    while fringe:
        node = fringe.pop()
        _count_visited()
        iteration += 1

        if debug and iteration <= debug_max_iterations:
            print(f"Iteración {iteration}: expandiendo {node} con g={node.path_cost}")

        # Cota: si ya tenemos una solución mejor, este nodo no puede mejorarla
        if node.path_cost >= best_cost:
            if debug and iteration <= debug_max_iterations:
                print(f"  -> Podado (g={node.path_cost} >= mejor_coste={best_cost})")
            continue

        # ¿Es objetivo?
        if problem.goal_test(node.state):
            if node.path_cost < best_cost:
                best_solution = node
                best_cost = node.path_cost
                if debug and iteration <= debug_max_iterations:
                    print(f"  -> NUEVA mejor solución encontrada con coste {best_cost}")

            # MODO TABLA: devolvemos aquí para que cuadren contadores típicos
            if return_on_goal:
                return best_solution

            # MODO ORIGINAL: seguimos por si existe una solución aún mejor
            continue

        # ¿Merece la pena expandir este estado?
        previous_best = closed.get(node.state, infinity)
        if node.path_cost >= previous_best:
            if debug and iteration <= debug_max_iterations:
                print(f"  -> No se expande {node} (existe camino mejor con coste {previous_best})")
            continue

        closed[node.state] = node.path_cost

        # Expandimos sucesores
        children = node.expand(problem)
        _count_generated(len(children))

        if debug and iteration <= debug_max_iterations:
            for child in children:
                print(f"    Generado hijo {child} con g={child.path_cost}")

        for child in children:
            if child.path_cost < best_cost:
                fringe.append(child)

    return best_solution


# ______________________________________________________________________________
# Ramificación y Acotación con Subestimación (heurística euclídea) - Parte 2
#



def branch_and_bound_subestimation_search(problem, debug=False, debug_max_iterations=5, return_on_goal=True):
    """Estrategia de Ramificación y Acotación con Subestimación (tipo A*).

    Prioridad:
        f(n) = g(n) + h(n)

    donde:
        - g(n) = node.path_cost
        - h(n) = heurística (en GPSProblem se calcula como distancia euclídea)

    Parámetros extra:
        return_on_goal (bool):
            - True: devuelve al encontrar solución (modo tabla).
            - False: sigue explorando por si mejora (modo original extendido).
    """
    reset_search_metrics("branch_and_bound_subestimation_search")

    def heuristic(node):
        h_fun = getattr(problem, "h", None)
        if callable(h_fun):
            return h_fun(node)
        return 0

    def f(node):
        return node.path_cost + heuristic(node)

    fringe = PriorityQueue(key=f)
    best_solution = None
    best_cost = infinity
    closed = {}

    root = Node(problem.initial)
    fringe.append(root)
    _count_generated(1)

    iteration = 0

    while fringe:
        node = fringe.pop()
        _count_visited()
        iteration += 1

        if debug and iteration <= debug_max_iterations:
            print(f"Iteración {iteration}: expandiendo {node} con "
                  f"g={node.path_cost}, h={heuristic(node)}, f={f(node)}")

        # Cota basada en el coste real de la mejor solución
        if node.path_cost >= best_cost:
            if debug and iteration <= debug_max_iterations:
                print(f"  -> Podado por coste real (g={node.path_cost} >= mejor_coste={best_cost})")
            continue

        # Cota basada en la subestimación (Branch & Bound con heurística)
        if f(node) >= best_cost:
            if debug and iteration <= debug_max_iterations:
                print(f"  -> Podado por cota heurística (f={f(node)} >= mejor_coste={best_cost})")
            continue

        if problem.goal_test(node.state):
            if node.path_cost < best_cost:
                best_solution = node
                best_cost = node.path_cost
                if debug and iteration <= debug_max_iterations:
                    print(f"  -> NUEVA mejor solución encontrada con coste {best_cost}")

            # MODO TABLA
            if return_on_goal:
                return best_solution

            # MODO ORIGINAL
            continue

        previous_best = closed.get(node.state, infinity)
        if node.path_cost >= previous_best:
            if debug and iteration <= debug_max_iterations:
                print(f"  -> No se expande {node} (existe camino mejor con coste {previous_best})")
            continue

        closed[node.state] = node.path_cost

        children = node.expand(problem)
        _count_generated(len(children))

        if debug and iteration <= debug_max_iterations:
            for child in children:
                print(f"    Generado hijo {child} con g={child.path_cost}, "
                      f"h={heuristic(child)}, f={f(child)}")

        for child in children:
            if f(child) < best_cost:
                fringe.append(child)

    return best_solution


# _____________________________________________________________________________
# The remainder of this file implements examples for the search algorithms.
# ______________________________________________________________________________
# Graphs and Graph Problems


class Graph:
    """A graph connects nodes (vertices) by edges (links)."""

    def __init__(self, dict=None, directed=True):
        self.dict = dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.dict.keys()):
            for (b, distance) in list(self.dict[a].items()):
                self.connect1(b, a, distance)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries."""
        links = self.dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        return list(self.dict.keys())


def UndirectedGraph(dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(dict=dict, directed=False)


def RandomGraph(nodes=list(range(10)), min_links=2, width=400, height=300,
                curvature=lambda: random.uniform(1.1, 1.5)):
    """Construct a random graph, with the specified nodes, and random links."""
    g = UndirectedGraph()
    g.locations = {}
    for node in nodes:
        g.locations[node] = (random.randrange(width), random.randrange(height))
    for i in range(min_links):
        for node in nodes:
            if len(g.get(node)) < min_links:
                here = g.locations[node]

                def distance_to_node(n):
                    if n is node or g.get(node, n):
                        return infinity
                    return distance(g.locations[n], here)

                neighbor = argmin(nodes, distance_to_node)
                d = distance(g.locations[neighbor], here) * curvature()
                g.connect(node, neighbor, int(d))
    return g


romania = UndirectedGraph(Dict(
    A=Dict(Z=75, S=140, T=118),
    B=Dict(U=85, P=101, G=90, F=211),
    C=Dict(D=120, R=146, P=138),
    D=Dict(M=75),
    E=Dict(H=86),
    F=Dict(S=99),
    H=Dict(U=98),
    I=Dict(V=92, N=87),
    L=Dict(T=111, M=70),
    O=Dict(Z=71, S=151),
    P=Dict(R=97),
    R=Dict(S=80),
    U=Dict(V=142)
))
romania.locations = Dict(
    A=(91, 492), B=(400, 327), C=(253, 288), D=(165, 299),
    E=(562, 293), F=(305, 449), G=(375, 270), H=(534, 350),
    I=(473, 506), L=(165, 379), M=(168, 339), N=(406, 537),
    O=(131, 571), P=(320, 368), R=(233, 410), S=(207, 457),
    T=(94, 410), U=(456, 350), V=(509, 444), Z=(108, 531)
)

australia = UndirectedGraph(Dict(
    T=Dict(),
    SA=Dict(WA=1, NT=1, Q=1, NSW=1, V=1),
    NT=Dict(WA=1, Q=1),
    NSW=Dict(Q=1, V=1)
))
australia.locations = Dict(
    WA=(120, 24), NT=(135, 20), SA=(135, 30),
    Q=(145, 20), NSW=(145, 32), T=(145, 42), V=(145, 37)
)


class GPSProblem(Problem):
    """The problem of searching in a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def successor(self, A):
        """Return a list of (action, result) pairs."""
        return [(B, B) for B in list(self.graph.get(A).keys())]

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or infinity)

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity
