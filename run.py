

import time
import search



def _ruta_formato(nodo):
    """Devuelve la ruta como lista de repr(Node) desde objetivo -> ... -> inicial,
    : [<Node B>, <Node P>, ...]
    """
    if nodo is None:
        return "SIN SOLUCIÓN"
    # nodo.path() devuelve [objetivo, ..., inicial]
    return "[" + ", ".join(repr(n) for n in nodo.path()) + "]"


def _ejecutar_y_medir(funcion, problema, **kwargs):
    """Ejecuta una búsqueda, mide tiempo, y devuelve (nodo_sol, metrics, tiempo)."""
    t0 = time.perf_counter()
    nodo_sol = funcion(problema, **kwargs) if kwargs else funcion(problema)
    t1 = time.perf_counter()
    metrics = search.get_last_search_metrics()
    return nodo_sol, metrics, (t1 - t0)


def _imprimir_columna(nombre_columna, nodo_sol, metrics, tiempo):
    """Imprime una 'columna' ."""
    if nodo_sol is None:
        coste = "-"
        ruta = "SIN SOLUCIÓN"
    else:
        coste = str(nodo_sol.path_cost)
        ruta = _ruta_formato(nodo_sol)

    print(f"--- COLUMNA: {nombre_columna} ---")
    print(f"Generados: {metrics['nodes_generated']}")
    print(f"Visitados: {metrics['nodes_visited']}")
    print(f"Costo total: {coste}")
    print(f"Tiempo: {tiempo:.6f} s")
    print(f"Ruta: {ruta}")
    print("")


def ejecutar_caso(case_id, origen, destino):
    """Ejecuta BFS/DFS/BB/A* para un par (origen -> destino) y lo imprime."""
    print("#" * 80)
    print(f"ID {case_id}: {origen} -> {destino}")
    print("#" * 80)
    print("")

    # ---------- BFS ----------
    problema = search.GPSProblem(origen, destino, search.romania)
    nodo, metrics, tiempo = _ejecutar_y_medir(search.breadth_first_graph_search, problema)
    _imprimir_columna("Amplitud", nodo, metrics, tiempo)

    # ---------- DFS ----------
    problema = search.GPSProblem(origen, destino, search.romania)
    nodo, metrics, tiempo = _ejecutar_y_medir(search.depth_first_graph_search, problema)
    _imprimir_columna("Profundidad", nodo, metrics, tiempo)

    # ---------- Branch & Bound ----------
    # return_on_goal=True para el formato típico de tabla (devuelve al hallar solución óptima)
    problema = search.GPSProblem(origen, destino, search.romania)
    nodo, metrics, tiempo = _ejecutar_y_medir(
        search.branch_and_bound_search,
        problema,
        debug=False,
        debug_max_iterations=5,
        return_on_goal=True
    )
    _imprimir_columna("Ramificación y Acotación", nodo, metrics, tiempo)

    # ---------- A* (Ramif. con Subestimación) ----------
    problema = search.GPSProblem(origen, destino, search.romania)
    nodo, metrics, tiempo = _ejecutar_y_medir(
        search.branch_and_bound_subestimation_search,
        problema,
        debug=False,
        debug_max_iterations=5,
        return_on_goal=True
    )
    _imprimir_columna("Ramif. con Subestimación (A*)", nodo, metrics, tiempo)



if __name__ == "__main__":
    print("=" * 80)
    print("PARTE OBLIGATORIA: DATOS PARA LA TABLA")
    print("=" * 80)
    print("")

    casos = [
        (1, "A", "B"),   # Arad -> Bucharest 
        (2, "O", "E"),   # Oradea -> Eforie
        (3, "G", "Z"),   # Giurgiu -> Zerind
        (4, "N", "D"),   # Neamt -> Dobreta
        (5, "M", "F"),   # Mehadia -> Fagaras
    ]

    # IMPORTANTE:
    # Nodos del grafo de Rumanía son letras:
    # A,B,C,D,E,F,G,H,I,L,M,N,O,P,R,S,T,U,V,Z
    # Si tu enunciado usa nombres completos, aquí debes mapearlos a letras.

    for (cid, ori, dst) in casos:
        ejecutar_caso(cid, ori, dst)

    print("=" * 80)
    print("FIN DE LA EJECUCIÓN GLOBAL")
    print("=" * 80)
