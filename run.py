# Search methods y comparación de estrategias sobre el grafo de Rumanía

import time
import search


def imprimir_ruta_y_coste(nodo, titulo):
    """Imprime la ruta (lista de nodos) y el coste total de un nodo solución."""
    if nodo is None:
        print(f"{titulo}: SIN SOLUCIÓN")
        return

    ruta_nodos = nodo.path()              # [objetivo, ..., inicial]
    ruta_estados = list(reversed([n.state for n in ruta_nodos]))
    print(f"{titulo}: {ruta_estados} | coste = {nodo.path_cost}")


def ejecutar_estrategia(nombre, funcion, problema, **kwargs):
    """Ejecuta una estrategia de búsqueda y devuelve un diccionario con resultados."""
    t0 = time.time()
    nodo_sol = funcion(problema, **kwargs) if kwargs else funcion(problema)
    t1 = time.time()

    metrics = search.get_last_search_metrics()

    if nodo_sol is None:
        ruta_estados = "SIN SOLUCIÓN"
        coste = None
    else:
        ruta_estados = " -> ".join(n.state for n in reversed(nodo_sol.path()))
        coste = nodo_sol.path_cost

    return {
        "estrategia": nombre,
        "ruta": ruta_estados,
        "coste": coste,
        "nodos_generados": metrics["nodes_generated"],
        "nodos_visitados": metrics["nodes_visited"],
        "tiempo": t1 - t0,
        "nodo": nodo_sol,
    }


def imprimir_tabla(resultados):
    """Imprime los resultados de varias estrategias en una tabla sencilla."""
    print("\nTabla comparativa de estrategias\n")
    print("{:<32} {:>6} {:>6} {:>8} {:>10}   {}".format(
        "Estrategia", "Gen", "Vis", "Coste", "Tiempo(s)", "Ruta"))
    print("-" * 100)
    for r in resultados:
        coste_str = "-" if r["coste"] is None else str(r["coste"])
        print("{:<32} {:>6} {:>6} {:>8} {:>10.6f}   {}".format(
            r["estrategia"],
            r["nodos_generados"],
            r["nodos_visitados"],
            coste_str,
            r["tiempo"],
            r["ruta"],
        ))


if __name__ == "__main__":
    # Problema de ejemplo: A -> B en el grafo de Rumanía
    ab = search.GPSProblem('A', 'B', search.romania)

    # -------------------------------------------------------------------------
    # Comportamiento original del run.py (BFS y DFS) para que no cambie nada
    # -------------------------------------------------------------------------

    print("=== Búsqueda en anchura (BFS) ===")
    bfs_node = search.breadth_first_graph_search(ab)
    imprimir_ruta_y_coste(bfs_node, "BFS")

    print("\n=== Búsqueda en profundidad (DFS) ===")
    dfs_node = search.depth_first_graph_search(ab)
    imprimir_ruta_y_coste(dfs_node, "DFS")

    # Comentario original del enunciado:
    # Result:
    # [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
    # [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450

    # -------------------------------------------------------------------------
    # Nuevas estrategias: Ramificación y Acotación
    # -------------------------------------------------------------------------

    print("\n=== Ramificación y Acotación (sin heurística) ===")
    bb_node = search.branch_and_bound_search(ab, debug=True, debug_max_iterations=5)
    imprimir_ruta_y_coste(bb_node, "Branch & Bound")

    print("\n=== Ramificación y Acotación con subestimación (heurística euclídea) ===")
    bb_h_node = search.branch_and_bound_subestimation_search(
        ab, debug=True, debug_max_iterations=5
    )
    imprimir_ruta_y_coste(bb_h_node, "Branch & Bound + h")

    # -------------------------------------------------------------------------
    # Tabla comparativa usando todas las estrategias
    # -------------------------------------------------------------------------

    print("\n\n=== Comparación global de estrategias para A -> B ===")

    resultados = []

    # BFS
    ab = search.GPSProblem('A', 'B', search.romania)
    resultados.append(ejecutar_estrategia(
        "Búsqueda en anchura (BFS)",
        search.breadth_first_graph_search,
        ab
    ))

    # DFS
    ab = search.GPSProblem('A', 'B', search.romania)
    resultados.append(ejecutar_estrategia(
        "Búsqueda en profundidad (DFS)",
        search.depth_first_graph_search,
        ab
    ))

    # Branch & Bound
    ab = search.GPSProblem('A', 'B', search.romania)
    resultados.append(ejecutar_estrategia(
        "Ramificación y Acotación",
        search.branch_and_bound_search,
        ab,
        debug=False,
        debug_max_iterations=5
    ))

    # Branch & Bound + heurística
    ab = search.GPSProblem('A', 'B', search.romania)
    resultados.append(ejecutar_estrategia(
        "Ramificación y Acotación + heurística",
        search.branch_and_bound_subestimation_search,
        ab,
        debug=False,
        debug_max_iterations=5
    ))

    imprimir_tabla(resultados)
