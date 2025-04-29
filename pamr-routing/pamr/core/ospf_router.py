class OSPFRouter:
    def __init__(self, topology):
        self.topology = topology
        self.routing_table = {}

    def calculate_shortest_paths(self):
        # Implement Dijkstra's algorithm for OSPF
        for source in self.topology:
            unvisited = {node: float('inf') for node in self.topology}
            unvisited[source] = 0
            visited = {}
            path = {}

            while unvisited:
                min_node = min(unvisited, key=unvisited.get)
                for neighbor, cost in self.topology[min_node].items():
                    if neighbor not in visited:
                        new_cost = unvisited[min_node] + cost
                        if new_cost < unvisited[neighbor]:
                            unvisited[neighbor] = new_cost
                            path[neighbor] = min_node
                visited[min_node] = unvisited[min_node]
                unvisited.pop(min_node)

            self.routing_table[source] = {node: self._reconstruct_path(path, source, node) for node in visited}

    def _reconstruct_path(self, path, start, end):
        if end not in path:
            return []
        current = end
        route = []
        while current != start:
            route.insert(0, current)
            current = path[current]
        route.insert(0, start)
        return route

    def find_path(self, source, destination):
        if source not in self.routing_table:
            self.calculate_shortest_paths()
        return self.routing_table[source].get(destination, []), len(self.routing_table[source].get(destination, []))