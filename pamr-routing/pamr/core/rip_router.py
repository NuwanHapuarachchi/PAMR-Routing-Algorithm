class RIPRouter:
    def __init__(self, topology):
        self.topology = topology
        self.routing_table = {}

    def initialize_routing_table(self):
        for node in self.topology:
            self.routing_table[node] = {neighbor: cost for neighbor, cost in self.topology[node].items()}
            self.routing_table[node][node] = 0

    def update_routing_table(self):
        for node in self.topology:
            for neighbor in self.topology[node]:
                for destination in self.routing_table[neighbor]:
                    new_cost = self.routing_table[node][neighbor] + self.routing_table[neighbor][destination]
                    if destination not in self.routing_table[node] or new_cost < self.routing_table[node][destination]:
                        self.routing_table[node][destination] = new_cost

    def find_path(self, source, destination):
        if source not in self.routing_table:
            self.initialize_routing_table()
            for _ in range(len(self.topology) - 1):
                self.update_routing_table()

        path = [source]
        current = source
        while current != destination:
            next_hop = min(self.routing_table[current], key=self.routing_table[current].get)
            path.append(next_hop)
            current = next_hop
        return path, len(path)