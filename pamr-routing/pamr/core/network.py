import networkx as nx
import random

class NetworkTopology:
    """Network topology management for PAMR routing."""
    
    def __init__(self, num_nodes=15, connectivity=0.3, seed=42):
        self.num_nodes = num_nodes
        self.connectivity = connectivity
        self.seed = seed
        self.graph = self._create_network()
        self.positions = nx.spring_layout(self.graph, seed=self.seed)
    
    def _create_network(self):
        """Create a random network with given parameters."""
        # Use Erdos-Renyi random graph model
        G = nx.erdos_renyi_graph(self.num_nodes, self.connectivity, directed=True, seed=self.seed)
        
        # Ensure graph is connected
        if not nx.is_strongly_connected(G):
            self._ensure_connectivity(G)
        
        # Initialize edge attributes
        self._initialize_edge_attributes(G)
        
        return G
    
    def _ensure_connectivity(self, G):
        """Ensure the graph is strongly connected."""
        # Find largest strongly connected component
        components = list(nx.strongly_connected_components(G))
        for i in range(1, len(components)):
            u = random.choice(list(components[i-1]))
            v = random.choice(list(components[i]))
            G.add_edge(u, v)
            G.add_edge(v, u)  # Adding bidirectional edge
    
    def _initialize_edge_attributes(self, G, pheromone_init=1.0):
        """Initialize edge attributes for the network."""
        for u, v in G.edges():
            # Set random distance/cost
            distance = random.uniform(1, 10)
            # Capacity (bandwidth) in arbitrary units
            capacity = random.uniform(10, 100)
            
            G[u][v]['distance'] = distance
            G[u][v]['pheromone'] = pheromone_init
            G[u][v]['capacity'] = capacity
            G[u][v]['traffic'] = 0
            G[u][v]['congestion'] = 0.0
