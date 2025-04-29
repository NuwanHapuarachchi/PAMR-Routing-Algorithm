import networkx as nx
import random
import numpy as np
import pickle
import os

class NetworkTopology:
    """Network topology management for PAMR routing with dynamic metrics."""
    
    def __init__(self, num_nodes=30, connectivity=0.3, seed=42, variation_factor=0.05):
        """Initialize network topology.
        
        Args:
            num_nodes: Number of nodes in the network
            connectivity: Probability of edge creation between nodes
            seed: Random seed for reproducibility
            variation_factor: Factor controlling the magnitude of dynamic variations (0.05 = 5%)
        """
        self.num_nodes = num_nodes
        self.connectivity = connectivity
        self.seed = seed
        self.variation_factor = variation_factor
        self.iteration = 0
        self.graph = self._create_network()
        self.positions = nx.spring_layout(self.graph, seed=self.seed)
        
    def _create_network(self):
        """Create a random network with given parameters."""
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
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
            # Set consistent initial values based on node indices
            # This ensures reproducibility while still having variety
            edge_seed = u * self.num_nodes + v
            rng = random.Random(edge_seed + self.seed)
            
            # Set random distance/cost (1-10)
            distance = rng.uniform(1, 10)
            # Capacity (bandwidth) in arbitrary units (10-100)
            capacity = rng.uniform(10, 100)
            
            G[u][v]['distance'] = distance
            G[u][v]['pheromone'] = pheromone_init
            G[u][v]['capacity'] = capacity
            G[u][v]['traffic'] = 0.0
            G[u][v]['congestion'] = 0.0
            # Store original values for controlled variations
            G[u][v]['base_distance'] = distance
            G[u][v]['base_capacity'] = capacity
    
    def update_dynamic_metrics(self, traffic_decay=0.7):
        """Update network metrics with small variations to simulate dynamic conditions.
        
        Args:
            traffic_decay: Factor that determines how much traffic decays each iteration (0.7 = 30% decay)
        """
        # Increment iteration counter
        self.iteration += 1
        
        # Set seed based on iteration for controlled randomness
        np.random.seed(self.seed + self.iteration)
        
        for u, v in self.graph.edges():
            # Generate small variations around the base values
            # Using a sine wave pattern with small random noise for smooth variations
            time_factor = np.sin(self.iteration / 10) * self.variation_factor
            noise = np.random.normal(0, self.variation_factor/3)
            
            # Update distance (small variations around base)
            variation = (time_factor + noise) * self.graph[u][v]['base_distance']
            self.graph[u][v]['distance'] = max(1.0, self.graph[u][v]['base_distance'] + variation)
            
            # Update capacity (small variations around base)
            variation = (time_factor + noise) * self.graph[u][v]['base_capacity']
            self.graph[u][v]['capacity'] = max(10.0, self.graph[u][v]['base_capacity'] + variation)
            
            # Apply traffic decay - simulate traffic clearing from the network (30% decay)
            self.graph[u][v]['traffic'] *= traffic_decay
            
            # Simulate traffic changes - small random adjustments 
            traffic_change = np.random.normal(0, self.variation_factor * 8)  # Reduced random traffic
            self.graph[u][v]['traffic'] = max(0.0, min(
                self.graph[u][v]['traffic'] + traffic_change,
                self.graph[u][v]['capacity'] * 0.8  # Cap at 80% of capacity instead of 90%
            ))
            
            # Update congestion based on traffic/capacity ratio
            self.graph[u][v]['congestion'] = min(0.95, self.graph[u][v]['traffic'] / self.graph[u][v]['capacity'])
    
    def save(self, filepath="network_state.pkl"):
        """Save the current network state to a file for reuse across simulations."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_nodes': self.num_nodes,
                'connectivity': self.connectivity,
                'seed': self.seed,
                'variation_factor': self.variation_factor,
                'iteration': self.iteration,
                'graph': self.graph
            }, f)
        return filepath
    
    @classmethod
    def load(cls, filepath="network_state.pkl"):
        """Load a network state from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        network = cls(
            num_nodes=data['num_nodes'],
            connectivity=data['connectivity'],
            seed=data['seed'],
            variation_factor=data['variation_factor']
        )
        network.iteration = data['iteration']
        network.graph = data['graph']
        network.positions = nx.spring_layout(network.graph, seed=network.seed)
        return network
    
    @classmethod
    def get_consistent_network(cls, filepath="consistent_network.pkl", force_new=False, **kwargs):
        """Get a consistent network for use across simulation files.
        
        Args:
            filepath: Path to save/load the network state
            force_new: If True, creates a new network even if file exists
            **kwargs: Parameters for network creation
        
        Returns:
            NetworkTopology instance
        """
        if os.path.exists(filepath) and not force_new:
            return cls.load(filepath)
        else:
            # If file doesn't exist or force_new is True, create new network
            network = cls(**kwargs)
            network.save(filepath)
            return network

# Debugging: Add print statements to confirm function execution
print("Debug: Starting to print metrics for nodes 0 and 1")

# Function to print metrics for specific source and destination nodes
def print_metrics_for_nodes(network, source, destination, iterations=20):
    print(f"Debug: Metrics for source node {source} and destination node {destination}:")
    print("Iteration | Edge (u, v) | Distance | Capacity | Traffic | Congestion")
    print("-" * 70)
    for i in range(iterations):
        print(f"Debug: Iteration {i+1}")  # Debugging iteration
        network.update_dynamic_metrics()
        for u, v in network.graph.edges():
            if u == source and v == destination:
                distance = network.graph[u][v]['distance']
                capacity = network.graph[u][v]['capacity']
                traffic = network.graph[u][v]['traffic']
                congestion = network.graph[u][v]['congestion']
                print(f"{i+1:9} | ({u:2}, {v:2}) | {distance:8.2f} | {capacity:8.2f} | {traffic:7.2f} | {congestion:9.2f}")

# Initialize consistent_network before using it
consistent_network = NetworkTopology.get_consistent_network(
    filepath="consistent_network.pkl",
    force_new=True,  # Use existing network if available
    num_nodes=10,  # Default number of nodes
    connectivity=0.02,  # Default connectivity
    seed=42,  # Default seed for reproducibility
    variation_factor=0.5  # Default variation factor
)

# Example usage for nodes 0 and 1
print_metrics_for_nodes(consistent_network, source=0, destination=1)

# For backward compatibility
network = consistent_network
