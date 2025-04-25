import numpy as np
import random
from ..core.pheromone import PheromoneManager

class PAMRSimulator:
    """Main simulation engine for PAMR protocol."""
    
    def __init__(self, network, router):
        self.network = network
        self.graph = network.graph
        self.router = router
        self.pheromone_manager = PheromoneManager(self.graph)
        self.metrics = {
            'path_lengths': [],
            'congestion_levels': [],
            'convergence_iterations': []
        }
    
    def update_congestion(self):
        """Update congestion levels based on traffic."""
        for u, v in self.graph.edges():
            capacity = self.graph[u][v]['capacity']
            traffic = self.graph[u][v]['traffic']
            # Calculate congestion as traffic/capacity ratio
            congestion = min(1.0, traffic / capacity)
            self.graph[u][v]['congestion'] = congestion
            
            # Add some randomness to traffic to simulate dynamic network conditions
            traffic_change = random.uniform(-0.1, 0.1) * capacity
            self.graph[u][v]['traffic'] = max(0, traffic + traffic_change)
    
    def route_packets(self, num_packets=50, random_src_dest=True, src=None, dest=None):
        """Route multiple packets through the network."""
        successful_paths = []
        
        for _ in range(num_packets):
            if random_src_dest:
                src = random.randint(0, self.network.num_nodes - 1)
                dest = random.randint(0, self.network.num_nodes - 1)
                while src == dest:
                    dest = random.randint(0, self.network.num_nodes - 1)
            
            path, quality = self.router.find_path(src, dest)
            if quality > 0:
                successful_paths.append((path, quality))
        
        return successful_paths
    
    def run_simulation(self, num_iterations=50, packets_per_iter=20):
        """Run the full simulation for specified iterations."""
        path_history = []
        
        for iteration in range(num_iterations):
            # Route packets
            successful_paths = self.route_packets(num_packets=packets_per_iter)
            path_history.append(successful_paths)
            
            # Update congestion levels
            self.update_congestion()
            
            # Update pheromones based on successful paths
            self.pheromone_manager.update_pheromones(successful_paths)
            
            # Collect metrics
            if successful_paths:
                avg_path_length = np.mean([len(path) for path, _ in successful_paths])
                self.metrics['path_lengths'].append(avg_path_length)
                
                congestion_levels = [self.graph[u][v]['congestion'] for u, v in self.graph.edges()]
                self.metrics['congestion_levels'].append(np.mean(congestion_levels))
        
        return path_history
