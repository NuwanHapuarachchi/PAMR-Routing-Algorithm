import numpy as np
import random
import time
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
            'convergence_iterations': [],
            'convergence_times': [],
            'path_qualities': []
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
    
    def run_simulation(self, num_iterations=100, packets_per_iter=10):
        """Run the PAMR routing simulation with dynamic network metrics."""
        path_history = []
        
        for i in range(num_iterations):
            # Update network metrics to simulate dynamic conditions
            self.network.update_dynamic_metrics()
            
            iteration_paths = []
            start_time = time.time()
            
            # Route packets between random nodes
            total_path_length = 0
            total_path_quality = 0
            max_congestion = 0
            
            for _ in range(packets_per_iter):
                # Pick random source and destination
                source, destination = random.sample(list(self.network.graph.nodes()), 2)
                
                # Find path
                path, quality = self.router.find_path(source, destination)
                
                if path and len(path) > 1:
                    # Update edge traffic and congestion based on path
                    self._update_network_congestion(path)
                    
                    # Store path information
                    iteration_paths.append((source, destination, path, quality))
                    
                    # Update metrics
                    total_path_length += len(path) - 1
                    total_path_quality += quality
            
            # Store metrics for this iteration
            convergence_time = time.time() - start_time
            self.metrics['convergence_times'].append(convergence_time)
            self.metrics['path_lengths'].append(total_path_length / packets_per_iter if packets_per_iter > 0 else 0)
            self.metrics['path_qualities'].append(total_path_quality / packets_per_iter if packets_per_iter > 0 else 0)
            self.metrics['congestion_levels'].append(max_congestion)
            
            # Add this iteration's paths to history
            path_history.append(iteration_paths)
        
        # Return collected path history for analysis
        return path_history
    
    def _update_network_congestion(self, path):
        """Update network congestion based on a routed path."""
        if not path or len(path) < 2:
            return
            
        # Update traffic and recalculate congestion for each link in the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Get current traffic and capacity values
            current_traffic = self.network.graph[u][v].get('traffic', 0)
            capacity = self.network.graph[u][v].get('capacity', 10)
            
            # Increment traffic (standard packet size of 1.0)
            self.network.graph[u][v]['traffic'] = current_traffic + 1.0
            
            # Calculate congestion as traffic / capacity ratio (without arbitrary cap)
            self.network.graph[u][v]['congestion'] = self.network.graph[u][v]['traffic'] / capacity
