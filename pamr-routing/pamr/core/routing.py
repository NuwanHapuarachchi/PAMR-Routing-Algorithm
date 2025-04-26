import random
import numpy as np
import networkx as nx

class PAMRRouter:
    """Path selection and routing logic for PAMR protocol."""
    
    def __init__(self, graph, alpha=2.0, beta=3.0, gamma=6.0, adapt_weights=True):
        self.graph = graph
        self.alpha = alpha  # Pheromone importance - reduced to prevent over-commitment to established paths
        self.beta = beta    # Distance importance
        self.gamma = gamma  # Congestion importance - increased for better congestion avoidance
        self.use_global_path = True  # Enable global path consideration
        self.adapt_weights = adapt_weights  # Dynamically adapt weights based on network conditions
        self.path_history = {}  # Track routing history for source-destination pairs
        self.iteration = 0  # Track iterations for adaptive weight adjustment
    
    def find_path(self, source, destination, max_steps=100):
        """Find a path from source to destination using PAMR."""
        if source == destination:
            return [source], 0
        
        # Try to find a globally optimal path first if the option is enabled
        if self.use_global_path:
            global_path = self._find_global_optimal_path(source, destination)
            if global_path:
                path_quality = self._calculate_path_quality(global_path)
                # Update traffic on the path
                for i in range(len(global_path) - 1):
                    u, v = global_path[i], global_path[i+1]
                    self.graph[u][v]['traffic'] += 1
                return global_path, path_quality
            
        # Fall back to the original PAMR algorithm if global path fails or is disabled
        path = [source]
        visited = {source}
        current = source
        step_count = 0
        
        while current != destination and step_count < max_steps:
            next_node = self._select_next_node(current, destination, visited)
            if next_node is None:
                # No path found
                return path, -1
                
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            step_count += 1
            
            # Update traffic on this edge
            if len(path) > 1:
                u, v = path[-2], path[-1]
                self.graph[u][v]['traffic'] += 1
        
        if current == destination:
            # Calculate path quality
            path_quality = self._calculate_path_quality(path)
            return path, path_quality
        else:
            # Path not found within max steps
            return path, -1
    
    def _find_global_optimal_path(self, source, destination):
        """Find a globally optimal path using a modified Dijkstra's algorithm."""
        try:
            # Define edge weight function that considers all PAMR factors
            def edge_weight(u, v, edge_data):
                pheromone = edge_data['pheromone']
                distance = edge_data['distance'] 
                congestion = edge_data['congestion']
                
                # Adaptive congestion factor - exponential penalty for high congestion
                congestion_factor = 1 + (congestion ** 3) * 12  # Stronger exponential penalty
                
                # Include pheromone in inverse proportion (higher pheromone = lower weight)
                pheromone_factor = 1 / (pheromone + 0.1)  # Add 0.1 to avoid division by zero
                
                # Adaptive weighting based on network conditions
                if self.adapt_weights and (source, destination) in self.path_history:
                    # If this source-destination pair has been routed before, check if quality is degrading
                    history = self.path_history[(source, destination)]
                    if len(history) >= 2 and history[-1]['quality'] < history[-2]['quality'] * 0.8:
                        # Quality degrading - increase weight of congestion even more
                        congestion_factor *= 2.0
                
                # Combined weight - lower is better
                return distance * congestion_factor * pheromone_factor
            
            # Use Dijkstra's algorithm with the custom weight function
            path = nx.shortest_path(self.graph, source, destination, weight=edge_weight)
            return path
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None
    
    def _select_next_node(self, current_node, destination, visited):
        """Select next node using PAMR algorithm."""
        neighbors = list(self.graph.successors(current_node))
        if not neighbors:
            return None
        
        # Calculate selection probabilities
        probabilities = []
        valid_neighbors = []
        
        for neighbor in neighbors:
            if neighbor in visited:
                continue
                
            valid_neighbors.append(neighbor)
            
            # Extract edge attributes
            pheromone = self.graph[current_node][neighbor]['pheromone']
            distance = self.graph[current_node][neighbor]['distance']
            congestion = self.graph[current_node][neighbor]['congestion']
            
            # Calculate desirability with more emphasis on avoiding congested links
            pheromone_factor = pheromone ** self.alpha
            distance_factor = (1.0 / distance) ** self.beta
            congestion_factor = (1.0 - congestion) ** self.gamma  # Increased gamma makes this more sensitive
            
            # Combined desirability
            desirability = pheromone_factor * distance_factor * congestion_factor
            probabilities.append(desirability)
        
        # If no valid neighbors, return None
        if not valid_neighbors:
            return None
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            # If all probabilities are 0, choose randomly
            return random.choice(valid_neighbors)
            
        probabilities = [p / total for p in probabilities]

        # Introduce more exploration to avoid getting stuck on same paths
        # Probabilistic selection with higher probability for better paths
        # but significantly more chance to explore alternatives
        if random.random() < 0.75:  # 75% of the time choose highest probability (was 85%)
            selected_idx = np.argmax(probabilities)
        else:
            # 25% of the time do weighted random selection for more exploration
            selected_idx = np.random.choice(range(len(valid_neighbors)), p=probabilities)
            
        return valid_neighbors[selected_idx]
    
    def _calculate_path_quality(self, path):
        """Calculate the quality of a path."""
        total_distance = 0
        max_congestion = 0
        avg_congestion = 0
        congestion_values = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_distance += self.graph[u][v]['distance']
            congestion_values.append(self.graph[u][v]['congestion'])
            max_congestion = max(max_congestion, self.graph[u][v]['congestion'])
        
        # Calculate average congestion as well
        avg_congestion = sum(congestion_values) / len(congestion_values) if congestion_values else 0
        
        # Improved path quality metric - balances distance and congestion impact
        # Congestion impact now scales more gradually
        congestion_impact = 1 + (max_congestion * 0.6 + avg_congestion * 0.4)
        path_quality = 1.0 / (total_distance * congestion_impact)
        
        # Store path history for adaptive routing
        path_key = (path[0], path[-1])
        if path_key not in self.path_history:
            self.path_history[path_key] = []
        
        # Add this result to path history
        self.path_history[path_key].append({
            'iteration': self.iteration,
            'quality': path_quality,
            'congestion': max_congestion,
            'path': path.copy()  # Store the actual path to track changes
        })
        
        # Limit history size
        if len(self.path_history[path_key]) > 5:
            self.path_history[path_key].pop(0)
            
        return path_quality
        
    def update_iteration(self):
        """Update the iteration counter for the router."""
        self.iteration += 1
