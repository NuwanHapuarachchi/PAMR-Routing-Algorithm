import random
import numpy as np
import networkx as nx

class PAMRRouter:
    """Path selection and routing logic for PAMR protocol."""
    
    def __init__(self, graph, alpha=2.0, beta=3.0, gamma=2.5):
        self.graph = graph
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Distance importance
        self.gamma = gamma  # Congestion importance
        self.use_global_path = True  # Enable global path consideration
    
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
                
                # Similar to OSPF, give congestion a higher weight
                congestion_factor = 1 + congestion * 5
                
                # Include pheromone in inverse proportion (higher pheromone = lower weight)
                pheromone_factor = 1 / (pheromone + 0.1)  # Add 0.1 to avoid division by zero
                
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
            
            # Calculate desirability
            pheromone_factor = pheromone ** self.alpha
            distance_factor = (1.0 / distance) ** self.beta
            congestion_factor = (1.0 - congestion) ** self.gamma
            
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

        # Select neighbor based on probability
        selected_idx = np.argmax(probabilities)  # Choose highest probability always
        return valid_neighbors[selected_idx]
    
    def _calculate_path_quality(self, path):
        """Calculate the quality of a path."""
        total_distance = 0
        max_congestion = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_distance += self.graph[u][v]['distance']
            max_congestion = max(max_congestion, self.graph[u][v]['congestion'])
        
        # Path quality metric - lower distance and congestion = higher quality
        path_quality = 1.0 / (total_distance * (1 + max_congestion))
        return path_quality
