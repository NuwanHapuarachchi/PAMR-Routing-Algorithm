class PheromoneManager:
    """Manages pheromone operations for PAMR routing."""
    
    def __init__(self, graph, evaporation_rate=0.15, min_pheromone=0.1, initial_pheromone=0.5):
        self.graph = graph
        self.evaporation_rate = evaporation_rate
        self.min_pheromone = min_pheromone
        self.initial_pheromone = initial_pheromone
        
        # Initialize pheromones to a lower value to make changes more noticeable
        for u, v in self.graph.edges():
            self.graph[u][v]['pheromone'] = initial_pheromone
    
    def update_pheromones(self, paths):
        """
        Update pheromone levels on all edges.
        
        Args:
            paths: List of tuples (path, path_quality) representing successful paths
        """
        self._evaporate_pheromones()
        self._deposit_pheromones(paths)
    
    def _evaporate_pheromones(self):
        """Evaporate pheromone on all edges."""
        for u, v in self.graph.edges():
            # Higher evaporation on congested links
            congestion = self.graph[u][v]['congestion']
            adjusted_rate = self.evaporation_rate * (1 + congestion)
            
            # Evaporate
            self.graph[u][v]['pheromone'] *= (1 - adjusted_rate)
            
            # Enforce minimum pheromone level to ensure exploration
            if self.graph[u][v]['pheromone'] < self.min_pheromone:
                self.graph[u][v]['pheromone'] = self.min_pheromone
    
    def _deposit_pheromones(self, paths):
        """Deposit pheromone on successful paths."""
        for path, path_quality in paths:
            # Only reinforce good paths
            if path_quality > 0:
                # Scale up the pheromone deposit (multiply by 10 to make changes more visible)
                pheromone_deposit = path_quality * 10
                
                # Deposit on each edge in the path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if (u, v) in self.graph.edges():
                        self.graph[u][v]['pheromone'] += pheromone_deposit
