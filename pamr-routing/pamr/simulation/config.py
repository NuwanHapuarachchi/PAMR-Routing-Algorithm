DEFAULT_CONFIG = {
    'num_nodes': 15,
    'connectivity': 0.3,
    'pheromone_init': 1.0,
    'evaporation_rate': 0.1,
    'alpha': 1.0,  # Pheromone importance
    'beta': 2.0,   # Distance importance
    'gamma': 1.0,  # Congestion importance
    'simulation_iterations': 50,
    'packets_per_iteration': 30,
    'prediction_lookahead': 5
}

class PAMRConfig:
    """Configuration management for PAMR simulations."""
    
    def __init__(self, config_dict=None):
        self.config = DEFAULT_CONFIG.copy()
        if config_dict:
            self.config.update(config_dict)
    
    def get(self, key):
        """Get configuration value."""
        return self.config.get(key)
    
    def set(self, key, value):
        """Set configuration value."""
        self.config[key] = value
        
    def to_dict(self):
        """Convert to dictionary."""
        return self.config.copy()
