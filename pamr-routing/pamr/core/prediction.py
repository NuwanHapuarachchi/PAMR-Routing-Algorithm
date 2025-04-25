import random
class SimplePredictor:
    """Simple congestion prediction for PAMR routing."""
    
    def __init__(self, graph, look_ahead=5):
        self.graph = graph
        self.look_ahead = look_ahead
    
    def predict_congestion(self):
        """Predict future congestion based on current traffic trends."""
        future_congestion = {}
        
        for u, v in self.graph.edges():
            current = self.graph[u][v]['congestion']
            # Simulate traffic growth trend with some randomness
            growth_rate = random.uniform(0.05, 0.2) if current < 0.5 else random.uniform(-0.1, 0.1)
            projected = min(1.0, current * (1 + growth_rate * self.look_ahead))
            future_congestion[(u, v)] = projected
            
        return future_congestion
