import matplotlib.pyplot as plt
import networkx as nx

class NetworkVisualizer:
    """Network visualization for PAMR protocol."""
    
    def __init__(self, network):
        self.network = network
        self.graph = network.graph
        self.pos = network.positions
    
    def visualize_network(self, source=None, destination=None, paths=None):
        """Visualize the network with path and pheromone levels."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Prepare node colors
        node_colors = ['lightblue'] * self.network.num_nodes
        if source is not None:
            node_colors[source] = 'green'
        if destination is not None:
            node_colors[destination] = 'red'
        
        # Highlight nodes on paths
        if paths:
            for path in paths:
                for node in path[1:-1]:  # Color intermediate nodes
                    node_colors[node] = 'orange'
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, self.pos, 
            node_color=node_colors,
            node_size=500, 
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            self.graph, self.pos, 
            font_size=10, 
            font_weight='bold',
            ax=ax
        )
        
        # Prepare edge colors based on pheromone
        edge_colors = [self.graph[u][v]['pheromone'] for u, v in self.graph.edges()]
        
        # Draw all edges
        edges = list(self.graph.edges())
        nx.draw_networkx_edges(
            self.graph, self.pos, 
            edgelist=edges,
            edge_color=edge_colors,
            edge_cmap=plt.cm.Blues,
            width=2,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            ax=ax
        )
        
        # Highlight paths if provided
        if paths:
            for i, path in enumerate(paths):
                # Use a different color for each path
                path_color = plt.cm.Set1(i % 9)
                
                # Create path edges
                path_edges = list(zip(path, path[1:]))
                
                nx.draw_networkx_edges(
                    self.graph, self.pos, 
                    edgelist=path_edges,
                    edge_color=path_color,
                    width=3,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=20,
                    ax=ax
                )
        
        # Create a colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.Blues,
            norm=plt.Normalize(
                vmin=min(edge_colors),
                vmax=max(edge_colors)
            )
        )
        sm.set_array([])
        
        # Add colorbar to the figure
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Pheromone Level')
        
        plt.title('PAMR Routing Simulation')
        plt.axis('off')
        plt.tight_layout()
        
        return fig
