import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class NetworkVisualizer:
    """Network visualization for PAMR protocol with enhanced metrics visualization."""
    
    def __init__(self, network):
        self.network = network
        self.graph = network.graph
        self.pos = network.positions
    
    def visualize_network(self, source=None, destination=None, paths=None, 
                          edge_attribute='pheromone', title=None):
        """Visualize the network with configurable edge attributes.
        
        Args:
            source: Source node (colored green)
            destination: Destination node (colored red)
            paths: List of paths to highlight
            edge_attribute: Edge attribute to visualize ('pheromone', 'distance', 
                           'capacity', 'traffic', or 'congestion')
            title: Custom title for the plot
        """
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
        
        # Choose colormap based on attribute
        if edge_attribute in ['congestion', 'traffic']:
            edge_cmap = plt.cm.Reds  # Red colormap for congestion/traffic
        elif edge_attribute == 'distance':
            edge_cmap = plt.cm.YlOrRd  # Yellow-Orange-Red for distance
        elif edge_attribute == 'capacity':
            edge_cmap = plt.cm.Greens  # Green colormap for capacity
        else:  # Default for pheromone
            edge_cmap = plt.cm.Blues
        
        # Get edge colors based on selected attribute
        edge_colors = []
        edges = list(self.graph.edges())
        
        for u, v in edges:
            if edge_attribute in self.graph[u][v]:
                edge_colors.append(self.graph[u][v][edge_attribute])
            else:
                edge_colors.append(0)  # Default if attribute doesn't exist
        
        # Draw all edges
        nx.draw_networkx_edges(
            self.graph, self.pos, 
            edgelist=edges,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
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
            cmap=edge_cmap,
            norm=plt.Normalize(
                vmin=min(edge_colors) if edge_colors else 0,
                vmax=max(edge_colors) if edge_colors else 1
            )
        )
        sm.set_array([])
        
        # Add colorbar to the figure
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        
        # Set colorbar label based on attribute
        attribute_labels = {
            'pheromone': 'Pheromone Level',
            'distance': 'Distance/Cost',
            'capacity': 'Bandwidth Capacity',
            'traffic': 'Traffic Load',
            'congestion': 'Congestion Level'
        }
        cbar.set_label(attribute_labels.get(edge_attribute, edge_attribute.capitalize()))
        
        # Set plot title
        if title:
            plt.title(title)
        else:
            plt.title(f'PAMR Network - {attribute_labels.get(edge_attribute, edge_attribute.capitalize())}')
        
        plt.axis('off')
        plt.tight_layout()
        
        return fig
    
    def display_network_stats(self):
        """Display key statistics about the network topology."""
        G = self.graph
        stats = {
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Average degree": float(sum(dict(G.degree()).values())) / G.number_of_nodes(),
            "Average out-degree": float(sum(dict(G.out_degree()).values())) / G.number_of_nodes(),
            "Average in-degree": float(sum(dict(G.in_degree()).values())) / G.number_of_nodes(),
            "Average distance": np.mean([G[u][v]['distance'] for u, v in G.edges()]),
            "Average capacity": np.mean([G[u][v]['capacity'] for u, v in G.edges()]),
            "Connectivity": self.network.connectivity,
            "Variation factor": self.network.variation_factor,
            "Seed": self.network.seed
        }
        
        return stats
    
    def visualize_metrics_distribution(self):
        """Visualize the distribution of different edge metrics."""
        metrics = ['distance', 'capacity', 'traffic', 'congestion', 'pheromone']
        available_metrics = []
        
        # Check which metrics are available in the graph
        for metric in metrics:
            if all(metric in self.graph[u][v] for u, v in self.graph.edges()):
                available_metrics.append(metric)
        
        # Create subplots for each available metric
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 3*len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]  # Make it iterable when only one subplot
        
        for i, metric in enumerate(available_metrics):
            values = [self.graph[u][v][metric] for u, v in self.graph.edges()]
            axes[i].hist(values, bins=20)
            axes[i].set_title(f'Distribution of {metric}')
            axes[i].set_xlabel(metric.capitalize())
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
