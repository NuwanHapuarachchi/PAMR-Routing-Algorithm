"""
PAMR Dynamic Routing Visualization Test

This script provides a clear visualization of how PAMR routing responds to congestion.
It creates a network and sends packets along specific paths to demonstrate:
1. How congestion builds up on heavily used paths
2. How pheromone levels increase on paths with successful transmissions
3. How the router dynamically selects alternative paths when congestion exceeds thresholds
4. How traffic naturally decays over time
"""

import sys
import os
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import webbrowser
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML
from matplotlib import animation
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for importing the PAMR package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PAMR components - use custom network topology to avoid debug printouts
from pamr.core.routing import PAMRRouter, AdvancedMultiPathRouter

# Custom network topology class to avoid the debug printouts in the original
class CleanNetworkTopology:
    """Network topology management for PAMR routing with no debug output"""
    
    def __init__(self, num_nodes=30, connectivity=0.3, seed=42, variation_factor=0.05):
        """Initialize network topology."""
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
    
    def _initialize_edge_attributes(self, G, pheromone_init=0.5):  # Lower initial pheromone level
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
            G[u][v]['pheromone'] = pheromone_init  # Lower initial pheromone level
            G[u][v]['capacity'] = capacity
            G[u][v]['traffic'] = 0.0
            G[u][v]['congestion'] = 0.0
            # Store original values for controlled variations
            G[u][v]['base_distance'] = distance
            G[u][v]['base_capacity'] = capacity
    
    def update_dynamic_metrics(self, traffic_decay=0.7):
        """Update network metrics with small variations."""
        pass  # Disabled for cleaner testing

class RoutingVisualizer:
    """Class for visualizing PAMR routing behavior with dynamic congestion"""
    
    def __init__(self, num_nodes=15, connectivity=0.3, seed=42, gamma=8.0):
        """
        Initialize the visualizer with a network and router
        
        Args:
            num_nodes: Number of nodes in the network
            connectivity: Probability of connection between nodes
            seed: Random seed for reproducibility
            gamma: Congestion importance factor for the router
        """
        # Create a simple network
        self.network = CleanNetworkTopology(
            num_nodes=num_nodes,
            connectivity=connectivity,
            seed=seed,
            variation_factor=0.0  # No random variation for clearer visualization
        )
        
        # Initialize with a clean graph state (no traffic or congestion)
        self.reset_network_state()
        
        # Create the PAMR router with high gamma to prioritize congestion avoidance
        self.router = PAMRRouter(
            self.network.graph, 
            alpha=2.0,    # Pheromone importance
            beta=3.0,     # Distance importance
            gamma=gamma,  # Congestion importance - higher values avoid congestion more
            adapt_weights=True
        )
        
        # Tracking metrics for visualization
        self.metrics = {
            'iterations': [],
            'path_quality': [],
            'path_lengths': [],
            'congestion_levels': [],
            'pheromone_levels': [],
            'paths_taken': [],
            'active_paths': {}
        }
        
        # For animation
        self.fig = None
        self.ax = None
        
    def null_update(self):
        """Placeholder to prevent default dynamic updates"""
        pass
        
    def reset_network_state(self):
        """Reset all traffic and congestion in the network to zero"""
        for u, v in self.network.graph.edges():
            self.network.graph[u][v]['traffic'] = 0.0
            self.network.graph[u][v]['congestion'] = 0.0
            
    def send_packets(self, source, destination, num_packets=1, visualize=False, output_path=None):
        """
        Send packets from source to destination, observing how routes change with congestion
        
        Args:
            source: Source node ID
            destination: Destination node ID
            num_packets: Number of packets to send
            visualize: Whether to create a dynamic visualization
            output_path: File path to save the visualization (if None, displays interactively)
        """
        print(f"Sending {num_packets} packets from Node {source} to Node {destination}")
        
        # Set up visualization if requested
        if visualize:
            self.setup_visualization()
        
        # Send packets one by one
        all_paths = []
        all_qualities = []
        all_congestion = []
        
        for i in range(num_packets):
            # Find path using PAMR
            path, initial_quality = self.router.find_path(source, destination)
            
            # Store the path taken
            all_paths.append(path)
            
            # Record the state before updates for visualization
            if visualize and i < num_packets-1:  # Don't animate the last state twice
                if hasattr(self, 'anim'):
                    self.update_animation(i)
            
            # Decay traffic on all edges to simulate packets leaving the network
            for u, v in self.network.graph.edges():
                # Decay traffic by 10%
                current_traffic = self.network.graph[u][v].get('traffic', 0)
                self.network.graph[u][v]['traffic'] = max(0, current_traffic * 0.9)
                
                # Update congestion based on new traffic
                capacity = self.network.graph[u][v].get('capacity', 10)
                self.network.graph[u][v]['congestion'] = min(
                    0.6,  # Lowering congestion threshold
                    self.network.graph[u][v]['traffic'] / capacity
                )
            
            # Now update the traffic on the selected path
            max_congestion = 0
            total_pheromone = 0
            
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                
                # Increase traffic on this edge
                self.network.graph[u][v]['traffic'] += 1.0
                
                # Update congestion based on traffic/capacity ratio
                capacity = self.network.graph[u][v]['capacity']
                new_congestion = min(0.6, self.network.graph[u][v]['traffic'] / capacity)
                self.network.graph[u][v]['congestion'] = new_congestion
                
                # Track max congestion for monitoring
                max_congestion = max(max_congestion, new_congestion)
                
                # Make pheromone update more dynamic by decreasing when congestion is high
                # This prevents the constant pheromone level issue
                current_pheromone = self.router.pheromone_table[u].get(v, 0.5)
                congestion_factor = 1.0 - (new_congestion * 0.8)  # Reduce pheromone more when congestion is high
                self.router.pheromone_table[u][v] = current_pheromone * congestion_factor
                
                # Track pheromone for this edge
                total_pheromone += self.router.pheromone_table[u].get(v, 0)
            
            # IMPORTANT FIX: Recalculate path quality based on CURRENT congestion values
            # This ensures path quality reflects the current network state
            quality = self.router._calculate_path_quality(path)
            all_qualities.append(quality)
            
            # Update our visualization metrics with current values
            self.metrics['iterations'].append(i+1)
            self.metrics['path_quality'].append(quality)
            self.metrics['path_lengths'].append(len(path)-1)
            self.metrics['paths_taken'].append(path)
            
            # Track the current active path for this source-destination pair
            key = (source, destination)
            self.metrics['active_paths'][key] = path
            
            # Update our metrics
            avg_pheromone = total_pheromone / (len(path)-1) if len(path) > 1 else 0
            self.metrics['congestion_levels'].append(max_congestion)
            self.metrics['pheromone_levels'].append(avg_pheromone)
            all_congestion.append(max_congestion)
            
            # Log what happened
            path_str = '->'.join([str(node) for node in path])
            print(f"Packet {i+1}: Path {path_str} | Quality: {quality:.4f} | Max Congestion: {max_congestion:.4f}")
            
            # Call router's update to handle path selection for next round
            self.router.update_iteration()
        
        # Show final animation frame if we're visualizing
        if visualize:
            # Update animation one last time
            self.update_animation(num_packets-1)
            
            # Save or display the animation
            if output_path:
                # Save the animation
                self.save_animation(output_path)
            else:
                # Display the animation
                plt.show()
                
        # Return the results
        return all_paths, all_qualities, all_congestion
    
    def setup_visualization(self):
        """Set up the matplotlib figure for visualization"""
        self.fig = plt.figure(figsize=(16, 9))
        self.ax = plt.subplot(111)
        
        # Set up graph layout
        self.pos = self.network.positions
        
        # We don't need to initialize animation here
        # Instead, we'll just draw the network state directly for each packet
    
    def update_animation(self, frame):
        """Update the animation for each frame"""
        self.ax.clear()
        
        # Draw the network
        self.draw_network_state()
        
        # Add title with current frame info
        if frame < len(self.metrics['iterations']):
            quality = self.metrics['path_quality'][frame]
            congestion = self.metrics['congestion_levels'][frame]
            self.ax.set_title(f"Packet {frame+1} | Path Quality: {quality:.4f} | Max Congestion: {congestion:.4f}")
        else:
            # Fallback if we haven't collected metrics for this frame yet
            self.ax.set_title(f"Packet {frame+1}")
        
        return self.ax
    
    def draw_network_state(self):
        """Draw the current state of the network with traffic and congestion"""
        # Get current network state
        G = self.network.graph
        
        # Create color maps for edges based on congestion and traffic
        congestion_cmap = plt.cm.RdYlGn_r  # Red for high congestion, green for low
        pheromone_cmap = plt.cm.Blues      # Blue for high pheromone
        
        # Draw all edges with color based on congestion
        for u, v in G.edges():
            congestion = G[u][v].get('congestion', 0)
            pheromone = self.router.pheromone_table[u].get(v, 0)
            
            # Normalize pheromone for coloring (assumes most pheromones are between 0 and 10)
            normalized_pheromone = min(1.0, pheromone / 10.0)
            
            # Edge width based on traffic
            width = 1.0 + 3.0 * congestion
            
            # Get the appropriate colors
            congestion_color = congestion_cmap(congestion)
            pheromone_color = pheromone_cmap(normalized_pheromone)
            
            # Draw the edge with congestion color
            nx.draw_networkx_edges(
                G, self.pos,
                edgelist=[(u, v)],
                width=width,
                edge_color=[congestion_color],
                arrows=True,
                arrowstyle='-|>',
                arrowsize=10
            )
        
        # Draw active paths with different colors for each source-destination pair
        path_colors = ['magenta', 'cyan', 'yellow', 'orange', 'purple']
        for i, ((source, dest), path) in enumerate(self.metrics['active_paths'].items()):
            color = path_colors[i % len(path_colors)]
            # Draw path edges with color
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                nx.draw_networkx_edges(
                    G, self.pos,
                    edgelist=[(u, v)],
                    width=2.0,
                    edge_color=color,
                    alpha=0.7,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=12
                )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, self.pos,
            node_size=500,
            node_color='lightblue',
            edgecolors='black'
        )
        
        # Highlight source and destination nodes for active paths
        for (source, dest) in self.metrics['active_paths'].keys():
            nx.draw_networkx_nodes(
                G, self.pos,
                nodelist=[source],
                node_size=600,
                node_color='green',
                edgecolors='black'
            )
            nx.draw_networkx_nodes(
                G, self.pos,
                nodelist=[dest],
                node_size=600,
                node_color='red',
                edgecolors='black'
            )
        
        # Add node labels
        nx.draw_networkx_labels(
            G, self.pos,
            font_size=10,
            font_weight='bold'
        )
        
        # Create a legend
        congestion_patch = mpatches.Patch(color='red', label='High Congestion')
        source_patch = mpatches.Patch(color='green', label='Source Node')
        dest_patch = mpatches.Patch(color='red', label='Destination Node')
        self.ax.legend(handles=[congestion_patch, source_patch, dest_patch])
        
        # Remove axis
        self.ax.axis('off')
    
    def save_animation(self, output_path):
        """Save the network state as a static image instead of an animation"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a new figure for the final state
        plt.figure(figsize=(16, 9))
        ax = plt.subplot(111)
        
        # Set the position
        self.pos = self.network.positions
        
        # Draw the final network state
        G = self.network.graph
        
        # Draw edges with color based on congestion
        congestion_cmap = plt.cm.RdYlGn_r
        for u, v in G.edges():
            congestion = G[u][v].get('congestion', 0)
            width = 1.0 + 3.0 * congestion
            congestion_color = congestion_cmap(congestion)
            
            nx.draw_networkx_edges(
                G, self.pos,
                edgelist=[(u, v)],
                width=width,
                edge_color=[congestion_color],
                arrows=True,
                arrowstyle='-|>',
                arrowsize=10,
                ax=ax
            )
        
        # Draw active paths
        path_colors = ['magenta', 'cyan', 'yellow', 'orange', 'purple']
        for i, ((source, dest), path) in enumerate(self.metrics['active_paths'].items()):
            color = path_colors[i % len(path_colors)]
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                nx.draw_networkx_edges(
                    G, self.pos,
                    edgelist=[(u, v)],
                    width=2.0,
                    edge_color=color,
                    alpha=0.7,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=12,
                    ax=ax
                )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, self.pos,
            node_size=500,
            node_color='lightblue',
            edgecolors='black',
            ax=ax
        )
        
        # Highlight source and destination nodes
        for (source, dest) in self.metrics['active_paths'].keys():
            nx.draw_networkx_nodes(
                G, self.pos,
                nodelist=[source],
                node_size=600,
                node_color='green',
                edgecolors='black',
                ax=ax
            )
            nx.draw_networkx_nodes(
                G, self.pos,
                nodelist=[dest],
                node_size=600,
                node_color='red',
                edgecolors='black',
                ax=ax
            )
        
        # Add node labels
        nx.draw_networkx_labels(
            G, self.pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        # Create a legend
        congestion_patch = mpatches.Patch(color='red', label='High Congestion')
        source_patch = mpatches.Patch(color='green', label='Source Node')
        dest_patch = mpatches.Patch(color='red', label='Destination Node')
        ax.legend(handles=[congestion_patch, source_patch, dest_patch])
        
        # Add title
        if self.metrics['congestion_levels']:
            quality = self.metrics['path_quality'][-1]
            congestion = self.metrics['congestion_levels'][-1]
            ax.set_title(f"Final State | Path Quality: {quality:.4f} | Max Congestion: {congestion:.4f}")
        
        # Remove axis
        ax.axis('off')
        
        # Save the figure
        output_path_png = output_path.replace('.gif', '.png')
        plt.savefig(output_path_png)
        print(f"Final network state saved to {output_path_png}")
        plt.close()
        
    def visualize_metrics(self, output_dir='./visualization_results'):
        """Create visualizations of the routing metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots for different metrics
        plt.figure(figsize=(12, 8))
        
        # 1. Path Quality Over Time
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['iterations'], self.metrics['path_quality'], 'b-', linewidth=2)
        plt.title('Path Quality Over Time')
        plt.xlabel('Packet Number')
        plt.ylabel('Path Quality')
        plt.grid(True)
        
        # 2. Congestion Levels Over Time
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['iterations'], self.metrics['congestion_levels'], 'r-', linewidth=2)
        plt.title('Max Congestion Over Time')
        plt.xlabel('Packet Number')
        plt.ylabel('Max Congestion')
        plt.grid(True)
        
        # 3. Pheromone Levels Over Time
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['iterations'], self.metrics['pheromone_levels'], 'g-', linewidth=2)
        plt.title('Average Pheromone Level Over Time')
        plt.xlabel('Packet Number')
        plt.ylabel('Avg Pheromone')
        plt.grid(True)
        
        # 4. Path Length Over Time
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['iterations'], self.metrics['path_lengths'], 'purple', linewidth=2)
        plt.title('Path Length Over Time')
        plt.xlabel('Packet Number')
        plt.ylabel('Number of Hops')
        plt.grid(True)
        
        plt.tight_layout()
        metrics_path = os.path.join(output_dir, 'routing_metrics.png')
        plt.savefig(metrics_path)
        plt.close()
        
        # Create a visualization of all paths taken between the source and destination
        self.visualize_path_changes(output_dir)
        
        return metrics_path
    
    def visualize_path_changes(self, output_dir):
        """Create a visualization showing how paths changed over time"""
        if not self.metrics['paths_taken']:
            return
        
        # Get source and destination from the first path
        first_path = self.metrics['paths_taken'][0]
        if len(first_path) < 2:
            return
            
        source = first_path[0]
        destination = first_path[-1]
        
        # Create a new figure
        plt.figure(figsize=(14, 10))
        
        # Draw network
        pos = self.network.positions
        G = self.network.graph
        
        # Draw all edges lightly
        nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
        
        # Highlight source and destination
        nx.draw_networkx_nodes(G, pos, nodelist=[source], node_size=400, node_color='green')
        nx.draw_networkx_nodes(G, pos, nodelist=[destination], node_size=400, node_color='red')
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Find all unique paths taken
        unique_paths = []
        for path in self.metrics['paths_taken']:
            if path not in unique_paths:
                unique_paths.append(path)
        
        # Color map for paths
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Draw each unique path with a different color
        legend_elements = []
        for i, path in enumerate(unique_paths):
            # Create path edges
            path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
            
            # Choose color (cycle through colors if more paths than colors)
            color = colors[i % len(colors)]
            
            # Draw the path
            nx.draw_networkx_edges(
                G, pos,
                edgelist=path_edges,
                width=2.0,
                edge_color=color,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=15
            )
            
            # Count how many times this path was used
            count = self.metrics['paths_taken'].count(path)
            path_str = '->'.join([str(node) for node in path])
            
            # Add to legend
            legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                        label=f"Path {i+1}: {path_str} (used {count} times)"))
        
        # Add source and destination to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                    markersize=10, label=f'Source (Node {source})'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                    markersize=10, label=f'Destination (Node {destination})'))
        
        # Add legend
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"All Paths Taken from Node {source} to Node {destination}")
        plt.axis('off')
        
        # Save figure
        path_changes_file = os.path.join(output_dir, f'path_changes_{source}_to_{destination}.png')
        plt.savefig(path_changes_file, bbox_inches='tight')
        plt.close()
        
        return path_changes_file
    
    def send_packets_multi_path(self, source, destination, num_packets=1, visualize=False, output_path=None):
        """
        Send packets from source to destination using multiple paths simultaneously
        
        Args:
            source: Source node ID
            destination: Destination node ID
            num_packets: Number of packets to send
            visualize: Whether to create a dynamic visualization
            output_path: File path to save the visualization (if None, displays interactively)
        """
        print(f"Sending {num_packets} packets from Node {source} to Node {destination} using multi-path routing")
        
        # Set up visualization if requested
        if visualize:
            self.setup_visualization()
        
        # Track all paths used
        all_paths = []
        all_qualities = []
        all_congestion = []
        
        # Track distribution of packets
        path_distribution = {}
        
        for i in range(num_packets):
            # Get multiple paths with distribution ratios
            path_options = self.router.get_multi_path_routing(source, destination)
            
            if not path_options:
                print(f"No path found from {source} to {destination}")
                break
                
            # Find the primary path (highest ratio)
            primary_path, primary_ratio = max(path_options, key=lambda x: x[1])
            primary_path_str = '->'.join([str(node) for node in primary_path])
            
            # Calculate quality and congestion for the primary path
            primary_quality = self.router._calculate_path_quality(primary_path)
            
            # Calculate max congestion on primary path
            primary_max_congestion = 0
            for j in range(len(primary_path)-1):
                u, v = primary_path[j], primary_path[j+1]
                primary_max_congestion = max(primary_max_congestion, 
                                            self.network.graph[u][v].get('congestion', 0))
            
            # Print detailed information about the primary path
            print(f"Packet {i+1}: Path {primary_path_str} | Quality: {primary_quality:.4f} | Max Congestion: {primary_max_congestion:.4f}")
            
            # If there are multiple paths, show them as alternatives
            if len(path_options) > 1:
                print(f"  Alternative paths:")
                for idx, (path, ratio) in enumerate(path_options):
                    if path != primary_path:  # Skip the primary path as we've already shown it
                        path_str = '->'.join([str(node) for node in path])
                        path_quality = self.router._calculate_path_quality(path)
                        
                        # Calculate max congestion on this path
                        path_max_congestion = 0
                        for j in range(len(path)-1):
                            u, v = path[j], path[j+1]
                            path_max_congestion = max(path_max_congestion, 
                                                      self.network.graph[u][v].get('congestion', 0))
                                                      
                        print(f"    Path: {path_str} | Quality: {path_quality:.4f} | Max Congestion: {path_max_congestion:.4f} | Traffic Share: {ratio:.2f}")
                
            # Track this path for visualization
            if primary_path_str not in path_distribution:
                path_distribution[primary_path_str] = 0
            path_distribution[primary_path_str] += primary_ratio
            
            # Simulate sending packets according to the distribution ratios
            for path, ratio in path_options:
                # Round the ratio to determine how many packets to send on this path
                packets_on_path = max(1, round(ratio * 1.0))  # At least 1 packet
                
                if packets_on_path > 0:
                    path_str = '->'.join([str(node) for node in path])
                    
                    # Recalculate current path quality based on actual congestion
                    quality = self.router._calculate_path_quality(path)
                    
                    # Update metrics for this path
                    all_paths.append(path)
                    all_qualities.append(quality)
                    
                    # Calculate max congestion on this path
                    max_congestion = 0
                    
                    # Update traffic on this path proportionally
                    for j in range(len(path)-1):
                        u, v = path[j], path[j+1]
                        
                        # Add fractional traffic based on ratio
                        self.network.graph[u][v]['traffic'] += packets_on_path
                        
                        # Update congestion
                        capacity = self.network.graph[u][v].get('capacity', 10)
                        new_congestion = min(0.6, self.network.graph[u][v]['traffic'] / capacity)
                        self.network.graph[u][v]['congestion'] = new_congestion
                        
                        # Track max congestion
                        max_congestion = max(max_congestion, new_congestion)
                    
                    all_congestion.append(max_congestion)
                    
                    # Store path for visualization
                    key = (source, destination)
                    # Track only the primary path for visualization simplicity
                    if ratio == max([r for _, r in path_options]):
                        self.metrics['active_paths'][key] = path
            
            # Update our visualization metrics with primary path data
            self.metrics['iterations'].append(i+1)
            self.metrics['path_quality'].append(primary_quality)
            self.metrics['path_lengths'].append(len(primary_path)-1)
            self.metrics['congestion_levels'].append(primary_max_congestion)
            
            # Calculate avg pheromone on primary path
            total_pheromone = 0
            for j in range(len(primary_path)-1):
                u, v = primary_path[j], primary_path[j+1]
                total_pheromone += self.router.pheromone_table[u].get(v, 0)
            avg_pheromone = total_pheromone / (len(primary_path)-1) if len(primary_path) > 1 else 0
            self.metrics['pheromone_levels'].append(avg_pheromone)
            
            # Store all used paths for visualization
            self.metrics['paths_taken'].append(primary_path)
            
            # Update animation if visualizing
            if visualize and i < num_packets-1:
                if hasattr(self, 'anim'):
                    self.update_animation(i)
            
            # Decay traffic on all edges
            for u, v in self.network.graph.edges():
                # Decay traffic by 10%
                current_traffic = self.network.graph[u][v].get('traffic', 0)
                self.network.graph[u][v]['traffic'] = max(0, current_traffic * 0.9)
                
                # Update congestion
                capacity = self.network.graph[u][v].get('capacity', 10)
                self.network.graph[u][v]['congestion'] = min(0.6, self.network.graph[u][v]['traffic'] / capacity)
            
            # Call router's update
            self.router.update_iteration()
        
        # Calculate which paths were used most frequently
        sorted_paths = sorted(path_distribution.items(), key=lambda x: x[1], reverse=True)
        print("\nPath Usage Summary:")
        for path_str, usage in sorted_paths:
            print(f"  {path_str}: {usage:.2f} packets")
            
        # Final visualization step
        if visualize:
            self.update_animation(num_packets-1)
            
            if output_path:
                self.save_animation(output_path)
            else:
                plt.show()
        
        return all_paths, all_qualities, all_congestion

def run_visualization_test():
    """Run a test of the Advanced Multi-Path routing visualization with sophisticated path discovery"""
    # Create output directory
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Import the singleton network from core.network
    from pamr.core.network import network
    
    print("Creating advanced multi-path routing visualizer...")
    
    # Create the advanced multi-path router using the imported network
    router = AdvancedMultiPathRouter(
        network.graph, 
        alpha=2.0,    # Pheromone importance
        beta=3.0,     # Distance importance
        gamma=15.0,   # Congestion importance - higher values avoid congestion more
        adapt_weights=True
    )
    
    # Create visualizer with advanced router
    visualizer = RoutingVisualizer(
        num_nodes=network.num_nodes, 
        connectivity=network.connectivity, 
        seed=network.seed, 
        gamma=15.0
    )
    visualizer.router = router  # Replace default router with advanced router
    visualizer.network = network  # Replace default network with the singleton network
    
    # Choose source and destination 
    source = 2  
    destination = 3  
    
    print(f"Selected source={source}, destination={destination}")
    print(f"Using advanced multi-path router with up to {router.max_paths_to_discover} diverse paths")
    print(f"Will simultaneously distribute traffic across up to {router.max_paths_to_use} paths")
    
    # Use advanced multi-path routing
    visualizer.send_packets_multi_path(
        source=source, 
        destination=destination, 
        num_packets=30,
        visualize=True,
        output_path=os.path.join(output_dir, 'routing_animation.png')
    )
    
    # Create metric visualizations
    metrics_path = visualizer.visualize_metrics(output_dir)
    
    # Open the metrics visualization
    webbrowser.open(f"file:///{os.path.abspath(metrics_path)}")
    
    print("\nExperiment complete!")
    print(f"Results available in: {os.path.abspath(output_dir)}")
    
    # Show path visualization
    path_changes_file = os.path.join(output_dir, f'path_changes_{source}_to_{destination}.png')
    if os.path.exists(path_changes_file):
        webbrowser.open(f"file:///{os.path.abspath(path_changes_file)}")

if __name__ == "__main__":
    run_visualization_test()