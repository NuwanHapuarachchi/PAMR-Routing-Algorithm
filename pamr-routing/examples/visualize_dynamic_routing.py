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
from argparse import ArgumentParser

# Add parent directory to path for importing the PAMR package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PAMR components - use custom network topology to avoid debug printouts
from pamr.core.routing import PAMRRouter, AdvancedMultiPathRouter
from pamr.core.network import NetworkTopology
from pamr.simulation.simulator import PAMRSimulator
from pamr.visualization.network_viz import NetworkVisualizer

# Import the new Mininet simulator
from pamr.simulation.mininet_sim import PAMRMininetSimulator

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
            'active_paths': {},
            'path_pheromones': {}  # Initialize the path_pheromones dictionary here
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
        
        # Create a larger figure to accommodate the additional plot
        plt.figure(figsize=(16, 12))
        
        # 1. Path Quality Over Time
        plt.subplot(2, 3, 1)
        plt.plot(self.metrics['iterations'], self.metrics['path_quality'], 'b-', linewidth=2)
        plt.title('Path Quality Over Time')
        plt.xlabel('Packet Number')
        plt.ylabel('Path Quality')
        plt.grid(True)
        
        # 2. Congestion Levels Over Time
        plt.subplot(2, 3, 2)
        plt.plot(self.metrics['iterations'], self.metrics['congestion_levels'], 'r-', linewidth=2)
        plt.title('Max Congestion Over Time')
        plt.xlabel('Packet Number')
        plt.ylabel('Max Congestion')
        plt.grid(True)
        
        # 3. Path Length Over Time
        plt.subplot(2, 3, 3)
        plt.plot(self.metrics['iterations'], self.metrics['path_lengths'], 'purple', linewidth=2)
        plt.title('Path Length Over Time')
        plt.xlabel('Packet Number')
        plt.ylabel('Number of Hops')
        plt.grid(True)
        
        # Path-specific pheromone levels (span the entire bottom row for better visibility)
        ax_pheromones = plt.subplot(2, 3, (4, 6))  # Span all three columns
        
        if 'path_pheromones' in self.metrics and self.metrics['path_pheromones']:
            # Get color cycle for different paths
            colors = plt.cm.tab10.colors
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            
            # Create a full x-axis range for all iterations
            all_iterations = self.metrics['iterations']
            
            # Plot each path's pheromone levels as separate lines
            for i, (path_str, pheromone_values) in enumerate(self.metrics['path_pheromones'].items()):
                # Fill missing values with NaN for cleaner plotting
                full_values = []
                for j, iter_num in enumerate(all_iterations):
                    if j < len(pheromone_values) and pheromone_values[j] is not None:
                        full_values.append(pheromone_values[j])
                    else:
                        full_values.append(float('nan'))
                
                # Choose color and marker for this path
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                # Count path usage for the legend
                usage_count = self.metrics['paths_taken'].count([int(n) for n in path_str.split('->')])
                
                # Plot with a continuous line connecting only the valid points
                mask = ~np.isnan(np.array(full_values))
                
                if any(mask):  # Only plot if we have valid data points
                    # Plot line with markers at data points
                    ax_pheromones.plot(
                        np.array(all_iterations)[mask], 
                        np.array(full_values)[mask], 
                        marker=marker, 
                        markersize=8,  # Slightly larger markers for better visibility
                        markevery=max(1, len(all_iterations)//15),  # Show markers at regular intervals
                        linestyle='-', 
                        linewidth=2.5,  # Slightly thicker lines
                        color=color, 
                        alpha=0.9,
                        label=f"Path {path_str[:25]}{'...' if len(path_str) > 25 else ''} ({usage_count})"
                    )
            
            # Add horizontal line at initial pheromone level (0.5)
            ax_pheromones.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Initial level')
            
            # Set y-axis to start from 0 (or slightly below for visibility)
            y_min = max(0, min([min([v for v in vals if v is not None] or [0]) 
                           for vals in self.metrics['path_pheromones'].values()] or [0]) - 0.05)
            y_max = max([max([v for v in vals if v is not None] or [0]) 
                         for vals in self.metrics['path_pheromones'].values()] or [0]) + 0.1
            
            ax_pheromones.set_ylim(y_min, y_max)
            
            # Add grid lines
            ax_pheromones.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend with path information - better positioning with more space
            ax_pheromones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
        
        ax_pheromones.set_title('Pheromone Levels By Path')
        ax_pheromones.set_xlabel('Packet Number')
        ax_pheromones.set_ylabel('Pheromone Level')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for the legend at the bottom
        metrics_path = os.path.join(output_dir, 'routing_metrics.png')
        plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create a visualization of all paths taken between the source and destination
        self.visualize_path_changes(output_dir)
        
        # Still create a separate, more detailed pheromone visualization
        if 'path_pheromones' in self.metrics and self.metrics['path_pheromones']:
            path_pheromone_path = self.visualize_path_pheromones(output_dir)
            print(f"Detailed path-specific pheromone visualization saved to: {path_pheromone_path}")
        
        return metrics_path
    
    def visualize_path_pheromones(self, output_dir):
        """Create a separate visualization showing pheromone levels for each unique path"""
        if 'path_pheromones' not in self.metrics or not self.metrics['path_pheromones']:
            return
            
        # Create a new figure for path-specific pheromone levels
        plt.figure(figsize=(16, 10))
        
        # Get color cycle for different paths
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Create a full x-axis range for all iterations
        all_iterations = self.metrics['iterations']
        
        # Track legend entries
        legend_entries = []
        
        # Create a separate subplot for a clearer comparison
        ax = plt.subplot(111)
        
        # Add a light gray grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot each path's pheromone levels as separate lines
        for i, (path_str, pheromone_values) in enumerate(self.metrics['path_pheromones'].items()):
            # Fill missing values with NaN for cleaner plotting
            full_values = []
            for j, iter_num in enumerate(all_iterations):
                if j < len(pheromone_values) and pheromone_values[j] is not None:
                    full_values.append(pheromone_values[j])
                else:
                    full_values.append(float('nan'))
            
            # Choose color and marker for this path
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Count path usage for the legend
            usage_count = self.metrics['paths_taken'].count([int(n) for n in path_str.split('->')])
            
            # Plot with a continuous line connecting only the valid points
            mask = ~np.isnan(full_values)
            
            if any(mask):  # Only plot if we have valid data points
                # Plot line with markers at data points
                line, = ax.plot(
                    np.array(all_iterations)[mask], 
                    np.array(full_values)[mask], 
                    marker=marker, 
                    markersize=8,
                    markevery=max(1, len(all_iterations)//20),  # Show markers at regular intervals
                    linestyle='-', 
                    linewidth=2.5, 
                    color=color, 
                    alpha=0.9,
                    label=f"Path {path_str} (used {usage_count} times)"
                )
                legend_entries.append(line)
                
                # Annotate the final pheromone value
                last_valid_idx = np.where(mask)[0][-1]
                ax.annotate(
                    f"{full_values[last_valid_idx]:.3f}",
                    xy=(all_iterations[last_valid_idx], full_values[last_valid_idx]),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color=color,
                    backgroundcolor='white',
                    alpha=0.8
                )
        
        # Add labels and title
        ax.set_title('Pheromone Levels By Path Over Iterations', fontsize=16, fontweight='bold')
        ax.set_xlabel('Iteration Number', fontsize=14)
        ax.set_ylabel('Average Pheromone Level', fontsize=14)
        
        # Set y-axis to start from 0 (or slightly below for visibility)
        y_min = max(0, min([min([v for v in vals if v is not None] or [0]) 
                       for vals in self.metrics['path_pheromones'].values()] or [0]) - 0.05)
        y_max = max([max([v for v in vals if v is not None] or [0]) 
                     for vals in self.metrics['path_pheromones'].values()] or [0]) + 0.1
        
        ax.set_ylim(y_min, y_max)
        
        # Add horizontal line at initial pheromone level (0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Initial pheromone level')
        
        # Add legend with path information in two columns
        if legend_entries:
            ax.legend(
                handles=legend_entries + [plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.5)], 
                loc='upper center', 
                bbox_to_anchor=(0.5, -0.1),
                ncol=2, 
                fontsize=12,
                frameon=True,
                shadow=True
            )
            
        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        # Save the figure with high DPI for better quality
        pheromone_path = os.path.join(output_dir, 'path_pheromone_levels.png')
        plt.savefig(pheromone_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Also save an interactive HTML version that allows zooming
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            
            # Create a DataFrame for Plotly
            df_data = []
            for path_str, pheromone_values in self.metrics['path_pheromones'].items():
                for iter_idx, pheromone in enumerate(pheromone_values):
                    if pheromone is not None:
                        df_data.append({
                            'Iteration': iter_idx + 1,
                            'Pheromone': pheromone,
                            'Path': path_str
                        })
            
            if df_data:
                df = pd.DataFrame(df_data)
                
                # Create interactive plot
                fig = px.line(
                    df, 
                    x='Iteration', 
                    y='Pheromone', 
                    color='Path',
                    markers=True,
                    title='Pheromone Levels By Path Over Iterations',
                    labels={'Pheromone': 'Average Pheromone Level', 'Iteration': 'Iteration Number'}
                )
                
                # Improve layout
                fig.update_layout(
                    legend_title_text='Path',
                    hovermode='closest',
                    xaxis=dict(title=dict(font=dict(size=14))),
                    yaxis=dict(title=dict(font=dict(size=14))),
                    title=dict(font=dict(size=16)),
                )
                
                # Add reference line for initial pheromone
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0.5,
                    x1=df['Iteration'].max(),
                    y1=0.5,
                    line=dict(color="gray", width=2, dash="dash"),
                )
                
                # Save as HTML
                html_path = os.path.join(output_dir, 'interactive_pheromone_levels.html')
                fig.write_html(html_path)
                print(f"Interactive pheromone visualization saved to: {html_path}")
        except ImportError:
            # If plotly is not available, just continue with the static image
            pass
            
        return pheromone_path
    
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
        Send packets from source to destination using multiple paths with alternating path selection
        
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
        
        # Path cache to store discovered paths between source and destination
        path_cache = []
        
        # Initialize the path_pheromones dictionary if it doesn't exist
        if 'path_pheromones' not in self.metrics:
            self.metrics['path_pheromones'] = {}
        
        # Track path iterations for each unique path
        path_iterations = {}
        
        # Set parameters for dynamic pheromone updates - ADJUSTED FOR BETTER BALANCE
        pheromone_reward = 0.13        # INCREASED: Amount to increase pheromone on successful paths
        pheromone_evaporation = 0.05  # REDUCED: Rate at which pheromones evaporate each iteration
        congestion_penalty = 0.2      # How much congestion reduces pheromone reinforcement
        min_pheromone = 0.1           # Minimum pheromone level to prevent complete evaporation
        max_pheromone = 2.0           # Maximum pheromone level to prevent runaway values
        
        # Print header for pheromone tracking in terminal
        print("\n{:<5} {:<40} {:<15}".format("Packet", "Path", "Avg Pheromone"))
        print("-" * 65)
        
        for i in range(num_packets):
            # Get multiple paths with distribution ratios
            path_options = self.router.get_multi_path_routing(source, destination)
            
            if not path_options:
                print(f"No path found from {source} to {destination}")
                break
                
            # Store all paths in our cache if this is a new discovery
            if not path_cache:
                path_cache = [path for path, _ in path_options]
            
            # Select a path for this packet - rotate through paths to balance traffic
            # This implements true per-packet multi-path routing
            selected_path_idx = i % len(path_options)
            path, _ = path_options[selected_path_idx]
            
            path_str = '->'.join([str(node) for node in path])
            
            # Calculate quality and congestion for the path
            quality = self.router._calculate_path_quality(path)
            
            # Calculate max congestion on path
            max_congestion = 0
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                max_congestion = max(max_congestion, 
                                    self.network.graph[u][v].get('congestion', 0))
            
            # Print detailed information about the selected path
            print(f"Packet {i+1}: Path {path_str} | Quality: {quality:.4f} | Max Congestion: {max_congestion:.4f}")
            
            # Show alternative paths that could have been taken
            if len(path_options) > 1:
                print(f"  Alternative paths:")
                for idx, (alt_path, ratio) in enumerate(path_options):
                    if alt_path != path:  # Skip the selected path as we've already shown it
                        alt_path_str = '->'.join([str(node) for node in alt_path])
                        alt_path_quality = self.router._calculate_path_quality(alt_path)
                        
                        # Calculate max congestion on this alternate path
                        alt_path_max_congestion = 0
                        for j in range(len(alt_path)-1):
                            u, v = alt_path[j], alt_path[j+1]
                            alt_path_max_congestion = max(alt_path_max_congestion, 
                                                     self.network.graph[u][v].get('congestion', 0))
                                                      
                        print(f"    Path: {alt_path_str} | Quality: {alt_path_quality:.4f} | Max Congestion: {alt_path_max_congestion:.4f} | Traffic Share: {ratio:.2f}")
                
            # Track this path for visualization
            if path_str not in path_distribution:
                path_distribution[path_str] = 0
            path_distribution[path_str] += 1
            
            # Recalculate current path quality based on actual congestion
            quality = self.router._calculate_path_quality(path)
            
            # Update metrics for this path
            all_paths.append(path)
            all_qualities.append(quality)
            
            # IMPORTANT: Apply global pheromone evaporation to all edges
            # This ensures pheromone levels change over time, even for paths that aren't used
            # Only apply evaporation to edges not in the current or recent paths
            for u in self.router.pheromone_table:
                for v in self.router.pheromone_table[u]:
                    # Evaporate pheromones on all edges
                    current = self.router.pheromone_table[u][v]
                    self.router.pheromone_table[u][v] = max(min_pheromone, current * (1 - pheromone_evaporation))
            
            # Update traffic on this path - only on the selected path, not on all paths
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                
                # Add single packet of traffic (value of 1.0)
                self.network.graph[u][v]['traffic'] += 1.0
                
                # Update congestion
                capacity = self.network.graph[u][v].get('capacity', 10)
                new_congestion = min(0.6, self.network.graph[u][v]['traffic'] / capacity)
                self.network.graph[u][v]['congestion'] = new_congestion
                
                # Track max congestion
                max_congestion = max(max_congestion, new_congestion)
                
                # IMPORTANT: Apply pheromone reinforcement to the selected path
                # The amount of reinforcement is reduced by congestion but enhanced by path quality
                congestion_factor = max(0, 1 - (new_congestion * congestion_penalty))
                reinforcement = pheromone_reward * (quality + 0.5) * congestion_factor  # Added 0.5 to ensure positive reinforcement
                
                # Add reinforcement - using a formula that allows pheromone growth for good paths
                current = self.router.pheromone_table[u].get(v, 0.5)
                # Stronger reinforcement for good paths, especially at lower pheromone levels
                self.router.pheromone_table[u][v] = min(max_pheromone, current + reinforcement)
            
            all_congestion.append(max_congestion)
            
            # Store path for visualization
            key = (source, destination)
            self.metrics['active_paths'][key] = path
            
            # Update our visualization metrics with current path data
            self.metrics['iterations'].append(i+1)
            self.metrics['path_quality'].append(quality)
            self.metrics['path_lengths'].append(len(path)-1)
            self.metrics['congestion_levels'].append(max_congestion)
            
            # Calculate and track pheromone levels for EACH path separately
            path_pheromone_values = {}
            
            # Print pheromone levels for all paths in this iteration
            print("\n{:<5} {:<40} {:<15} {:<10} {:<15} {:<15}".format(
                f"#{i+1}", "Path", "Pheromone", "Quality", "Max Congestion", "Traffic Share"))
            print("-" * 105)
            
            for idx, (p_option, traffic_share) in enumerate(path_options):
                p_str = '->'.join([str(node) for node in p_option])
                
                # Initialize for new paths
                if p_str not in self.metrics['path_pheromones']:
                    self.metrics['path_pheromones'][p_str] = [None] * i  # Fill with None for previous iterations
                    path_iterations[p_str] = []
                
                # Calculate average pheromone for this path
                total_pheromone = 0
                max_congestion = 0
                for j in range(len(p_option)-1):
                    u, v = p_option[j], p_option[j+1]
                    total_pheromone += self.router.pheromone_table[u].get(v, 0)
                    max_congestion = max(max_congestion, self.network.graph[u][v].get('congestion', 0))
                
                avg_pheromone = total_pheromone / (len(p_option)-1) if len(p_option) > 1 else 0
                
                # Calculate path quality
                path_quality = self.router._calculate_path_quality(p_option)
                
                # Store in temporary dict for printing
                path_pheromone_values[p_str] = avg_pheromone
                
                # Store pheromone value
                self.metrics['path_pheromones'][p_str].append(avg_pheromone)
                path_iterations[p_str].append(i+1)  # Store which iteration this path was considered
                
                # Print metrics for this path
                is_selected = " *" if p_str == path_str else ""
                print("{:<5} {:<40} {:<15.6f} {:<10.4f} {:<15.4f} {:<15.2f}{}".format(
                    "", p_str, avg_pheromone, path_quality, max_congestion, traffic_share, is_selected))
            
            print("")  # Extra space for readability
            
            # Make sure all paths have the same number of entries
            max_length = i + 1
            for p_str in self.metrics['path_pheromones']:
                current_length = len(self.metrics['path_pheromones'][p_str])
                if current_length < max_length:
                    # Add None for iterations where this path wasn't considered
                    self.metrics['path_pheromones'][p_str].extend([None] * (max_length - current_length))
            
            # Also track the selected path's pheromone in the main metrics
            total_pheromone = 0
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                total_pheromone += self.router.pheromone_table[u].get(v, 0)
            avg_pheromone = total_pheromone / (len(path)-1) if len(path) > 1 else 0
            self.metrics['pheromone_levels'].append(avg_pheromone)
            
            # Store all used paths for visualization
            self.metrics['paths_taken'].append(path)
            
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
        
        # Print summary of pheromone levels for each path at the end
        print("\n=== FINAL PHEROMONE LEVELS BY PATH ===")
        print("{:<40} {:<15} {:<10} {:<15} {:<15}".format(
            "Path", "Final Pheromone", "Usage", "Quality", "Max Congestion"))
        print("-" * 95)
        
        # Get all unique paths that have been used
        all_path_strings = list(self.metrics['path_pheromones'].keys())
        
        # Calculate the final average pheromone for each path
        for path_str in all_path_strings:
            # Get the last non-None pheromone value
            pheromone_values = [p for p in self.metrics['path_pheromones'][path_str] if p is not None]
            if pheromone_values:
                final_pheromone = pheromone_values[-1]
                usage = path_distribution.get(path_str, 0)
                
                # Calculate final path quality and congestion
                path_nodes = [int(n) for n in path_str.split('->')]
                path_quality = self.router._calculate_path_quality(path_nodes)
                
                # Get max congestion for this path
                max_congestion = 0
                for j in range(len(path_nodes)-1):
                    u, v = path_nodes[j], path_nodes[j+1]
                    max_congestion = max(max_congestion, self.network.graph[u][v].get('congestion', 0))
                
                print("{:<40} {:<15.6f} {:<10} {:<15.4f} {:<15.4f}".format(
                    path_str, final_pheromone, usage, path_quality, max_congestion))
        
        # Calculate which paths were used most frequently
        sorted_paths = sorted(path_distribution.items(), key=lambda x: x[1], reverse=True)
        print("\nPath Usage Summary:")
        for path_str, usage in sorted_paths:
            print(f"  {path_str}: {usage} packets")
            
        # Final visualization step
        if visualize:
            self.update_animation(num_packets-1)
            
            if output_path:
                self.save_animation(output_path)
            else:
                plt.show()
        
        return all_paths, all_qualities, all_congestion

def main():
    parser = ArgumentParser(description='Visualize dynamic routing with PAMR protocol')
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes in the network')
    parser.add_argument('--connectivity', type=float, default=0.3, help='Connectivity parameter for network generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--iterations', type=int, default=10, help='Number of routing iterations to simulate')
    parser.add_argument('--packets', type=int, default=50, help='Number of packets to route per iteration')
    parser.add_argument('--use-mininet', action='store_true', help='Use Mininet for simulation and visualization')
    args = parser.parse_args()
    
    # Create network topology
    network = NetworkTopology(
        num_nodes=args.nodes,
        connectivity=args.connectivity,
        seed=args.seed
    )

    # If using Mininet, run a Mininet simulation
    if args.use_mininet:
        from mininet.log import setLogLevel
        setLogLevel('info')
        
        # Create output directory
        os.makedirs("mininet_results", exist_ok=True)
        
        # Create a Mininet simulator
        simulator = PAMRMininetSimulator()
        
        # Build Mininet topology from PAMR network
        simulator.build_from_pamr_network(network)
        
        # Visualize the network topology before starting
        simulator.visualize_network(
            title="PAMR Network in Mininet",
            save_path="mininet_results/mininet_topology.png"
        )
        
        print("Starting Mininet simulation. Press Ctrl+D to exit the Mininet CLI when done.")
        
        # Start the simulation
        try:
            net = simulator.start()
            
            # Generate some traffic between random nodes
            nodes = list(network.graph.nodes())
            for i in range(min(3, len(nodes))):
                source = nodes[i]
                destination = nodes[(i + len(nodes)//2) % len(nodes)]
                
                # Run an experiment
                print(f"\nRouting from {source} to {destination}:")
                result = simulator.run_experiment(source, destination, protocol='pamr', num_packets=args.packets)
                
                # Visualize the result
                if result['path']:
                    simulator.visualize_network(
                        title=f"PAMR Path from {source} to {destination}",
                        highlight_path=result['path'],
                        save_path=f"mininet_results/path_{source}_to_{destination}.png"
                    )
            
            # Allow user to interact with the network
            simulator.run_cli()
            
        finally:
            # Stop the simulation
            simulator.stop()
            
        print("Mininet simulation complete. Check 'mininet_results' directory for visualization results.")
        return
    
    # Otherwise, run the original visualization
    # ... existing code ...

if __name__ == '__main__':
    main()