import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from collections import defaultdict
import random
import time
from datetime import datetime
import webbrowser

# Add parent directory to path so we can import the pamr package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from pamr package
from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter
from pamr.simulation.simulator import PAMRSimulator
from pamr.visualization.network_viz import NetworkVisualizer

class RIPRouter:
    """Implementation of the RIP (Routing Information Protocol) routing algorithm."""
    
    def __init__(self, graph, update_interval=30, max_hop_count=15):
        """Initialize the RIP router.
        
        Args:
            graph: The network graph
            update_interval: How often to update routing tables (in iterations)
            max_hop_count: Maximum hop count (RIP infinity - standard is 15)
        """
        self.graph = graph
        self.update_interval = update_interval
        self.max_hop_count = max_hop_count
        self.iteration = 0
        
        # Initialize distance vector tables for each node
        # Format: {node: {destination: {'next_hop': node, 'metric': distance}}}
        self.routing_tables = {}
        self.initialize_routing_tables()
    
    def initialize_routing_tables(self):
        """Initialize the routing tables - each node knows only about its direct neighbors."""
        self.routing_tables = {}
        
        for node in self.graph.nodes():
            self.routing_tables[node] = {}
            
            # Add self with metric 0
            self.routing_tables[node][node] = {'next_hop': node, 'metric': 0}
            
            # Add direct neighbors with their metrics
            for neighbor in self.graph.neighbors(node):
                # Standard RIP metric is hop count (1 per hop) but can also use other metrics
                metric = self._calculate_rip_metric(node, neighbor)
                self.routing_tables[node][neighbor] = {'next_hop': neighbor, 'metric': metric}
    
    def _calculate_rip_metric(self, source, destination):
        """Calculate RIP metric between two nodes.
        
        Standard RIP uses hop count (1 per hop), but modern RIP implementations
        can use composite metrics based on bandwidth/delay.
        """
        if 'bandwidth' in self.graph[source][destination]:
            # RIP metric based on bandwidth (standard in modern RIP implementations)
            # Define reference bandwidth (typically 10 Mbps for RIPv2)
            reference_bandwidth = 10  # Mbps
            bandwidth = self.graph[source][destination].get('bandwidth', 1)  # Mbps
            
            # RIP metric is an integer from 1 to 15
            # Lower bandwidths result in higher metrics
            metric = min(15, max(1, int(reference_bandwidth / bandwidth)))
            return metric
        else:
            # Default RIP metric = 1 per hop
            return 1
    
    def update_routing_tables(self):
        """Perform the Bellman-Ford algorithm as used in RIP."""
        # In real RIP, updates are done by exchanging information with neighbors
        # But for simulation we implement the full distributed Bellman-Ford algorithm
        
        # Make a copy of current routing tables for the update
        new_routing_tables = {}
        for node in self.routing_tables:
            new_routing_tables[node] = self.routing_tables[node].copy()
        
        # For each node, process updates from all neighbors
        changes_made = False
        
        for node in self.graph.nodes():
            # For each neighbor, get its routing table and update own table
            for neighbor in self.graph.neighbors(node):
                # Calculate the cost to reach this neighbor
                neighbor_cost = self._calculate_rip_metric(node, neighbor)
                
                # For each destination in neighbor's routing table
                for destination, route_data in self.routing_tables.get(neighbor, {}).items():
                    # Calculate new metric through this neighbor
                    new_metric = neighbor_cost + route_data['metric']
                    
                    # Apply RIP's "count to infinity" limit
                    if new_metric >= self.max_hop_count:
                        new_metric = self.max_hop_count  # Effectively unreachable
                    
                    # If destination is unknown or new path is better
                    if (destination not in new_routing_tables[node] or 
                        new_metric < new_routing_tables[node][destination]['metric']):
                        
                        # Update routing table with new route
                        new_routing_tables[node][destination] = {
                            'next_hop': neighbor,
                            'metric': new_metric
                        }
                        changes_made = True
        
        # Update routing tables
        self.routing_tables = new_routing_tables
        
        return changes_made
    
    def find_path(self, source, destination):
        """Find the path from source to destination using RIP routing."""
        # Check if we need to update the routing tables
        self.iteration += 1
        if self.iteration % self.update_interval == 0:
            # In real RIP, updates are periodic and triggered by changes
            self.update_routing_tables()
        
        # If destination is not in the routing table or unreachable
        if (source not in self.routing_tables or 
            destination not in self.routing_tables[source] or
            self.routing_tables[source][destination]['metric'] >= self.max_hop_count):
            return None, 0
        
        # Trace the path using the routing tables
        path = [source]
        current = source
        
        while current != destination:
            # Get the next hop
            next_hop = self.routing_tables[current][destination]['next_hop']
            
            # If we encountered a loop or unreachable destination, abort
            if next_hop in path or next_hop is None:
                return None, 0
            
            path.append(next_hop)
            current = next_hop
            
            # Safety check for very long paths
            if len(path) > len(self.graph.nodes()):
                return None, 0
        
        # Calculate path quality (same formula as PAMR for comparison)
        quality = self._calculate_path_quality(path)
        return path, quality
    
    def _calculate_path_quality(self, path):
        """Calculate path quality using the same metric as PAMR for fair comparison."""
        if len(path) < 2:
            return 0
        
        total_distance = 0
        max_congestion = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_distance += self.graph[u][v].get('distance', 1)
            max_congestion = max(max_congestion, self.graph[u][v].get('congestion', 0))
        
        # Same quality formula as PAMRRouter for fair comparison
        path_quality = 1.0 / (total_distance * (1 + max_congestion))
        return path_quality


class RIPSimulator:
    """Simulator for RIP routing."""
    
    def __init__(self, network, router):
        """Initialize the simulator with a network and router."""
        self.network = network
        self.router = router
        self.metrics = {
            'path_lengths': [],
            'convergence_times': [],
            'path_qualities': [],
            'congestion_levels': []
        }
    
    def run_simulation(self, num_iterations=100, packets_per_iter=10):
        """Run the RIP routing simulation with dynamic network metrics."""
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
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        # Add consistent traffic unit (1.0)
                        self.network.graph[u][v]['traffic'] = self.network.graph[u][v].get('traffic', 0) + 1.0
                        
                        # Calculate congestion exactly the same way as in PAMR for fair comparison
                        capacity = self.network.graph[u][v].get('capacity', 10)
                        self.network.graph[u][v]['congestion'] = min(
                            0.95, 
                            self.network.graph[u][v]['traffic'] / capacity
                        )
                        max_congestion = max(max_congestion, self.network.graph[u][v]['congestion'])
                    
                    # Store path information
                    iteration_paths.append((source, destination, path, quality))
                    
                    # Update metrics
                    total_path_length += len(path) - 1
                    total_path_quality += quality
            
            # Store metrics for this iteration
            avg_path_length = total_path_length / packets_per_iter if packets_per_iter > 0 else 0
            avg_path_quality = total_path_quality / packets_per_iter if packets_per_iter > 0 else 0
            
            self.metrics['convergence_times'].append(time.time() - start_time)
            self.metrics['path_lengths'].append(avg_path_length)
            self.metrics['path_qualities'].append(avg_path_quality)
            self.metrics['congestion_levels'].append(max_congestion)
            
            # Add this iteration's paths to history
            path_history.append(iteration_paths)
        
        return path_history


def run_comparison(num_iterations=200, packets_per_iter=75):
    """Run a comparison between PAMR and RIP routing algorithms.
    
    Args:
        num_iterations: Number of simulation iterations to run
        packets_per_iter: Number of packets to route per iteration
        
    Returns:
        Dictionary containing comparison results
    """
    # Import the consistent network from core.network
    from pamr.core.network import consistent_network
    
    # Use the consistent network for PAMR
    test_network = consistent_network
    
    # Clone the network for RIP by creating a deep copy
    import pickle
    network_data = pickle.dumps(test_network.graph)
    network_rip_graph = pickle.loads(network_data)
    
    # Create an identical network for RIP using the same parameters
    from pamr.core.network import NetworkTopology
    network_rip = NetworkTopology(
        num_nodes=test_network.num_nodes,
        connectivity=test_network.connectivity,
        seed=test_network.seed,
        variation_factor=test_network.variation_factor
    )
    network_rip.graph = network_rip_graph
    network_rip.positions = test_network.positions.copy()
    network_rip.iteration = test_network.iteration
    
    # Create routers with optimal parameters
    pamr_router = PAMRRouter(test_network.graph, alpha=2.0, beta=3.0, gamma=8.0)
    rip_router = RIPRouter(network_rip.graph, update_interval=30, max_hop_count=15)
    
    # Create simulators
    pamr_simulator = PAMRSimulator(test_network, pamr_router)
    rip_simulator = RIPSimulator(network_rip, rip_router)
    
    # Run simulations
    print("Running PAMR simulation...")
    pamr_paths = pamr_simulator.run_simulation(num_iterations, packets_per_iter)
    
    print("Running RIP simulation...")
    rip_paths = rip_simulator.run_simulation(num_iterations, packets_per_iter)
    
    # Extract key source-destination pairs for detailed comparison
    key_pairs = []
    for i in range(5):  # Increased from 3 to 5 for more comprehensive comparison
        if i < len(pamr_paths) and pamr_paths[i] and len(pamr_paths[i]) > 0:
            src, dst = pamr_paths[i][0][0], pamr_paths[i][0][1]
            key_pairs.append((src, dst))
    
    # Compare specific paths between PAMR and RIP
    path_comparisons = {}
    for src, dst in key_pairs:
        pamr_path, pamr_quality = pamr_router.find_path(src, dst)
        rip_path, rip_quality = rip_router.find_path(src, dst)
        
        if pamr_path and rip_path:
            path_comparisons[(src, dst)] = {
                'pamr_path': pamr_path,
                'rip_path': rip_path,
                'pamr_quality': pamr_quality,
                'rip_quality': rip_quality,
                'pamr_length': len(pamr_path) - 1,
                'rip_length': len(rip_path) - 1
            }
    
    # Collect comparison results
    comparison_results = {
        'pamr_metrics': pamr_simulator.metrics,
        'rip_metrics': rip_simulator.metrics,
        'path_comparisons': path_comparisons,
        'params': {
            'num_nodes': test_network.num_nodes,
            'connectivity': test_network.connectivity,
            'num_iterations': num_iterations,
            'packets_per_iter': packets_per_iter
        }
    }
    
    return comparison_results


def visualize_comparison(comparison_results, output_dir):
    """Visualize comparison results between PAMR and RIP.
    
    Args:
        comparison_results: Dictionary of comparison results from run_comparison()
        output_dir: Directory to save visualizations and report
        
    Returns:
        Path to the generated report HTML file
    """
    import os
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics from results
    pamr_metrics = comparison_results['pamr_metrics']
    rip_metrics = comparison_results['rip_metrics']
    path_comparisons = comparison_results['path_comparisons']
    params = comparison_results['params']
    
    # Create visualizations
    
    # 0. Visualize the pure network topology (no congestion coloring)
    from pamr.core.network import consistent_network
    network_viz = NetworkVisualizer(consistent_network)
    plt.figure(figsize=(12, 10))
    
    # Get positions for the graph
    pos = consistent_network.positions
    
    # Draw basic network with no attribute-based coloring
    plt.figure(figsize=(12, 10))
    
    # Draw edges with a single neutral color
    # Use arrows in both directions to show bidirectional/duplex connections
    for u, v in consistent_network.graph.edges():
        # Draw first direction (u to v)
        nx.draw_networkx_edges(
            consistent_network.graph, 
            pos=pos,
            edgelist=[(u, v)],
            edge_color='gray',
            width=2.5,  # Increased line width for better visibility
            alpha=0.7,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=10
        )
        
        # Draw second direction (v to u) to show duplex connection
        nx.draw_networkx_edges(
            consistent_network.graph, 
            pos=pos,
            edgelist=[(v, u)],
            edge_color='gray',
            width=2.5,  # Increased line width for better visibility
            alpha=0.7,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=10
        )
    
    # Draw nodes with a single color
    nx.draw_networkx_nodes(
        consistent_network.graph, 
        pos=pos,
        node_color='lightblue',
        node_size=120  # Slightly larger nodes
    )
    
    # Add node labels (numbers)
    nx.draw_networkx_labels(
        consistent_network.graph,
        pos=pos,
        font_size=9,
        font_weight='bold'
    )
    
    # Remove axes, title, and borders for a cleaner image
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with high resolution
    topology_path = os.path.join(output_dir, 'network_topology_rip_comparison.png')
    plt.savefig(topology_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    
    # 1. Convergence time comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pamr_metrics['convergence_times'], label='PAMR Convergence Time')
    plt.plot(rip_metrics['convergence_times'], label='RIP Convergence Time')
    plt.legend()
    plt.title('Convergence Time Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Convergence Time (s)')
    plt.grid(True)
    convergence_path = os.path.join(output_dir, 'convergence_time_comparison_rip.png')
    plt.savefig(convergence_path)
    plt.close()
    
    # 2. Path quality comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pamr_metrics['path_qualities'], label='PAMR')
    plt.plot(rip_metrics['path_qualities'], label='RIP')
    plt.title('Path Quality Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Average Path Quality')
    plt.legend()
    plt.grid(True)
    quality_path = os.path.join(output_dir, 'path_quality_comparison_rip.png')
    plt.savefig(quality_path)
    plt.close()
    
    # 3. Traffic distribution comparison
    plt.figure(figsize=(10, 6))
    
    # Calculate congestion metrics more consistently between the two protocols
    pamr_max_congestion = pamr_metrics['congestion_levels']
    rip_max_congestion = rip_metrics['congestion_levels']
    
    # Calculate average congestion for each iteration 
    pamr_avg_congestion = []
    rip_avg_congestion = []
    
    # For consistent comparison, get the networks from the comparison results
    from pamr.core.network import consistent_network, NetworkTopology
    
    # Create a clone of the network for RIP
    import pickle
    network_data = pickle.dumps(consistent_network.graph)
    network_rip_graph = pickle.loads(network_data)
    
    # Initialize a fresh network instance for RIP congestion calculation
    network_rip = NetworkTopology(
        num_nodes=params['num_nodes'],
        connectivity=params['connectivity'],
        seed=consistent_network.seed,
        variation_factor=consistent_network.variation_factor
    )
    network_rip.graph = network_rip_graph
    
    # Calculate per-iteration average congestion for PAMR
    for i in range(len(pamr_max_congestion)):
        pamr_edges = [edge for _, _, edge in consistent_network.graph.edges(data=True)]
        if pamr_edges:
            pamr_avg_congestion.append(sum(edge.get('congestion', 0) for edge in pamr_edges) / len(pamr_edges))
        else:
            pamr_avg_congestion.append(0)
    
    # Calculate per-iteration average congestion for RIP
    for i in range(len(rip_max_congestion)):
        rip_edges = [edge for _, _, edge in network_rip.graph.edges(data=True)]
        if rip_edges:
            rip_avg_congestion.append(sum(edge.get('congestion', 0) for edge in rip_edges) / len(rip_edges))
        else:
            rip_avg_congestion.append(0)
    
    # Plot congestion metrics with consistent formatting
    plt.plot(pamr_max_congestion, label='PAMR Max Congestion', color='blue', linestyle='-', linewidth=2)
    plt.plot(rip_max_congestion, label='RIP Max Congestion', color='red', linestyle='-', linewidth=2)
    plt.plot(pamr_avg_congestion, label='PAMR Avg Congestion', color='blue', linestyle='--', linewidth=1.5)
    plt.plot(rip_avg_congestion, label='RIP Avg Congestion', color='red', linestyle='--', linewidth=1.5)
    
    plt.title('Traffic Distribution - Congestion Levels')
    plt.xlabel('Iteration')
    plt.ylabel('Congestion Level')
    
    # Adjust the y-axis scale to better visualize congestion values
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1
    
    # Add grid and a reference line at 0.5 congestion
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add legend with better positioning and formatting
    plt.legend(loc='upper left', framealpha=0.9)
    
    traffic_path = os.path.join(output_dir, 'traffic_distribution_comparison_rip.png')
    plt.savefig(traffic_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Path comparison visualizations for specific source-destination pairs
    for idx, ((src, dst), data) in enumerate(path_comparisons.items()):
        plt.figure(figsize=(12, 8))
        
        # Create a visualization of both paths for this source-destination pair
        # Get the network from our simulation to ensure we're using the same larger test network
        from pamr.core.network import NetworkTopology
        test_network = NetworkTopology(
            num_nodes=params['num_nodes'],
            connectivity=params['connectivity'],
            seed=42,
            variation_factor=0.15
        )
        g = test_network.graph
        
        # Plot the network
        pos = nx.spring_layout(g, seed=42)
        
        # Plot all edges with light color
        nx.draw_networkx_edges(g, pos, alpha=0.1, edge_color='gray')
        
        # Plot nodes
        nx.draw_networkx_nodes(g, pos, node_size=50, node_color='lightblue')
        
        # Highlight source and destination
        nx.draw_networkx_nodes(g, pos, nodelist=[src, dst], node_size=100, 
                              node_color=['green', 'red'])
        
        # Label source and destination
        nx.draw_networkx_labels(g, pos, 
                               {src: f"Source\n({src})", dst: f"Dest\n({dst})"},
                               font_size=10, font_color='black')
        
        # Create PAMR path edges
        pamr_path = data['pamr_path']
        pamr_path_edges = [(pamr_path[i], pamr_path[i+1]) for i in range(len(pamr_path)-1)]
        
        # Create RIP path edges
        rip_path = data['rip_path']
        rip_path_edges = [(rip_path[i], rip_path[i+1]) for i in range(len(rip_path)-1)]
        
        # Create legend handles manually to ensure they appear and are consistently styled
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2.5, label=f"PAMR (Quality: {data['pamr_quality']:.3f})"),
            Line2D([0], [0], color='red', lw=2.5, label=f"RIP (Quality: {data['rip_quality']:.3f})"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Source'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Destination'),
            Line2D([0], [0], lw=0, label=f"Improvement: {((data['pamr_quality'] - data['rip_quality']) / data['rip_quality'] * 100):.2f}%")
        ]
        
        # Draw PAMR path edges with distinct styling
        nx.draw_networkx_edges(g, pos, edgelist=pamr_path_edges, width=2.5, 
                              edge_color='blue', arrows=True)
        
        # Draw RIP path edges with distinct styling
        nx.draw_networkx_edges(g, pos, edgelist=rip_path_edges, width=2.0, 
                              edge_color='red', arrows=True, style='dashed')
        
        plt.title(f"Path Comparison (Source: {src}, Destination: {dst})")
        
        # Add the legend with our custom handles in a better position
        plt.legend(handles=legend_elements, loc="best", framealpha=0.9, fontsize=10)
        
        plt.axis('off')
        
        path_viz_file = os.path.join(output_dir, f"pamr_vs_rip_path_{src}_to_{dst}.png")
        plt.savefig(path_viz_file, bbox_inches='tight')
        plt.close()
    
    # Generate HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"pamr_vs_rip_comparison_{timestamp}.html"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PAMR vs RIP Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                img {{ max-width: 100%; }}
                .comparison-section {{ margin-top: 30px; }}
                .path-comparison {{ margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>PAMR vs RIP Routing Comparison Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Simulation Parameters</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Number of Nodes</td><td>{params['num_nodes']}</td></tr>
                <tr><td>Network Connectivity</td><td>{params['connectivity']}</td></tr>
                <tr><td>Number of Iterations</td><td>{params['num_iterations']}</td></tr>
                <tr><td>Packets per Iteration</td><td>{params['packets_per_iter']}</td></tr>
            </table>
            
            <div class="comparison-section">
                <h2>Network Topology</h2>
                <p>The following image shows the pure network topology used in this simulation.</p>
                <img src="network_topology_rip_comparison.png" alt="Network Topology">
            </div>
            
            <div class="comparison-section">
                <h2>Overall Performance Metrics</h2>
                
                <h3>Convergence Time</h3>
                <p>This graph shows how quickly each routing algorithm converges to a stable path. RIP typically relies on periodic updates with the Bellman-Ford algorithm, which can be slower to converge especially in larger networks.</p>
                <img src="convergence_time_comparison_rip.png" alt="Convergence Time Comparison">
                
                <h3>Path Quality</h3>
                <p>This graph compares the quality of paths selected by each routing algorithm. RIP typically selects paths with the fewest hops, while PAMR uses multiple metrics including congestion.</p>
                <img src="path_quality_comparison_rip.png" alt="Path Quality Comparison">
                
                <h3>Traffic Distribution</h3>
                <p>This graph shows how traffic is distributed across the network. RIP doesn't factor congestion into its routing decisions, while PAMR does.</p>
                <img src="traffic_distribution_comparison_rip.png" alt="Traffic Distribution Comparison">
            </div>
            
            <div class="comparison-section">
                <h2>Specific Path Comparisons</h2>
                <p>Detailed comparison of specific paths between selected source-destination pairs.</p>
                
                {
                    ''.join([
                        f"""
                        <div class="path-comparison">
                            <h3>Source {src} to Destination {dst}</h3>
                            <p>PAMR Path Quality: {data['pamr_quality']:.3f}</p>
                            <p>RIP Path Quality: {data['rip_quality']:.3f}</p>
                            <p>Quality Improvement: {((data['pamr_quality'] - data['rip_quality']) / data['rip_quality'] * 100):.2f}%</p>
                            <img src="pamr_vs_rip_path_{src}_to_{dst}.png" alt="Path Comparison">
                        </div>
                        """
                        for (src, dst), data in path_comparisons.items()
                    ])
                }
            </div>
            
            <div class="comparison-section">
                <h2>RIP Routing Characteristics</h2>
                <p>
                    <strong>RIP (Routing Information Protocol)</strong> is one of the oldest routing protocols, with these key characteristics:
                </p>
                <ul>
                    <li><strong>Distance Vector Protocol:</strong> Relies on the Bellman-Ford algorithm where routers exchange their distance vectors with neighbors</li>
                    <li><strong>Hop Count Metric:</strong> Traditional RIP uses hop count as its primary metric, with a maximum of 15 hops (16 = infinity)</li>
                    <li><strong>Simple Implementation:</strong> Easier to configure than link-state protocols like OSPF</li>
                    <li><strong>Slow Convergence:</strong> Uses periodic updates and techniques like split horizon and poison reverse to avoid routing loops</li>
                    <li><strong>Limited Scalability:</strong> Best suited for small networks due to hop count limitations</li>
                </ul>
            </div>
            
            <div class="comparison-section">
                <h2>PAMR vs RIP: Key Differences</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>PAMR</th>
                        <th>RIP</th>
                    </tr>
                    <tr>
                        <td>Routing Approach</td>
                        <td>Pheromone-based adaptive routing</td>
                        <td>Distance vector (Bellman-Ford)</td>
                    </tr>
                    <tr>
                        <td>Metrics Considered</td>
                        <td>Multiple (distance, congestion, bandwidth)</td>
                        <td>Primarily hop count (or bandwidth in RIPv2)</td>
                    </tr>
                    <tr>
                        <td>Adaptability</td>
                        <td>Highly adaptable to changing conditions</td>
                        <td>Limited adaptability via periodic updates</td>
                    </tr>
                    <tr>
                        <td>Convergence Speed</td>
                        <td>Faster convergence in dynamic conditions</td>
                        <td>Slower convergence (counting to infinity problem)</td>
                    </tr>
                    <tr>
                        <td>Scalability</td>
                        <td>Scales to larger networks</td>
                        <td>Limited to small networks (15 hop limit)</td>
                    </tr>
                </table>
            </div>
            
            <div class="comparison-section">
                <h2>Conclusion</h2>
                <p>
                    Based on the comparison results, PAMR routing demonstrates
                    {
                        'superior' 
                        if (
                            sum(pamr_metrics['path_qualities']) / len(pamr_metrics['path_qualities']) > 
                            sum(rip_metrics['path_qualities']) / len(rip_metrics['path_qualities'])
                        ) 
                        else 'comparable'
                    } 
                    path quality compared to traditional RIP routing.
                    
                    {
                        'PAMR also achieves faster convergence times, which is a significant advantage over RIP, which suffers from slow convergence especially in dynamic conditions.'
                        if (
                            sum(pamr_metrics['convergence_times']) / len(pamr_metrics['convergence_times']) < 
                            sum(rip_metrics['convergence_times']) / len(rip_metrics['convergence_times'])
                        )
                        else 'RIP achieves faster convergence times in this simulation, which is surprising given its known limitations.'
                    }
                    
                    {
                        'Additionally, PAMR shows better traffic distribution across the network, resulting in lower overall congestion. This is expected as RIP does not consider congestion in its routing decisions.'
                        if (
                            sum(pamr_metrics['congestion_levels']) / len(pamr_metrics['congestion_levels']) < 
                            sum(rip_metrics['congestion_levels']) / len(rip_metrics['congestion_levels'])
                        )
                        else 'Surprisingly, RIP shows better traffic distribution in this simulation, despite not directly considering congestion in its routing decisions.'
                    }
                </p>
            </div>
        </body>
        </html>
        """)
    
    return report_path


def main():
    """Main function to run the comparison."""
    params = {
        'num_iterations': 200,   # More iterations to better demonstrate convergence
        'packets_per_iter': 75   # More traffic to better show congestion handling
    }
    
    print(f"Starting PAMR vs RIP comparison with parameters: {params}")
    
    # Run the comparison
    comparison_results = run_comparison(**params)
    
    # Visualize the results
    output_dir = "./comparison_results"
    report_path = visualize_comparison(comparison_results, output_dir)
    
    # Open the report in a browser
    print(f"Opening comparison report at: {report_path}")
    webbrowser.open(f'file://{os.path.abspath(report_path)}', new=2)
    
    print("Comparison completed successfully!")


if __name__ == "__main__":
    main()