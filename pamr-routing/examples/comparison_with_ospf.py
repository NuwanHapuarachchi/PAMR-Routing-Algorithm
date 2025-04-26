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

class OSPFRouter:
    """Simplified simulation of OSPF routing algorithm."""
    
    def __init__(self, graph, link_state_interval=10):
        """Initialize the OSPF router.
        
        Args:
            graph: The network graph
            link_state_interval: How often to update link state (in iterations)
        """
        self.graph = graph
        self.link_state_interval = link_state_interval
        self.iteration = 0
        self.routing_tables = {}
        self.update_link_state()
    
    def update_link_state(self):
        """Update the link state database (recalculate shortest paths)."""
        self.routing_tables = {}
        
        # For each node, calculate shortest paths to all other nodes
        for source in self.graph.nodes():
            # Calculate shortest paths based on current link costs
            shortest_paths = nx.single_source_dijkstra_path(
                self.graph, 
                source, 
                weight=self._ospf_link_cost
            )
            self.routing_tables[source] = shortest_paths
    
    def _ospf_link_cost(self, u, v, edge_data):
        """Calculate OSPF link cost based on distance and congestion."""
        # Standard OSPF cost is 100/bandwidth, but we'll use distance and congestion
        base_cost = edge_data['distance']
        congestion_factor = 1 + edge_data['congestion'] * 5  # Congestion has higher impact in OSPF
        return base_cost * congestion_factor
    
    def find_path(self, source, destination):
        """Find the path from source to destination using OSPF routing."""
        # Check if we need to update the link state
        self.iteration += 1
        if self.iteration % self.link_state_interval == 0:
            self.update_link_state()
        
        # Get the path from routing table
        if source in self.routing_tables and destination in self.routing_tables[source]:
            path = self.routing_tables[source][destination]
            
            # Calculate path quality (same formula as PAMR for comparison)
            quality = self._calculate_path_quality(path)
            return path, quality
        
        return None, 0
    
    def _calculate_path_quality(self, path):
        """Calculate path quality using the same metric as PAMR."""
        if len(path) < 2:
            return 0
        
        total_distance = 0
        max_congestion = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_distance += self.graph[u][v]['distance']
            max_congestion = max(max_congestion, self.graph[u][v]['congestion'])
        
        # Same quality formula as PAMRRouter for fair comparison
        path_quality = 1.0 / (total_distance * (1 + max_congestion))
        return path_quality


class OSPFSimulator:
    """Simulator for OSPF routing."""
    
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
        """Run the OSPF routing simulation with dynamic network metrics."""
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
                        self.network.graph[u][v]['traffic'] += 1.0
                        self.network.graph[u][v]['congestion'] = min(
                            0.95, 
                            self.network.graph[u][v]['traffic'] / self.network.graph[u][v]['capacity']
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


def run_comparison(num_nodes=25, connectivity=0.3, num_iterations=150, packets_per_iter=50):
    """Run a comparison between PAMR and OSPF routing algorithms.
    
    Args:
        num_nodes: Number of nodes in the network
        connectivity: Connectivity probability between nodes
        num_iterations: Number of simulation iterations to run
        packets_per_iter: Number of packets to route per iteration
        
    Returns:
        Dictionary containing comparison results
    """
    # Import the singleton network instance from core.network
    from pamr.core.network import network
    
    # Create two identical networks for PAMR and OSPF by cloning the main network
    # This ensures both algorithms use exactly the same network topology and conditions
    network_pamr = network
    
    # Clone the network for OSPF by creating a deep copy
    import pickle
    network_data = pickle.dumps(network_pamr.graph)
    network_ospf_graph = pickle.loads(network_data)
    
    # Create a new NetworkTopology with the same parameters but use the cloned graph
    from pamr.core.network import NetworkTopology
    network_ospf = NetworkTopology(
        num_nodes=network.num_nodes,
        connectivity=network.connectivity,
        seed=network.seed,
        variation_factor=network.variation_factor
    )
    network_ospf.graph = network_ospf_graph
    network_ospf.positions = network_pamr.positions.copy()
    network_ospf.iteration = network_pamr.iteration
    
    # Create routers with optimal parameters
    pamr_router = PAMRRouter(network_pamr.graph, alpha=2.0, beta=3.0, gamma=2.5)
    ospf_router = OSPFRouter(network_ospf.graph)
    
    # Create simulators
    pamr_simulator = PAMRSimulator(network_pamr, pamr_router)
    ospf_simulator = OSPFSimulator(network_ospf, ospf_router)
    
    # Run simulations
    print("Running PAMR simulation...")
    pamr_paths = pamr_simulator.run_simulation(num_iterations, packets_per_iter)
    
    print("Running OSPF simulation...")
    ospf_paths = ospf_simulator.run_simulation(num_iterations, packets_per_iter)
    
    # Extract key source-destination pairs for detailed comparison
    key_pairs = []
    for i in range(3):  # Select 3 source-destination pairs
        if i < len(pamr_paths) and pamr_paths[i]:
            src, dst = pamr_paths[i][0][0], pamr_paths[i][0][1]
            key_pairs.append((src, dst))
    
    # Compare specific paths between PAMR and OSPF
    path_comparisons = {}
    for src, dst in key_pairs:
        pamr_path, pamr_quality = pamr_router.find_path(src, dst)
        ospf_path, ospf_quality = ospf_router.find_path(src, dst)
        
        path_comparisons[(src, dst)] = {
            'pamr_path': pamr_path,
            'ospf_path': ospf_path,
            'pamr_quality': pamr_quality,
            'ospf_quality': ospf_quality
        }
    
    # Collect comparison results
    comparison_results = {
        'pamr_metrics': pamr_simulator.metrics,
        'ospf_metrics': ospf_simulator.metrics,
        'path_comparisons': path_comparisons,
        'params': {
            'num_nodes': network.num_nodes,  # Use actual network params instead of function args
            'connectivity': network.connectivity,
            'num_iterations': num_iterations,
            'packets_per_iter': packets_per_iter
        }
    }
    
    return comparison_results


def visualize_comparison(comparison_results, output_dir):
    """Visualize comparison results between PAMR and OSPF.
    
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
    ospf_metrics = comparison_results['ospf_metrics']
    path_comparisons = comparison_results['path_comparisons']
    params = comparison_results['params']
    
    # Create visualizations
    
    # 1. Convergence time comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pamr_metrics['convergence_times'], label='PAMR')
    plt.plot(ospf_metrics['convergence_times'], label='OSPF')
    plt.title('Convergence Time Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Convergence Time (s)')
    plt.legend()
    plt.grid(True)
    convergence_path = os.path.join(output_dir, 'convergence_time_comparison.png')
    plt.savefig(convergence_path)
    plt.close()
    
    # 2. Path quality comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pamr_metrics['path_qualities'], label='PAMR')
    plt.plot(ospf_metrics['path_qualities'], label='OSPF')
    plt.title('Path Quality Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Average Path Quality')
    plt.legend()
    plt.grid(True)
    quality_path = os.path.join(output_dir, 'path_quality_comparison.png')
    plt.savefig(quality_path)
    plt.close()
    
    # 3. Traffic distribution comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pamr_metrics['congestion_levels'], label='PAMR')
    plt.plot(ospf_metrics['congestion_levels'], label='OSPF')
    plt.title('Traffic Distribution Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Congestion Level')
    plt.legend()
    plt.grid(True)
    traffic_path = os.path.join(output_dir, 'traffic_distribution_comparison.png')
    plt.savefig(traffic_path)
    plt.close()
    
    # 4. Path comparison visualizations for specific source-destination pairs
    for idx, ((src, dst), data) in enumerate(path_comparisons.items()):
        plt.figure(figsize=(12, 8))
        
        # Create a visualization of both paths for this source-destination pair
        # Get the network from somewhere in the PAMR module
        from pamr.core.network import network
        g = network.graph
        
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
        
        # Create OSPF path edges
        ospf_path = data['ospf_path']
        ospf_path_edges = [(ospf_path[i], ospf_path[i+1]) for i in range(len(ospf_path)-1)]
        
        # Draw PAMR path edges
        nx.draw_networkx_edges(g, pos, edgelist=pamr_path_edges, width=2.0, 
                              edge_color='blue', label=f"PAMR (Quality: {data['pamr_quality']:.3f})")
        
        # Draw OSPF path edges
        nx.draw_networkx_edges(g, pos, edgelist=ospf_path_edges, width=2.0, 
                              edge_color='orange', label=f"OSPF (Quality: {data['ospf_quality']:.3f})")
        
        plt.title(f"Path Comparison (Source: {src}, Destination: {dst})")
        plt.legend()
        plt.axis('off')
        
        path_viz_file = os.path.join(output_dir, f"ospf_vs_pamr_ml_path_{src}_to_{dst}.png")
        plt.savefig(path_viz_file, bbox_inches='tight')
        plt.close()
    
    # Generate HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"pamr_vs_ospf_comparison_{timestamp}.html"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PAMR vs OSPF Comparison Report</title>
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
            <h1>PAMR vs OSPF Routing Comparison Report</h1>
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
                <h2>Overall Performance Metrics</h2>
                
                <h3>Convergence Time</h3>
                <p>This graph shows how quickly each routing algorithm converges to a stable path.</p>
                <img src="convergence_time_comparison.png" alt="Convergence Time Comparison">
                
                <h3>Path Quality</h3>
                <p>This graph compares the quality of paths selected by each routing algorithm.</p>
                <img src="path_quality_comparison.png" alt="Path Quality Comparison">
                
                <h3>Traffic Distribution</h3>
                <p>This graph shows how traffic is distributed across the network.</p>
                <img src="traffic_distribution_comparison.png" alt="Traffic Distribution Comparison">
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
                            <p>OSPF Path Quality: {data['ospf_quality']:.3f}</p>
                            <p>Quality Improvement: {((data['pamr_quality'] - data['ospf_quality']) / data['ospf_quality'] * 100):.2f}%</p>
                            <img src="ospf_vs_pamr_ml_path_{src}_to_{dst}.png" alt="Path Comparison">
                        </div>
                        """
                        for (src, dst), data in path_comparisons.items()
                    ])
                }
            </div>
            
            <div class="comparison-section">
                <h2>Conclusion</h2>
                <p>
                    Based on the comparison results, PAMR routing demonstrates
                    {
                        'superior' 
                        if (
                            sum(pamr_metrics['path_qualities']) / len(pamr_metrics['path_qualities']) > 
                            sum(ospf_metrics['path_qualities']) / len(ospf_metrics['path_qualities'])
                        ) 
                        else 'comparable'
                    } 
                    path quality compared to traditional OSPF routing.
                    
                    {
                        'PAMR also achieves faster convergence times, which can be critical in dynamic network conditions.'
                        if (
                            sum(pamr_metrics['convergence_times']) / len(pamr_metrics['convergence_times']) < 
                            sum(ospf_metrics['convergence_times']) / len(ospf_metrics['convergence_times'])
                        )
                        else 'OSPF achieves faster convergence times in this simulation.'
                    }
                    
                    {
                        'Additionally, PAMR shows better traffic distribution across the network, resulting in lower overall congestion.'
                        if (
                            sum(pamr_metrics['congestion_levels']) / len(pamr_metrics['congestion_levels']) < 
                            sum(ospf_metrics['congestion_levels']) / len(ospf_metrics['congestion_levels'])
                        )
                        else 'OSPF shows better traffic distribution in this simulation.'
                    }
                </p>
            </div>
        </body>
        </html>
        """)
    
    return report_path


def main():
    """Main function to run the comparison."""
    # Define simulation parameters
    params = {
        'num_nodes': 25,
        'connectivity': 0.3,
        'num_iterations': 150,
        'packets_per_iter': 50
    }
    
    print(f"Starting PAMR vs OSPF comparison with parameters: {params}")
    
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