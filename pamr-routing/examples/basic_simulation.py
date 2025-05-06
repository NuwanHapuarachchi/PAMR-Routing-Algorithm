import sys
import os
import matplotlib.pyplot as plt
import panel as pn
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import webbrowser
import threading
import time

# Add parent directory to path so we can import the pamr package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from pamr package AFTER adding path
from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter
from pamr.simulation.simulator import PAMRSimulator
from pamr.visualization.network_viz import NetworkVisualizer
from pamr.visualization.pyviz_network import PyVizNetworkVisualizer
from pamr.visualization.metrics_viz import NetworkMetricsVisualizer

def main():
    # Always import the central network definition
    from pamr.core.network import network

    # Create router with optimized parameters
    router = PAMRRouter(network.graph, alpha=12.0, beta=3.0, gamma=21)
    
    # Create simulator
    simulator = PAMRSimulator(network, router)
    
    # Run simulation
    print("Running PAMR simulation...")
    path_history = simulator.run_simulation(num_iterations=50, packets_per_iter=30)
    
    # Create visualizer
    visualizer = NetworkVisualizer(network)
    
    # Choose a source and multiple destinations for visualization
    source = 0
    destinations = [5]  # Adding multiple destinations for comparison
    
    # Find multiple paths
    paths = []
    for dest in destinations:
        # Use Dijkstra's algorithm with modified weights to find best path
        
        def edge_weight(u, v, edge_data):
            pheromone = edge_data['pheromone']
            distance = edge_data['distance']
            congestion = edge_data['congestion']
            
            # Combined weight - lower is better
            weight = (1 / pheromone) * distance * (1 + congestion)
            return weight
        
        try:
            path = nx.shortest_path(
                network.graph, source, dest, 
                weight=edge_weight
            )
            paths.append(path)
            print(f"Best path from {source} to {dest}: {path}")
        except nx.NetworkXNoPath:
            print(f"No path found from {source} to {dest}")
    
    # Visualize the network with multiple paths using matplotlib
    fig = visualizer.visualize_network(source, None, paths)
    fig.savefig('pamr_routing_paths.png')
    
    # Choose visualization method - both can be True now
    show_interactive = True
    show_metrics = True
    
    # Generate the metrics report in a background thread so it doesn't block
    if show_metrics:
        print("Generating advanced metrics report...")
        
        # Create a metrics visualizer
        metrics_viz = NetworkMetricsVisualizer()
        
        # Extract node metrics
        node_metrics = {}
        for node in network.graph.nodes():
            # Calculate average values for adjacent edges
            neighbors = list(network.graph.neighbors(node))
            node_metrics[node] = {
                'node_id': node,
                'degree': network.graph.degree(node),
                'centrality': nx.degree_centrality(network.graph)[node],
                'avg_pheromone': np.mean([network.graph[node][neighbor]['pheromone'] 
                                         for neighbor in neighbors]) if neighbors else 0,
                'avg_congestion': np.mean([network.graph[node][neighbor]['congestion'] 
                                          for neighbor in neighbors]) if neighbors else 0,
                'traffic_handled': sum([network.graph[node][neighbor]['traffic'] 
                                      for neighbor in neighbors]) if neighbors else 0
            }
        
        # Convert to pandas DataFrame
        node_df = pd.DataFrame.from_dict(node_metrics, orient='index')
        
        # Extract path metrics from the simulation - FIX HERE
        path_metrics = []
        for iter_idx, iter_paths in enumerate(path_history):
            for item in iter_paths:
                # Try to extract path and quality based on common formats
                path = None
                quality = None
                src = None
                dest = None
                
                # Handle different formats of data in iter_paths
                if isinstance(item, tuple) or isinstance(item, list):
                    if len(item) == 2:
                        # Format is (path, quality)
                        path, quality = item
                    elif len(item) == 4:
                        # Format is (source, destination, path, quality)
                        src, dest, path, quality = item
                    elif len(item) == 3:
                        # Format is (source, destination, path)
                        src, dest, path = item
                        quality = 0  # Default quality
                else:
                    continue
                
                # If path wasn't extracted or is empty, skip this item
                if not path or len(path) < 2:
                    continue
                    
                # If source/destination weren't extracted, get them from the path
                if src is None:
                    src = path[0]
                if dest is None:
                    dest = path[-1]
                
                # Calculate path metrics
                total_distance = 0
                total_pheromone = 0
                max_congestion = 0
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    total_distance += network.graph[u][v]['distance']
                    total_pheromone += network.graph[u][v]['pheromone']
                    max_congestion = max(max_congestion, network.graph[u][v]['congestion'])
                
                path_metrics.append({
                    'source': src,
                    'destination': dest,
                    'iteration': iter_idx,
                    'path_length': len(path),
                    'path_quality': quality,
                    'total_distance': total_distance,
                    'avg_pheromone': total_pheromone / (len(path) - 1),
                    'max_congestion': max_congestion,
                    'path': '->'.join(map(str, path))
                })
        
        # Convert to pandas DataFrame
        path_df = pd.DataFrame(path_metrics) if path_metrics else pd.DataFrame()
        
        # Create time series metrics
        timestamps = pd.date_range(start='2023-01-01', periods=len(simulator.metrics['path_lengths']), freq='5min')
        time_series = pd.DataFrame({
            'timestamp': timestamps,
            'avg_path_length': simulator.metrics['path_lengths'],
            'avg_congestion': simulator.metrics.get('congestion_levels', [0] * len(simulator.metrics['path_lengths']))
        })
        
        # Set data for metrics visualizer
        metrics_viz.set_data(
            node_metrics=node_df,
            path_metrics=path_df,
            time_series_data=time_series,
            network_graph=network.graph
        )
        
        # Generate advanced report
        os.makedirs('./reports', exist_ok=True)
        report_filename = f"pamr_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = metrics_viz.generate_advanced_report(
            output_dir='./reports',
            filename=report_filename
        )
        
        full_report_path = os.path.abspath(os.path.join('./reports', report_filename))
        print(f"Advanced metrics report generated: {full_report_path}")
        
        # Open the report in a new browser tab
        def open_report_browser():
            time.sleep(1)  # Small delay to ensure file is written
            webbrowser.open(f'file://{full_report_path}', new=2)
            print(f"Metrics report opened in new browser tab.")
        
        # Start a thread to open the report
        if report_path:
            report_thread = threading.Thread(target=open_report_browser)
            report_thread.daemon = True
            report_thread.start()
    
    # Analyze path selection in detail
    if show_metrics:
        print("Analyzing path selection decisions...")
        
        # Import the path analyzer
        from pamr.utils.path_analyzer import PathAnalyzer
        
        # Create analyzer instance
        path_analyzer = PathAnalyzer(network, router)
        
        # Analyze each destination path
        analysis_reports = []
        for i, dest in enumerate(destinations):
            if i < len(paths):
                # Generate detailed analysis report
                print(f"Analyzing path from {source} to {dest}...")
                report_files = path_analyzer.analyze_path(source, dest)
                analysis_reports.append(report_files)
                
                # Open the HTML report in browser
                webbrowser.open(f'file://{os.path.abspath(report_files["html_report"])}', new=2)
                
                print(f"Path analysis saved to: {report_files['html_report']}")
                print(f"Detailed metrics saved to: {report_files['csv_data']}")
    
    # Now run the interactive visualization separately
    if show_interactive:
        print("Setting up interactive network visualization...")
        
        # Create PyViz visualization in the main thread
        pyviz_visualizer = PyVizNetworkVisualizer(network)
        
        # Create interactive visualization with ALL paths
        network_viz = pyviz_visualizer.visualize_network(source, destinations, paths)
        
        # Save interactive visualization to HTML
        interactive_html = 'pamr_interactive_network.html' 
        network_viz.save(interactive_html)
        print(f"Interactive visualization saved to '{interactive_html}'")

        # Launch the interactive visualization server in the main thread
        print("Launching interactive network visualization server... (Press Ctrl+C to exit)")
        try:
            pn.serve(network_viz, start=True, show=True, port=5006)
        except KeyboardInterrupt:
            print("\nVisualization server stopped.")
        except Exception as e:
            print(f"\nError in interactive visualization: {str(e)}")
            # Fallback to opening static HTML
            full_path = os.path.abspath(interactive_html)
            print(f"Opening static visualization file at: {full_path}")
            webbrowser.open(f'file://{full_path}', new=1)
    
    print("Simulation completed. Visualizations saved to files.")

if __name__ == "__main__":
    main()
