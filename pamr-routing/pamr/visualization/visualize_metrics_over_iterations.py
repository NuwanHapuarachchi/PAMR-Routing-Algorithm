import os
import sys
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter
from pamr.visualization.pyviz_network import PyVizNetworkVisualizer
from pamr.visualization.metrics_viz import NetworkMetricsVisualizer

def run_simulation_with_metrics_tracking(iterations=10, output_dir="reports"):
    """
    Run a PAMR simulation for multiple iterations, tracking metrics at each step.
    
    Args:
        iterations: Number of iterations to run
        output_dir: Directory to save the output reports
    
    Returns:
        Path to the generated HTML report
    """
    print(f"Running PAMR simulation with metrics tracking for {iterations} iterations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a network topology
    print("Creating network topology...")
    network = NetworkTopology(
        num_nodes=12,
        connectivity=0.3,
        seed=42,
        variation_factor=0.1
    )
    
    # Create a PAMR router with graph from the network
    router = PAMRRouter(network.graph)  # Fix: Pass the graph, not the network object
    
    # Create visualizers
    pyviz = PyVizNetworkVisualizer(network)
    metrics_viz = NetworkMetricsVisualizer()
    
    # Choose source and destination nodes for consistent path evaluation
    source = 0
    destinations = [6, 9, 11]
    
    # Table to store metrics for each iteration
    all_metrics = []
    iteration_paths = []
    
    # Run the simulation for specified number of iterations
    print(f"Starting iterations...")
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        # Update network metrics to simulate dynamic conditions
        network.update_dynamic_metrics()
        
        # Find paths to each destination
        paths = []
        metrics_for_iter = {'iteration': i+1}
        
        for dest in destinations:
            # Find path using PAMR routing
            path, quality = router.find_path(source, dest)
            paths.append(path)
            
            # Record path quality
            metrics_for_iter[f'path_quality_{source}_to_{dest}'] = quality
            metrics_for_iter[f'path_hops_{source}_to_{dest}'] = len(path) - 1
            
            # Get edge metrics for the path
            congestion_values = []
            pheromone_values = []
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                congestion_values.append(network.graph[u][v]['congestion'])
                pheromone_values.append(network.graph[u][v]['pheromone'])
            
            metrics_for_iter[f'max_congestion_{source}_to_{dest}'] = max(congestion_values) if congestion_values else 0
            metrics_for_iter[f'avg_pheromone_{source}_to_{dest}'] = sum(pheromone_values) / len(pheromone_values) if pheromone_values else 0
        
        # Get overall network metrics
        edge_metrics = {
            'avg_congestion': np.mean([network.graph[u][v]['congestion'] for u, v in network.graph.edges()]),
            'max_congestion': np.max([network.graph[u][v]['congestion'] for u, v in network.graph.edges()]),
            'avg_pheromone': np.mean([network.graph[u][v]['pheromone'] for u, v in network.graph.edges()]),
            'avg_traffic': np.mean([network.graph[u][v]['traffic'] for u, v in network.graph.edges()])
        }
        metrics_for_iter.update(edge_metrics)
        
        # Add metrics to the table
        all_metrics.append(metrics_for_iter)
        
        # Track iteration in the PyViz visualizer
        pyviz.track_iteration_metrics(i+1, paths, source, destinations)
        
        # Store paths for visualization
        iteration_paths.append(paths)
        
        # Simulate packet routing to update pheromones
        for _ in range(50):  # 50 packets per iteration
            src = random.choice(list(network.graph.nodes()))
            dst = random.choice([n for n in network.graph.nodes() if n != src])
            # Use find_path instead of route_packet
            router.find_path(src, dst)
    
    # Convert metrics to DataFrame for reporting
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, f"pamr_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")
    
    # Create comparison visualizations
    create_metrics_visualizations(metrics_df, output_dir)
    
    # Generate comprehensive HTML report with PyViz
    # Use the last set of paths for the final visualization
    print("Generating interactive HTML report...")
    dashboard, report_path = pyviz.create_comprehensive_report(
        source=source,
        destinations=destinations,
        paths=paths,  # Last iteration paths
        output_dir=output_dir
    )
    
    print(f"Report generated: {report_path}")
    return report_path

def create_metrics_visualizations(metrics_df, output_dir):
    """
    Create static visualizations of the metrics for the report.
    
    Args:
        metrics_df: DataFrame containing metrics for each iteration
        output_dir: Directory to save the visualizations
    """
    print("Creating metrics visualizations...")
    
    # 1. Create path quality comparison chart
    path_quality_cols = [col for col in metrics_df.columns if col.startswith('path_quality_')]
    
    plt.figure(figsize=(12, 6))
    for col in path_quality_cols:
        plt.plot(metrics_df['iteration'], metrics_df[col], marker='o', label=col.replace('path_quality_', ''))
    
    plt.title('Path Quality Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Path Quality')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_quality_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Create congestion comparison chart
    congestion_cols = ['avg_congestion', 'max_congestion'] + [col for col in metrics_df.columns if col.startswith('max_congestion_')]
    
    plt.figure(figsize=(12, 6))
    for col in congestion_cols:
        plt.plot(metrics_df['iteration'], metrics_df[col], marker='o', label=col.replace('max_congestion_', ''))
    
    plt.title('Congestion Levels Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Congestion')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'congestion_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Create pheromone comparison chart
    pheromone_cols = ['avg_pheromone'] + [col for col in metrics_df.columns if col.startswith('avg_pheromone_')]
    
    plt.figure(figsize=(12, 6))
    for col in pheromone_cols:
        plt.plot(metrics_df['iteration'], metrics_df[col], marker='o', label=col.replace('avg_pheromone_', ''))
    
    plt.title('Pheromone Levels Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Pheromone')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pheromone_comparison.png'), dpi=300)
    plt.close()
    
    # 4. Create a heatmap visualization of the metrics correlation
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation between numeric columns
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    corr_matrix = metrics_df[numeric_cols].corr()
    
    # Plot heatmap
    import seaborn as sns
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=.5
    )
    plt.title('Metrics Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation.png'), dpi=300)
    plt.close()
    
    # 5. Create a table summary of metrics for the 10 iterations
    plt.figure(figsize=(12, len(metrics_df) * 0.5 + 2))
    plt.axis('off')
    
    # Key metrics to include in the table
    key_metrics = ['iteration', 'avg_congestion', 'max_congestion', 'avg_pheromone']
    path_metrics = [col for col in metrics_df.columns if col.startswith('path_quality_')]
    table_cols = key_metrics + path_metrics
    
    # Create table data
    table_data = metrics_df[table_cols].values
    
    # Column labels
    col_labels = [col.replace('path_quality_', 'Path: ') for col in table_cols]
    
    # Create the table
    table = plt.table(
        cellText=np.round(table_data, 4),
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(col_labels)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Metrics Variation Over 10 Iterations', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    report_path = run_simulation_with_metrics_tracking(iterations=10)
    
    # Open the report in a browser
    import webbrowser
    import time
    time.sleep(1)  # Small delay to ensure file is written
    webbrowser.open(f'file://{report_path}', new=2)
    print(f"Report opened in browser.")