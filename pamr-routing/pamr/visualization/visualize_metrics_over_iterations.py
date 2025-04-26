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

from pamr.core.network import consistent_network  # Import the consistent network instance
from pamr.core.routing import PAMRRouter
from pamr.core.pheromone import PheromoneManager
from pamr.visualization.pyviz_network import PyVizNetworkVisualizer
from pamr.visualization.metrics_viz import NetworkMetricsVisualizer

def run_simulation_with_metrics_tracking(iterations=100, output_dir="reports"):
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
    
    # Use the consistent network from network.py
    print("Using consistent network topology...")
    network = consistent_network
    print(f"Network properties: {network.num_nodes} nodes, {network.connectivity} connectivity")
    
    # Create a PAMR router with graph from the network - use optimized parameters
    router = PAMRRouter(
        network.graph,
        alpha=2.0,   # Reduced pheromone importance to prevent path stickiness
        beta=3.0,    # Distance importance
        gamma=30.0,   # Higher congestion avoidance factor
        adapt_weights=True  # Enable adaptive weight adjustments
    )
    
    # Create pheromone manager for updating pheromones
    pheromone_manager = PheromoneManager(network.graph)
    
    # Create visualizers
    pyviz = PyVizNetworkVisualizer(network)
    metrics_viz = NetworkMetricsVisualizer()
    
    # Choose source and destination nodes for consistent path evaluation
    source = 0
    destinations = [4]
    
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
            
            # Print the selected path for this iteration
            print(f"  Path from {source} to {dest}: {path} (quality: {quality:.4f})")
            
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
        successful_paths = []  # List to collect successful paths and their qualities
        
        for _ in range(50):  # 50 packets per iteration
            src = random.choice(list(network.graph.nodes()))
            dst = random.choice([n for n in network.graph.nodes() if n != src])
            
            # Use find_path to get both path and quality
            path, quality = router.find_path(src, dst)
            
            # Only add successful paths to the collection
            if quality > 0:
                successful_paths.append((path, quality))
        
        # Update pheromones based on the successful paths
        pheromone_manager.update_pheromones(successful_paths)
        
        # Update router's iteration counter for adaptive routing
        router.update_iteration()
    
    # Convert metrics to DataFrame for reporting
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, f"pamr_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")
    
    # Create comparison visualizations
    create_metrics_visualizations(metrics_df, output_dir, iteration_paths)
    
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

def create_metrics_visualizations(metrics_df, output_dir, path_history=None):
    """
    Create static visualizations of the metrics for the report.
    
    Args:
        metrics_df: DataFrame containing metrics for each iteration
        output_dir: Directory to save the visualizations
        path_history: Optional history of paths taken during simulation
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
    
    # 5. Create a table summary of key metrics
    plt.figure(figsize=(14, min(len(metrics_df), 20) * 0.5 + 2))
    plt.axis('off')
    
    # Key metrics to include in the table - limit to fewer rows for readability with 100 iterations
    key_metrics = ['iteration', 'avg_congestion', 'max_congestion', 'avg_pheromone']
    path_metrics = [col for col in metrics_df.columns if col.startswith('path_quality_')]
    table_cols = key_metrics + path_metrics
    
    # Select a subset of iterations for the table if more than 20
    if len(metrics_df) > 20:
        # Choose iterations at regular intervals
        indices = np.linspace(0, len(metrics_df)-1, 20, dtype=int)
        table_data = metrics_df.iloc[indices][table_cols].values
    else:
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
    
    plt.title(f'Metrics Variation Over {len(metrics_df)} Iterations (Sample)', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=300)
    plt.close()
    
    # 6. NEW: Create visualization focusing on Path 0->6 changes over time
    if 'path_hops_0_to_6' in metrics_df.columns:
        plt.figure(figsize=(14, 8))
        
        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Plot 1: Path Quality
        ax1.plot(metrics_df['iteration'], metrics_df['path_quality_0_to_6'], 'b-', marker='o', markersize=4)
        ax1.set_ylabel('Path Quality')
        ax1.set_title('Path Quality from Node 0 to Node 6 Over Time')
        ax1.grid(True)
        
        # Plot 2: Path Hops (length)
        ax2.plot(metrics_df['iteration'], metrics_df['path_hops_0_to_6'], 'r-', marker='s', markersize=4)
        ax2.set_ylabel('Path Hops')
        ax2.set_title('Path Length from Node 0 to Node 6 Over Time')
        ax2.grid(True)
        
        # Plot 3: Max Congestion & Avg Pheromone
        ax3.plot(metrics_df['iteration'], metrics_df['max_congestion_0_to_6'], 'g-', marker='^', markersize=4, label='Max Congestion')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(metrics_df['iteration'], metrics_df['avg_pheromone_0_to_6'], 'm-', marker='*', markersize=4, label='Avg Pheromone')
        ax3.set_ylabel('Max Congestion', color='g')
        ax3_twin.set_ylabel('Avg Pheromone', color='m')
        ax3.set_title('Congestion and Pheromone Levels for Path 0 to 6')
        ax3.grid(True)
        ax3.set_xlabel('Iteration')
        
        # Add legend for the third subplot
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'path_0_to_6_metrics.png'), dpi=300)
        plt.close()
        
        # If we have path history, create path changes visualization
        if path_history is not None:
            # Extract paths from node 0 to node 6
            path_changes = []
            for i, paths in enumerate(path_history):
                if paths and len(paths) > 0:
                    # Find the path to destination 6 (should be first in the list if destinations=[6,9,11])
                    path_to_6 = paths[0]
                    path_str = ' → '.join(map(str, path_to_6))
                    path_changes.append({'iteration': i+1, 'path': path_str})
            
            if path_changes:
                # Convert to DataFrame
                path_df = pd.DataFrame(path_changes)
                
                # Count occurrences of each path
                path_counts = path_df['path'].value_counts()
                
                # Create bar chart of path frequencies
                plt.figure(figsize=(12, 6))
                path_counts.plot(kind='bar', color='skyblue')
                plt.title('Frequency of Different Paths from Node 0 to Node 6')
                plt.xlabel('Path')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'path_0_to_6_frequencies.png'), dpi=300)
                plt.close()
                
                # Create a timeline showing when path changes occurred
                plt.figure(figsize=(14, 6))
                
                # Get unique paths
                unique_paths = path_df['path'].unique()
                path_to_index = {path: i for i, path in enumerate(unique_paths)}
                
                # Convert paths to numeric indices for plotting
                path_indices = [path_to_index[p] for p in path_df['path']]
                
                # Create scatter plot
                plt.scatter(path_df['iteration'], path_indices, c='blue', s=50)
                
                # Add lines connecting points
                plt.plot(path_df['iteration'], path_indices, 'b-', alpha=0.5)
                
                # Set y-ticks to show actual paths
                plt.yticks(range(len(unique_paths)), unique_paths)
                
                plt.title('Path Changes from Node 0 to Node 6 Over Time')
                plt.xlabel('Iteration')
                plt.ylabel('Path Taken')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'path_0_to_6_changes.png'), dpi=300)
                plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    report_path = run_simulation_with_metrics_tracking(iterations=100)
    
    # Open the report in a browser using absolute path
    import webbrowser
    import time
    import os
    time.sleep(1)  # Small delay to ensure file is written
    
    # Convert to absolute path
    abs_report_path = os.path.abspath(report_path)
    print(f"Opening report at: {abs_report_path}")
    
    # Use file:// protocol with absolute path
    webbrowser.open(f'file://{abs_report_path}', new=2)
    print(f"Report opened in browser.")