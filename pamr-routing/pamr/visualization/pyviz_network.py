import holoviews as hv
import networkx as nx
import numpy as np
import panel as pn
import pandas as pd
import time  # Add time import for timestamp generation
from holoviews import opts, dim
from bokeh.models import HoverTool, ColumnDataSource, DataTable, TableColumn

class PyVizNetworkVisualizer:
    """Network visualization for PAMR protocol using PyViz (HoloViews/Bokeh)."""
    
    def __init__(self, network):
        self.network = network
        self.graph = network.graph
        self.pos = network.positions
        # Initialize HoloViews extension with Bokeh
        hv.extension('bokeh')
        # Store metrics over iterations
        self.iteration_metrics = []
    
    def calculate_path_metrics(self, paths, source, destination):
        """Calculate detailed metrics for paths and alternatives."""
        # Store path metrics for analysis
        path_metrics = []
        
        # Get the chosen path
        chosen_path = paths[0] if paths else None
        
        # Get all possible paths up to a reasonable length
        all_paths = list(nx.all_simple_paths(self.graph, source, destination, cutoff=8))[:10]  # Limit to 10 paths
        
        for path in all_paths:
            # Calculate path metrics
            total_distance = 0
            total_pheromone = 0
            max_congestion = 0
            avg_pheromone = 0
            avg_congestion = 0
            path_weight = 0  # Lower is better for routing decisions
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                pheromone = self.graph[u][v]['pheromone']
                distance = self.graph[u][v]['distance']
                congestion = self.graph[u][v]['congestion']
                
                # Accumulate metrics
                total_distance += distance
                total_pheromone += pheromone
                max_congestion = max(max_congestion, congestion)
                
                # Calculate weight (same as in path finding)
                edge_weight = (1 / pheromone) * distance * (1 + congestion)
                path_weight += edge_weight
            
            # Calculate averages
            path_length = len(path) - 1  # Number of edges
            avg_pheromone = total_pheromone / path_length if path_length > 0 else 0
            avg_congestion = max_congestion  # Using max as the representative value
            
            # Calculate path quality (same as in PAMRRouter._calculate_path_quality)
            path_quality = 1.0 / (total_distance * (1 + max_congestion))
            
            # Is this the chosen path?
            is_chosen = (path == chosen_path)
            
            # Add path data
            path_metrics.append({
                'Path': '→'.join(str(n) for n in path),
                'Length': len(path),
                'Total Distance': round(total_distance, 2),
                'Max Congestion': round(max_congestion, 2),
                'Avg Pheromone': round(avg_pheromone, 2),
                'Path Quality': round(path_quality, 4),
                'Weight': round(path_weight, 2),
                'Is Chosen': is_chosen
            })
        
        # Sort by path weight (lower is better)
        path_metrics.sort(key=lambda x: x['Weight'])
        
        return path_metrics
    
    def visualize_network(self, source=None, destinations=None, paths=None):
        """
        Visualize the network with path and pheromone levels using HoloViews.
        Returns an interactive HoloViews object.
        """
        # Create node data
        node_data = []
        for node in self.graph.nodes():
            node_type = 'regular'
            if node == source:
                node_type = 'source'
            elif destinations and node in destinations:
                node_type = 'destination'
            elif paths and any(node in path[1:-1] for path in paths):
                node_type = 'path'
                
            node_data.append({
                'index': node,
                'x': self.pos[node][0],
                'y': self.pos[node][1],
                'type': node_type
            })
        
        # Create node HoloMap
        nodes = hv.Dataset(node_data, kdims=['index', 'x', 'y', 'type'])
        
        # Create edge data
        edge_data = []
        for u, v in self.graph.edges():
            edge_data.append({
                'source': u,
                'target': v,
                'pheromone': self.graph[u][v]['pheromone'],
                'distance': self.graph[u][v]['distance'],
                'congestion': self.graph[u][v]['congestion'],
                'traffic': self.graph[u][v]['traffic'],
                'x0': self.pos[u][0],
                'y0': self.pos[u][1],
                'x1': self.pos[v][0],
                'y1': self.pos[v][1],
                'on_path': False
            })
        
        # Mark edges that are on paths
        if paths:
            for path in paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    for edge in edge_data:
                        if edge['source'] == u and edge['target'] == v:
                            edge['on_path'] = True
        
        # Create edge HoloMap
        edges = hv.Dataset(edge_data, kdims=['source', 'target', 'x0', 'y0', 'x1', 'y1', 
                                             'pheromone', 'distance', 'congestion', 'traffic', 'on_path'])
        
        # Create node visualization
        node_points = hv.Points(nodes, kdims=['x', 'y'], vdims=['index', 'type'])
        
        # Node styling based on type
        node_points = node_points.opts(
            opts.Points(
                color='type', 
                cmap={'regular': 'lightblue', 'source': 'green', 'destination': 'red', 'path': 'orange'},
                size=25,  # Change this value to make nodes larger
                tools=['hover'],
                hover_tooltips=[('Node ID', '@index'), ('Type', '@type')]
            )
        )
        
        # Create edge visualization
        edge_segments = hv.Segments(edges, kdims=['x0', 'y0', 'x1', 'y1'], 
                                   vdims=['source', 'target', 'pheromone', 'distance', 'congestion', 'traffic', 'on_path'])
        
        # Edge styling with proper path coloring
        edge_segments = edge_segments.opts(
            opts.Segments(
                # Color path edges red, other edges by pheromone level
                color=dim('on_path').categorize({True: 'red', False: 'blue'}),
                alpha=dim('pheromone').norm() * 0.8 + 0.2,  # Use pheromone for edge opacity
                line_width=dim('on_path').categorize({True: 4, False: 1}),
                tools=['hover'],
                hover_tooltips=[
                    ('Source', '@source'), 
                    ('Target', '@target'),
                    ('Pheromone', '@pheromone'),
                    ('Distance', '@distance'),
                    ('Congestion', '@congestion'),
                    ('Traffic', '@traffic'),
                    ('On Path', '@on_path')
                ]
            )
        )
        
        # Create node labels
        node_labels = hv.Labels(nodes, kdims=['x', 'y'], vdims=['index'])
        node_labels = node_labels.opts(
            opts.Labels(text_font_size='8pt', text_color='black')
        )
        
        # Combine visualizations
        network_viz = edge_segments * node_points * node_labels
        
        # Set plot options
        network_viz = network_viz.opts(
            opts.Segments(width=1200, height=800, title='PAMR Routing Simulation (Interactive)'),
            opts.Points(width=1200, height=800),
            opts.Labels(width=1200, height=800)
        )
        
        # Create path metrics tabs for each destination
        if paths and source is not None and destinations is not None:
            dest_tabs = []
            
            # For each destination, create a tab with path analysis
            for i, dest in enumerate(destinations):
                if i < len(paths):
                    dest_path = paths[i]
                    
                    # Calculate metrics for this specific path
                    path_metrics = self.calculate_path_metrics([dest_path], source, dest)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(path_metrics)
                    
                    # Create the destination tab with path details
                    metrics_panel = pn.Column(
                        pn.pane.Markdown(f"## Path Analysis: {source} → {dest}"),
                        pn.pane.Markdown(f"**Selected Path:** {' → '.join(str(n) for n in dest_path)}"),
                        pn.pane.Markdown("### Path Metrics:"),
                        pn.pane.DataFrame(df.drop(columns=['Is Chosen']) if 'Is Chosen' in df.columns else df, 
                                         width=800),
                        pn.layout.Divider(),
                        pn.pane.Markdown("### Edge-by-Edge Analysis:"),
                        self._create_edge_metrics_table(dest_path)
                    )
                    dest_tabs.append((f"Path to {dest}", metrics_panel))
                
            # Create tabbed interface for all destinations
            metrics_tabs = pn.Tabs(*dest_tabs)
            
            # Add explanation for path selection algorithm
            explanation = pn.pane.Markdown("""
            ## PAMR Path Selection Analysis
            
            The PAMR algorithm selects paths based on three main factors:
            
            1. **Pheromone levels** (α): Higher values indicate successful past routes
            2. **Distance** (β): Shorter physical distances are preferred
            3. **Congestion** (γ): Less congested links are preferred
            
            The combined weight formula is: `weight = (1/pheromone) * distance * (1 + congestion)`
            
            Lower weights result in preferred paths. Select different destination tabs to see detailed path analyses.
            """)
            
            # Return the visualization with metrics dashboard
            return pn.Column(
                pn.Row(network_viz, sizing_mode='stretch_width'),
                pn.layout.Divider(),
                explanation,
                metrics_tabs
            )
        
        # If no paths, return just the network visualization
        return network_viz

    def _create_edge_metrics_table(self, path):
        """Create a detailed edge-by-edge metrics table for a path."""
        edge_data = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Get edge properties
            pheromone = self.graph[u][v]['pheromone']
            distance = self.graph[u][v]['distance']
            congestion = self.graph[u][v]['congestion']
            
            # Calculate edge weight (same as in routing)
            edge_weight = (1 / pheromone) * distance * (1 + congestion)
            
            edge_data.append({
                'Edge': f"{u} → {v}",
                'Pheromone': round(pheromone, 4),
                'Distance': round(distance, 2),
                'Congestion': round(congestion, 2),
                'Edge Weight': round(edge_weight, 2)
            })
        
        # Create DataFrame
        df = pd.DataFrame(edge_data)
        
        # Add totals
        if len(edge_data) > 0:
            total_row = {
                'Edge': 'TOTAL',
                'Pheromone': sum(edge['Pheromone'] for edge in edge_data),
                'Distance': sum(edge['Distance'] for edge in edge_data),
                'Congestion': max(edge['Congestion'] for edge in edge_data),
                'Edge Weight': sum(edge['Edge Weight'] for edge in edge_data)
            }
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        return pn.pane.DataFrame(df, width=800)
    
    def track_iteration_metrics(self, iteration, paths=None, source=None, destinations=None):
        """Track network metrics for the current iteration."""
        # Get general network metrics
        metrics = {
            'iteration': iteration,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'avg_congestion': np.mean([self.graph[u][v]['congestion'] for u, v in self.graph.edges()]),
            'max_congestion': np.max([self.graph[u][v]['congestion'] for u, v in self.graph.edges()]),
            'avg_pheromone': np.mean([self.graph[u][v]['pheromone'] for u, v in self.graph.edges()]),
            'avg_traffic': np.mean([self.graph[u][v]['traffic'] for u, v in self.graph.edges()]),
        }
        
        # Add path metrics if paths are provided
        if paths and source is not None and destinations is not None:
            for i, dest in enumerate(destinations):
                if i < len(paths):
                    path = paths[i]
                    path_metrics = self._calculate_path_metrics_basic(path)
                    metrics[f'path_{source}_to_{dest}_quality'] = path_metrics['quality']
                    metrics[f'path_{source}_to_{dest}_length'] = path_metrics['length']
                    metrics[f'path_{source}_to_{dest}_congestion'] = path_metrics['max_congestion']
                    metrics[f'path_{source}_to_{dest}_hops'] = len(path) - 1
        
        # Add iteration metrics to history
        self.iteration_metrics.append(metrics)
        return metrics
    
    def _calculate_path_metrics_basic(self, path):
        """Calculate basic metrics for a path."""
        if len(path) < 2:
            return {'quality': 0, 'length': 0, 'max_congestion': 0}
        
        total_distance = 0
        max_congestion = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_distance += self.graph[u][v]['distance']
            max_congestion = max(max_congestion, self.graph[u][v]['congestion'])
        
        # Calculate path quality
        path_quality = 1.0 / (total_distance * (1 + max_congestion))
        
        return {
            'quality': path_quality,
            'length': total_distance,
            'max_congestion': max_congestion
        }
    
    def visualize_metrics_over_iterations(self, metrics_to_show=None, height=500, width=1000):
        """
        Create an interactive visualization of network metrics over iterations.
        
        Args:
            metrics_to_show: List of metrics to visualize. If None, shows a default set.
            height: Height of the plot in pixels
            width: Width of the plot in pixels
            
        Returns:
            A Panel dashboard with interactive metric visualizations
        """
        if not self.iteration_metrics:
            return pn.Column(pn.pane.Markdown("## No iteration metrics available yet"))
        
        # Convert metrics history to DataFrame
        metrics_df = pd.DataFrame(self.iteration_metrics)
        
        # Default metrics to show if none specified
        if metrics_to_show is None:
            general_metrics = ['avg_congestion', 'max_congestion', 'avg_pheromone', 'avg_traffic']
            path_metrics = [col for col in metrics_df.columns if col.startswith('path_') and col.endswith('_quality')]
            metrics_to_show = general_metrics + path_metrics[:3]  # Limit to 3 path metrics to avoid overcrowding
        
        # Create metrics visualization
        metrics_plot = self._create_metrics_line_plot(metrics_df, metrics_to_show, height, width)
        
        # Create convergence analysis
        convergence_analysis = self._analyze_convergence(metrics_df, width)
        
        # Create iteration comparison table showing first, middle and last iterations
        comparison_table = self._create_iteration_comparison_table(metrics_df)
        
        # Combine into a dashboard
        dashboard = pn.Column(
            pn.pane.Markdown("# Network Metrics Over Iterations"),
            pn.pane.Markdown("## Metrics Trends"),
            metrics_plot,
            pn.pane.Markdown("## Convergence Analysis"),
            convergence_analysis,
            pn.pane.Markdown("## Iteration Comparison"),
            comparison_table,
            pn.pane.Markdown("## Statistical Summary"),
            pn.pane.DataFrame(metrics_df.describe(), width=width)
        )
        
        return dashboard
    
    def _create_metrics_line_plot(self, metrics_df, metrics_to_show, height, width):
        """Create line plots for metrics over iterations."""
        # Create line plots for each metric
        plots = []
        for metric in metrics_to_show:
            if metric in metrics_df.columns:
                # Create holoviews line plot
                line = hv.Curve(
                    (metrics_df['iteration'], metrics_df[metric]), 
                    kdims=['Iteration'], 
                    vdims=[metric.replace('_', ' ').title()]
                ).opts(
                    width=width,
                    height=height//len(metrics_to_show),
                    tools=['hover'],
                    title=metric.replace('_', ' ').title()
                )
                plots.append(line)
        
        # Overlay all plots in a layout
        layout = hv.Layout(plots).cols(1)
        return layout
    
    def _analyze_convergence(self, metrics_df, width=1000):
        """Analyze convergence patterns in metrics."""
        # Identify metrics that show convergence
        converged_metrics = []
        convergence_data = []
        
        for col in metrics_df.columns:
            if col not in ['iteration', 'timestamp'] and metrics_df[col].dtype != 'object':
                # Calculate convergence using rolling window standard deviation
                if len(metrics_df) >= 5:  # Need enough data points
                    rolling_std = metrics_df[col].rolling(window=3).std()
                    # Check if the standard deviation decreases over time
                    first_std = rolling_std.dropna().head(3).mean()
                    last_std = rolling_std.dropna().tail(3).mean()
                    
                    if last_std < first_std * 0.5:  # Convergence detected
                        converged_metrics.append(col)
                        convergence_data.append({
                            'Metric': col.replace('_', ' ').title(),
                            'Starting Variability': first_std,
                            'Ending Variability': last_std,
                            'Convergence Rate': (1 - (last_std / first_std)) * 100
                        })
        
        # Create a DataFrame for the convergence data
        if convergence_data:
            convergence_df = pd.DataFrame(convergence_data)
            return pn.pane.DataFrame(convergence_df, width=width)
        else:
            return pn.pane.Markdown("No clear convergence patterns detected yet. Need more iterations.")
    
    def _create_iteration_comparison_table(self, metrics_df):
        """Create a table comparing metrics across iterations."""
        if len(metrics_df) < 2:
            return pn.pane.Markdown("Not enough iterations for comparison")
        
        # Select first, middle and last iterations for comparison
        first_iteration = metrics_df.iloc[0].copy()
        mid_idx = len(metrics_df) // 2
        middle_iteration = metrics_df.iloc[mid_idx].copy()
        last_iteration = metrics_df.iloc[-1].copy()
        
        # Prepare comparison data
        comparison_data = []
        
        for col in metrics_df.columns:
            if col not in ['iteration', 'timestamp'] and metrics_df[col].dtype != 'object':
                metric_name = col.replace('_', ' ').title()
                first_val = first_iteration[col]
                middle_val = middle_iteration[col]
                last_val = last_iteration[col]
                
                # Calculate change percentages
                mid_change = ((middle_val - first_val) / first_val * 100) if first_val != 0 else 0
                overall_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                
                comparison_data.append({
                    'Metric': metric_name,
                    'First Iteration': round(first_val, 4),
                    'Middle Iteration': round(middle_val, 4),
                    'Last Iteration': round(last_val, 4),
                    'Mid-point Change %': round(mid_change, 2),
                    'Overall Change %': round(overall_change, 2)
                })
        
        # Convert to DataFrame and create a panel table
        comparison_df = pd.DataFrame(comparison_data)
        return pn.pane.DataFrame(comparison_df)
    
    def create_comprehensive_report(self, source=None, destinations=None, paths=None, output_dir=None):
        """
        Create a comprehensive interactive HTML report with all metrics, visualizations and tables.
        
        Args:
            source: Source node ID
            destinations: List of destination node IDs
            paths: List of paths from source to each destination
            output_dir: Directory to save the report (if None, returns the Panel dashboard without saving)
            
        Returns:
            Panel dashboard object and path to saved HTML file (if output_dir provided)
        """
        import time
        
        # Network visualization
        network_viz = self.visualize_network(source, destinations, paths)
        
        # Metrics over iterations
        metrics_viz = self.visualize_metrics_over_iterations()
        
        # Path quality comparison if paths are available
        path_comparison = None
        if paths and source is not None and destinations is not None:
            path_comparison = self._create_path_quality_comparison(source, destinations, paths)
        
        # Create dashboard
        dashboard = pn.Column(
            pn.pane.Markdown(f"# PAMR Protocol Analysis Report"),
            pn.pane.Markdown(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"),
            
            pn.pane.Markdown("## Network Visualization"),
            network_viz,
            
            pn.pane.Markdown("## Metrics Analysis Over Iterations"),
            metrics_viz
        )
        
        # Add path comparison if available
        if path_comparison:
            dashboard.append(pn.pane.Markdown("## Path Quality Comparison"))
            dashboard.append(path_comparison)
        
        # Save to HTML if output directory is provided
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pamr_metrics_report_{timestamp}.html"
            output_path = os.path.join(output_dir, filename)
            
            dashboard.save(output_path)
            print(f"Comprehensive report saved to: {output_path}")
            return dashboard, output_path
        
        return dashboard
    
    def _create_path_quality_comparison(self, source, destinations, paths):
        """Create a visualization comparing path qualities."""
        # Prepare data for visualization
        data = []
        
        for i, dest in enumerate(destinations):
            if i < len(paths):
                path = paths[i]
                path_metrics = self._calculate_path_metrics_basic(path)
                
                # Get all possible paths for comparison
                all_paths = list(nx.all_simple_paths(self.graph, source, dest, cutoff=8))[:5]  # Limit to 5 alternatives
                
                # Calculate metrics for all alternative paths
                alternatives = []
                for alt_path in all_paths:
                    if alt_path != path:  # Skip the chosen path
                        alt_metrics = self._calculate_path_metrics_basic(alt_path)
                        alternatives.append({
                            'Path': ' → '.join(str(n) for n in alt_path),
                            'Quality': alt_metrics['quality'],
                            'Distance': alt_metrics['length'],
                            'Max Congestion': alt_metrics['max_congestion'],
                            'Hops': len(alt_path) - 1,
                            # Add a comparison indicator
                            'Better Than Chosen': alt_metrics['quality'] > path_metrics['quality']
                        })
                
                data.append({
                    'Source': source,
                    'Destination': dest,
                    'Chosen Path': ' → '.join(str(n) for n in path),
                    'Chosen Quality': path_metrics['quality'],
                    'Chosen Distance': path_metrics['length'],
                    'Chosen Congestion': path_metrics['max_congestion'],
                    'Alternatives': alternatives
                })
        
        # Create path comparison visualization
        if not data:
            return pn.pane.Markdown("No path data available for comparison")
        
        # Create tabs for each source-destination pair
        tabs = []
        for path_data in data:
            # Create a DataFrame for alternatives
            if path_data['Alternatives']:
                alt_df = pd.DataFrame(path_data['Alternatives'])
                # Calculate quality ratios
                quality_ratios = alt_df['Quality'] / path_data['Chosen Quality']
                # Add the quality ratio to the dataframe
                alt_df['Quality Ratio'] = quality_ratios
                
                # Create the bar plot of quality ratios
                # Fix: Instead of using a list of colors, use a color dimension based on a column
                quality_plot = hv.Bars(
                    alt_df, 
                    kdims=['Path'], 
                    vdims=['Quality Ratio', 'Better Than Chosen']
                ).opts(
                    color='Better Than Chosen',  # Fixed: Use a dimension instead of a list
                    cmap={True: 'green', False: 'red'},  # Mapping for the dimension values
                    width=800,
                    height=300,
                    title=f"Alternative Path Quality Comparison (Ratio to Chosen Path)",
                    tools=['hover'],
                    yformatter='%.2f',
                    ylim=(0, max(quality_ratios) * 1.1 if len(quality_ratios) > 0 else 2)
                )
                
                # Create a tabular comparison
                comparison = pn.Column(
                    pn.pane.Markdown(f"### Source {path_data['Source']} to Destination {path_data['Destination']}"),
                    pn.pane.Markdown(f"**Chosen Path:** {path_data['Chosen Path']}"),
                    pn.pane.Markdown(f"**Quality:** {path_data['Chosen Quality']:.4f} | **Distance:** {path_data['Chosen Distance']:.2f} | **Congestion:** {path_data['Chosen Congestion']:.2f}"),
                    pn.pane.Markdown("### Alternative Paths"),
                    pn.pane.DataFrame(alt_df),
                    pn.pane.Markdown("### Quality Comparison (Higher is Better)"),
                    quality_plot
                )
                
                tabs.append((f"Path to {path_data['Destination']}", comparison))
            else:
                # No alternatives available
                tabs.append((
                    f"Path to {path_data['Destination']}", 
                    pn.pane.Markdown(f"No alternative paths available for comparison")
                ))
        
        return pn.Tabs(*tabs)