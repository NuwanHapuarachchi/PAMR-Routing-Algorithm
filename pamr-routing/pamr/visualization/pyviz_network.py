import holoviews as hv
import networkx as nx
import numpy as np
import panel as pn
import pandas as pd
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