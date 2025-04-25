import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from plotly.subplots import make_subplots
from scipy import stats  # Used in statistical calculations throughout the class
from typing import List, Tuple
import warnings
import os
import logging
from sklearn.ensemble import IsolationForest
from datetime import datetime
import threading
import time
import webbrowser

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkMetricsVisualizer:
    """Advanced visualization and analysis of network metrics."""
    
    def __init__(self, data_dir: str = None):
        """Initialize the visualizer with optional data directory."""
        self.data_dir = data_dir
        self.node_metrics = None
        self.path_metrics = None
        self.time_series_data = None
        self.network_graph = None
        self.color_palette = sns.color_palette("viridis", 10)
        plt.style.use('ggplot')
        warnings.filterwarnings('ignore')
        
    def load_data(self, node_metrics_file: str = None, path_metrics_file: str = None,
                 time_series_file: str = None, graph_file: str = None) -> None:
        """Load metrics data from files."""
        try:
            if node_metrics_file and self.data_dir:
                self.node_metrics = pd.read_csv(os.path.join(self.data_dir, node_metrics_file))
                logger.info(f"Loaded node metrics: {self.node_metrics.shape}")
            
            if path_metrics_file and self.data_dir:
                self.path_metrics = pd.read_csv(os.path.join(self.data_dir, path_metrics_file))
                logger.info(f"Loaded path metrics: {self.path_metrics.shape}")
                
            if time_series_file and self.data_dir:
                self.time_series_data = pd.read_csv(os.path.join(self.data_dir, time_series_file))
                self.time_series_data['timestamp'] = pd.to_datetime(self.time_series_data['timestamp'])
                logger.info(f"Loaded time series data: {self.time_series_data.shape}")
                
            if graph_file and self.data_dir:
                self.network_graph = nx.read_graphml(os.path.join(self.data_dir, graph_file))
                logger.info(f"Loaded network graph: {len(self.network_graph.nodes)} nodes, {len(self.network_graph.edges)} edges")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def set_data(self, node_metrics: pd.DataFrame = None, path_metrics: pd.DataFrame = None,
                time_series_data: pd.DataFrame = None, network_graph: nx.Graph = None) -> None:
        """Set data directly from DataFrames and NetworkX graph."""
        if node_metrics is not None:
            self.node_metrics = node_metrics
        if path_metrics is not None:
            self.path_metrics = path_metrics
        if time_series_data is not None:
            self.time_series_data = time_series_data
        if network_graph is not None:
            self.network_graph = network_graph
            
    def generate_summary_statistics(self, data: pd.DataFrame = None, 
                                    group_by: str = None) -> pd.DataFrame:
        """Generate comprehensive summary statistics for metrics."""
        if data is None:
            if self.node_metrics is not None:
                data = self.node_metrics
            elif self.path_metrics is not None:
                data = self.path_metrics
            else:
                raise ValueError("No data available for analysis")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if group_by:
            grouped = data.groupby(group_by)
            stats_df = pd.DataFrame()
            
            for name, group in grouped:
                group_stats = group[numeric_cols].describe().T
                group_stats['group'] = name
                group_stats['skewness'] = group[numeric_cols].skew()
                group_stats['kurtosis'] = group[numeric_cols].kurtosis()
                group_stats['iqr'] = group_stats['75%'] - group_stats['25%']
                group_stats['cv'] = group_stats['std'] / group_stats['mean'].replace(0, np.nan)
                stats_df = pd.concat([stats_df, group_stats])
                
            return stats_df
        else:
            stats_df = data[numeric_cols].describe().T
            stats_df['skewness'] = data[numeric_cols].skew()
            stats_df['kurtosis'] = data[numeric_cols].kurtosis()
            stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
            stats_df['cv'] = stats_df['std'] / stats_df['mean'].replace(0, np.nan)
            return stats_df
            
    def visualize_node_metrics(self, metrics: List[str] = None, 
                              top_n: int = 10, 
                              plot_type: str = 'bar') -> go.Figure:
        """Visualize node metrics with advanced analytics."""
        if self.node_metrics is None:
            raise ValueError("Node metrics data not loaded")
            
        if metrics is None:
            numeric_cols = self.node_metrics.select_dtypes(include=[np.number]).columns.tolist()
            metrics = numeric_cols[:min(5, len(numeric_cols))]
            
        # Identify node identifier column
        id_col = next((col for col in self.node_metrics.columns if 'id' in col.lower() 
                      or 'node' in col.lower()), self.node_metrics.columns[0])
            
        # Select top performing nodes
        top_nodes = {}
        for metric in metrics:
            sorted_df = self.node_metrics.sort_values(by=metric, ascending=False)
            top_nodes[metric] = sorted_df.head(top_n)[id_col].tolist()
            
        # Create visualization
        if plot_type == 'bar':
            fig = make_subplots(rows=len(metrics), cols=1, 
                               subplot_titles=[f"Top {top_n} Nodes by {m}" for m in metrics],
                               vertical_spacing=0.1)
            
            for i, metric in enumerate(metrics, 1):
                sorted_df = self.node_metrics.sort_values(by=metric, ascending=False).head(top_n)
                fig.add_trace(
                    go.Bar(
                        x=sorted_df[id_col],
                        y=sorted_df[metric],
                        text=sorted_df[metric].round(2),
                        textposition='auto',
                        marker_color=px.colors.qualitative.Plotly,
                        name=metric
                    ),
                    row=i, col=1
                )
                
                # Add average line
                avg = self.node_metrics[metric].mean()
                fig.add_shape(type="line", line=dict(dash='dash', width=2, color="red"),
                             x0=0, y0=avg, x1=top_n-1, y1=avg,
                             row=i, col=1)
                
            fig.update_layout(height=300*len(metrics), width=900, 
                             title_text="Node Performance Analysis", showlegend=False)
            
        elif plot_type == 'radar':
            # For radar chart, select nodes that appear most frequently in top performers
            all_top_nodes = [node for nodes in top_nodes.values() for node in nodes]
            node_counts = pd.Series(all_top_nodes).value_counts()
            key_nodes = node_counts.head(min(5, len(node_counts))).index.tolist()
            
            # Create radar chart for these key nodes
            fig = go.Figure()
            
            for node in key_nodes:
                node_data = self.node_metrics[self.node_metrics[id_col] == node]
                
                if len(node_data) == 0:
                    continue
                    
                values = []
                for metric in metrics:
                    if metric in node_data.columns and pd.api.types.is_numeric_dtype(node_data[metric]):
                        values.append(node_data[metric].values[0])
                    else:
                        values.append(0)
                
                # Normalize values for better radar visualization
                max_vals = self.node_metrics[metrics].max()
                values_norm = [val/max_val if max_val != 0 else 0 
                              for val, max_val in zip(values, max_vals)]
                
                fig.add_trace(go.Scatterpolar(
                    r=values_norm,
                    theta=metrics,
                    fill='toself',
                    name=str(node)
                ))
                
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Multi-metric Node Performance Comparison"
            )
            
        return fig
    
    def analyze_path_performance(self, source_dest_pairs: List[Tuple] = None, 
                                metrics: List[str] = None,
                                time_window: Tuple[str, str] = None) -> go.Figure:
        """Advanced analysis of path performance between sources and destinations."""
        if self.path_metrics is None:
            raise ValueError("Path metrics data not loaded")
            
        # Identify source, destination columns
        src_col = next((col for col in self.path_metrics.columns 
                        if 'source' in col.lower() or 'src' in col.lower()), None)
        dst_col = next((col for col in self.path_metrics.columns 
                        if 'dest' in col.lower() or 'dst' in col.lower()), None)
        
        if src_col is None or dst_col is None:
            raise ValueError("Could not identify source and destination columns")
            
        if metrics is None:
            numeric_cols = self.path_metrics.select_dtypes(include=[np.number]).columns.tolist()
            metrics = numeric_cols[:min(3, len(numeric_cols))]
            
        # Filter by time window if provided and possible
        filtered_data = self.path_metrics
        if time_window and 'timestamp' in self.path_metrics.columns:
            start, end = time_window
            filtered_data = filtered_data[(filtered_data['timestamp'] >= start) & 
                                         (filtered_data['timestamp'] <= end)]
            
        # Get unique source-destination pairs if not provided
        if source_dest_pairs is None:
            all_pairs = filtered_data[[src_col, dst_col]].drop_duplicates().values
            pair_counts = filtered_data.groupby([src_col, dst_col]).size().sort_values(ascending=False)
            if len(pair_counts) > 0:
                source_dest_pairs = [tuple(x) for x in pair_counts.head(5).index.tolist()]
            else:
                logger.warning("No source-destination pairs found in data")
                source_dest_pairs = []
        
        # Create multi-metric, multi-path comparison
        fig = make_subplots(rows=len(metrics), cols=1, 
                           subplot_titles=[f"Path Performance: {m}" for m in metrics],
                           shared_xaxes=True, vertical_spacing=0.1)
        
        for i, metric in enumerate(metrics, 1):
            for src, dst in source_dest_pairs:
                path_data = filtered_data[(filtered_data[src_col] == src) & 
                                         (filtered_data[dst_col] == dst)]
                
                if len(path_data) == 0:
                    continue
                    
                # For time series data
                if 'timestamp' in path_data.columns:
                    path_data = path_data.sort_values('timestamp')
                    fig.add_trace(
                        go.Scatter(
                            x=path_data['timestamp'],
                            y=path_data[metric],
                            mode='lines+markers',
                            name=f"{src}->{dst}",
                            line=dict(width=2),
                            marker=dict(size=6)
                        ),
                        row=i, col=1
                    )
                # For non-time series, use path_id or index
                else:
                    path_id_col = next((col for col in path_data.columns 
                                       if 'path' in col.lower() and 'id' in col.lower()), None)
                    
                    x_vals = path_data[path_id_col] if path_id_col else path_data.index
                    fig.add_trace(
                        go.Bar(
                            x=x_vals,
                            y=path_data[metric],
                            name=f"{src}->{dst}",
                            text=path_data[metric].round(2),
                        ),
                        row=i, col=1
                    )
        
        # Add statistical annotations
        for i, metric in enumerate(metrics, 1):
            metric_stats = filtered_data.groupby([src_col, dst_col])[metric].agg(['mean', 'std', 'min', 'max'])
            if not metric_stats.empty:
                stats_text = (f"Overall stats - Mean: {metric_stats['mean'].mean():.2f}, "
                             f"StdDev: {metric_stats['std'].mean():.2f}")
                
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.5, y=1.0 - (i-1)/len(metrics),
                    text=stats_text,
                    showarrow=False,
                    font=dict(size=10)
                )
            
        fig.update_layout(height=300*len(metrics), title_text="Path Performance Analysis",
                         legend_title="Source-Destination Pairs")
        
        return fig
    
    def visualize_network_topology(self, color_by: str = None, 
                                  size_by: str = None,
                                  highlight_paths: List[List[str]] = None) -> go.Figure:
        """Visualize network topology with metrics-based styling."""
        if self.network_graph is None:
            raise ValueError("Network graph not loaded")
            
        # Position nodes using force-directed layout
        pos = nx.spring_layout(self.network_graph)
        
        # Create node trace
        node_x = []
        node_y = []
        for node in self.network_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=15,
                colorbar=dict(
                    thickness=15,
                    title=dict(
                        text=color_by if color_by else 'Node Importance',
                        side='right'
                    ),
                    xanchor='left',
                ),
                line=dict(width=2)
            )
        )
        
        # Add node attributes
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in self.network_graph.nodes():
            node_info = [f"ID: {node}"]
            
            # Add node attributes
            for key, value in self.network_graph.nodes[node].items():
                if isinstance(value, (int, float)):
                    node_info.append(f"{key}: {value:.2f}")
                else:
                    node_info.append(f"{key}: {value}")
                    
            node_text.append('<br>'.join(node_info))
            
            # Set node color based on attribute
            if color_by and color_by in self.network_graph.nodes[node]:
                try:
                    node_colors.append(float(self.network_graph.nodes[node][color_by]))
                except (ValueError, TypeError):
                    node_colors.append(0)
            else:
                # Default to degree centrality
                node_colors.append(self.network_graph.degree(node))
                
            # Set node size based on attribute
            if size_by and size_by in self.network_graph.nodes[node]:
                try:
                    attr_value = float(self.network_graph.nodes[node][size_by])
                    node_sizes.append(max(15, 15 + 10 * attr_value))
                except (ValueError, TypeError):
                    node_sizes.append(15)
            else:
                node_sizes.append(15)
                
        node_trace.marker.color = node_colors
        node_trace.marker.size = node_sizes
        node_trace.text = node_text
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in self.network_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create highlighted path traces if requested
        path_traces = []
        if highlight_paths:
            colors = px.colors.qualitative.Plotly
            for i, path in enumerate(highlight_paths):
                if len(path) < 2:
                    continue
                    
                path_x = []
                path_y = []
                
                try:
                    for j in range(len(path)-1):
                        if path[j] in pos and path[j+1] in pos:  # Check if nodes exist
                            x0, y0 = pos[path[j]]
                            x1, y1 = pos[path[j+1]]
                            path_x.extend([x0, x1, None])
                            path_y.extend([y0, y1, None])
                    
                    if path_x:
                        path_trace = go.Scatter(
                            x=path_x, y=path_y,
                            line=dict(width=3, color=colors[i % len(colors)]),
                            hoverinfo='text',
                            text=f"Path {i+1}",
                            mode='lines',
                            name=f"Path {i+1}"
                        )
                        path_traces.append(path_trace)
                    else:
                        logger.warning(f"Path {i+1} has no valid node positions and won't be displayed")
                except Exception as e:
                    logger.error(f"Error processing path {i+1}: {str(e)}")
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace, *path_traces],
                      layout=go.Layout(
                          title=dict(
                              text='Network Topology with Metrics',
                              font=dict(size=16)
                          ),
                          showlegend=True,
                          hovermode='closest',
                          margin=dict(b=20,l=5,r=5,t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                      ))
        
        return fig
    
    def detect_anomalies(self, data: pd.DataFrame = None, 
                        metrics: List[str] = None,
                        method: str = 'zscore',
                        threshold: float = 3.0) -> pd.DataFrame:
        """Detect anomalies in network metrics using various methods."""
        if data is None:
            if self.node_metrics is not None:
                data = self.node_metrics
            elif self.path_metrics is not None:
                data = self.path_metrics
            else:
                raise ValueError("No data available for analysis")
                
        if metrics is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            metrics = numeric_cols
            
        # Create a DataFrame to store anomaly flags
        anomalies = pd.DataFrame(index=data.index)
        
        # Detect anomalies based on chosen method
        if method == 'zscore':
            for col in metrics:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:  # Avoid division by zero
                    z_scores = (data[col] - mean) / std
                    anomalies[f'{col}_anomaly'] = (abs(z_scores) > threshold).astype(int)
                    anomalies[f'{col}_zscore'] = z_scores
                else:
                    logger.warning(f"Column {col} has zero standard deviation, skipping z-score calculation")
                    anomalies[f'{col}_anomaly'] = 0
                    anomalies[f'{col}_zscore'] = 0
                    
        elif method == 'iqr':
            for col in metrics:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    anomalies[f'{col}_anomaly'] = ((data[col] < lower_bound) | 
                                                (data[col] > upper_bound)).astype(int)
                else:
                    logger.warning(f"Column {col} has zero IQR, skipping anomaly detection")
                    anomalies[f'{col}_anomaly'] = 0
                
        elif method == 'isolation_forest':
            try:
                subset = data[metrics].copy()
                subset = subset.fillna(subset.mean())
                
                if len(subset) >= 10 and subset.shape[1] > 0:
                    contamination = min(0.05, 1/len(subset))
                    model = IsolationForest(contamination=contamination, random_state=42)
                    predictions = model.fit_predict(subset)
                    anomalies['anomaly_score'] = predictions
                    anomalies['is_anomaly'] = (predictions == -1).astype(int)
                else:
                    logger.warning("Insufficient data for Isolation Forest algorithm")
                    anomalies['is_anomaly'] = 0
            except Exception as e:
                logger.error(f"Error in Isolation Forest: {str(e)}")
                anomalies['is_anomaly'] = 0
        
        # Count total anomalies per row
        anomaly_cols = [col for col in anomalies.columns if 'anomaly' in col]
        if anomaly_cols:
            anomalies['total_anomalies'] = anomalies[anomaly_cols].sum(axis=1)
            
        return anomalies
        
    def create_correlation_heatmap(self, data: pd.DataFrame = None, 
                                 metrics: List[str] = None) -> go.Figure:
        """Create an interactive correlation heatmap for network metrics."""
        if data is None:
            if self.node_metrics is not None:
                data = self.node_metrics
            elif self.path_metrics is not None:
                data = self.path_metrics
            else:
                raise ValueError("No data available for analysis")
                
        if metrics is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            metrics = numeric_cols
            
        if not metrics:
            raise ValueError("No numeric metrics available for correlation analysis")
            
        try:
            clean_data = data[metrics].dropna()
            if len(clean_data) < 2:
                raise ValueError("Insufficient data points for correlation analysis after removing NaNs")
                
            corr_matrix = clean_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size":10},
                hoverinfo='text',
                hovertext=[[f'{col1} vs {col2}: {val:.4f}' 
                           for col2, val in zip(corr_matrix.columns, row)]
                          for col1, row in zip(corr_matrix.index, corr_matrix.values)]
            ))
            
            fig.update_layout(
                title='Correlation Heatmap of Network Metrics',
                xaxis_title='Metrics',
                yaxis_title='Metrics',
                height=700,
                width=700
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                text=f"Error creating correlation heatmap: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(
                title='Correlation Analysis Error',
                height=400,
                width=700
            )
            return fig
    
    def generate_advanced_report(self, output_dir: str = "./reports", 
                                filename: str = "network_metrics_report.html"):
        """Generate a comprehensive HTML report with all analyses."""
        import plotly.io as pio
        
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, filename)
        
        # Start building HTML report
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Advanced Network Metrics Analysis</title>",
            "    <style>",
            "        body {font-family: Arial, sans-serif; margin: 40px;}",
            "        .section {margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px;}",
            "        h1 {color: #2c3e50;}",
            "        h2 {color: #3498db;}",
            "        table {border-collapse: collapse; width: 100%;}",
            "        th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}",
            "        th {background-color: #f2f2f2;}",
            "        tr:nth-child(even) {background-color: #f9f9f9;}",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>Advanced Network Metrics Analysis Report</h1>",
            f"    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]
        
        # Add Node Metrics section if available
        if self.node_metrics is not None:
            html_content.extend([
                "    <div class='section'>",
                "        <h2>Node Metrics Summary</h2>"
            ])
            
            try:
                stats_df = self.generate_summary_statistics(self.node_metrics)
                html_content.append(stats_df.to_html(classes='dataframe', float_format='%.3f'))
            except Exception as e:
                html_content.append(f"<p>Error generating statistics: {str(e)}</p>")
            
            try:
                fig = self.visualize_node_metrics()
                html_content.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            except Exception as e:
                html_content.append(f"<p>Error generating node metrics visualization: {str(e)}</p>")
                
            html_content.append("    </div>")
        
        if self.path_metrics is not None:
            html_content.extend([
                "    <div class='section'>",
                "        <h2>Path Performance Analysis</h2>"
            ])
            
            try:
                fig = self.analyze_path_performance()
                html_content.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            except Exception as e:
                html_content.append(f"<p>Error generating path performance visualization: {str(e)}</p>")
                
            html_content.append("    </div>")
        
        if self.network_graph is not None:
            html_content.extend([
                "    <div class='section'>",
                "        <h2>Network Topology Analysis</h2>"
            ])
            
            try:
                fig = self.visualize_network_topology()
                html_content.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            except Exception as e:
                html_content.append(f"<p>Error generating network topology visualization: {str(e)}</p>")
                
            html_content.append("    </div>")
            
        html_content.extend([
            "    <div class='section'>",
            "        <h2>Metric Correlation Analysis</h2>"
        ])
        
        try:
            data = self.node_metrics if self.node_metrics is not None else self.path_metrics
            if data is not None:
                fig = self.create_correlation_heatmap(data)
                html_content.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            else:
                html_content.append("<p>No data available for correlation analysis</p>")
        except Exception as e:
            html_content.append(f"<p>Error generating correlation heatmap: {str(e)}</p>")
            
        html_content.append("    </div>")
        
        html_content.extend([
            "    <div class='section'>",
            "        <h2>Anomaly Detection Results</h2>"
        ])
        
        try:
            data = self.node_metrics if self.node_metrics is not None else self.path_metrics
            if data is not None:
                anomalies = self.detect_anomalies(data)
                if 'total_anomalies' in anomalies.columns:
                    anomaly_summary = pd.DataFrame({
                        'Total Anomalies': [anomalies['total_anomalies'].sum()],
                        'Rows with Anomalies': [(anomalies['total_anomalies'] > 0).sum()],
                        'Percentage Affected': [f"{((anomalies['total_anomalies'] > 0).mean() * 100):.2f}%"]
                    })
                    html_content.append(anomaly_summary.to_html(classes='dataframe', index=False))
                    
                    top_anomalies = anomalies.sort_values('total_anomalies', ascending=False).head(10)
                    if not top_anomalies.empty:
                        html_content.append("<h3>Top 10 Anomalies</h3>")
                        html_content.append(top_anomalies.to_html(classes='dataframe', float_format='%.3f'))
                else:
                    html_content.append("<p>No anomalies detected in the data</p>")
            else:
                html_content.append("<p>No data available for anomaly detection</p>")
        except Exception as e:
            html_content.append(f"<p>Error performing anomaly detection: {str(e)}</p>")
            
        html_content.append("    </div>")
        
        html_content.extend([
            "</body>",
            "</html>"
        ])
        
        try:
            with open(report_path, 'w') as f:
                f.write('\n'.join(html_content))
                
            logger.info(f"Report generated: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error writing report to file: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Create a visualizer instance
    metrics_viz = NetworkMetricsVisualizer()
    
    # Create sample data for demonstration
    import pandas as pd
    import numpy as np
    import networkx as nx
    
    # Sample node metrics data
    node_df = pd.DataFrame({
        'node_id': [f'node_{i}' for i in range(10)],
        'centrality': np.random.uniform(0, 1, 10),
        'avg_pheromone': np.random.uniform(0, 5, 10),
        'traffic_handled': np.random.randint(10, 100, 10)
    })
    
    # Sample path metrics data
    path_df = pd.DataFrame({
        'path_id': [f'path_{i}' for i in range(8)],
        'source': [f'node_{i}' for i in range(8)],
        'destination': [f'node_{i+1}' for i in range(8)],
        'latency': np.random.uniform(1, 10, 8),
        'bandwidth': np.random.uniform(10, 100, 8)
    })
    
    # Sample time series data
    time_series = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=24, freq='H'),
        'metric': np.random.uniform(0, 100, 24)
    })
    
    # Sample network graph
    network = nx.Graph()
    for i in range(10):
        network.add_node(f'node_{i}', centrality=np.random.random())
    for i in range(9):
        network.add_edge(f'node_{i}', f'node_{i+1}')
    
    # Set data for metrics visualizer
    metrics_viz.set_data(
        node_metrics=node_df,
        path_metrics=path_df,
        time_series_data=time_series,
        network_graph=network
    )
    try:
        # Generate node metric plots (simpler, less likely to fail)
        node_viz = metrics_viz.visualize_node_metrics(
            metrics=['centrality', 'avg_pheromone', 'traffic_handled'],
            plot_type='bar'
        )
        
        # Generate path performance analysis (simpler version)
        path_viz = None
        if not path_df.empty:
            path_viz = metrics_viz.analyze_path_performance()
        
        # Generate correlation heatmap
        corr_viz = metrics_viz.create_correlation_heatmap()
        
        # Save these visualizations separately as fallbacks
        from plotly.offline import plot
        plot(node_viz, filename='./reports/node_metrics.html', auto_open=False)
        if path_viz is not None:
            plot(path_viz, filename='./reports/path_performance.html', auto_open=False)
        plot(corr_viz, filename='./reports/metrics_correlation.html', auto_open=False)
        
        print("Individual metrics visualizations generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate individual metrics: {str(e)}")
    
    # Generate advanced report
    os.makedirs('./reports', exist_ok=True)
    report_filename = f"pamr_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    try:
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
            
    except Exception as e:
        print(f"Error generating advanced report: {str(e)}")
        print("Opening individual metrics visualizations instead...")
        
        # Open individual visualization files as fallback
        for file in ['node_metrics.html', 'metrics_correlation.html']:
            file_path = os.path.abspath(f'./reports/{file}')
            if os.path.exists(file_path):
                webbrowser.open(f'file://{file_path}', new=2)
                time.sleep(1)  # Delay to avoid browser throttling