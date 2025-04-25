import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from tabulate import tabulate
import json

class PathAnalyzer:
    """Analyze path selection decisions in PAMR routing."""
    
    def __init__(self, network, router):
        """Initialize with network and router objects."""
        self.network = network
        self.router = router
        self.graph = network.graph
        self.alpha = router.alpha
        self.beta = router.beta
        self.gamma = router.gamma
        
    def analyze_path(self, source, destination, output_dir='./analysis'):
        """Generate comprehensive analysis of path selection from source to destination."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Find the actual path using PAMR's own algorithm
        pamr_path, path_quality = self.router.find_path(source, destination)
        
        # Find all possible paths up to reasonable length
        all_simple_paths = list(nx.all_simple_paths(self.graph, source, destination, cutoff=8))[:10]
        
        # Calculate weights for Dijkstra's algorithm (same as in basic_simulation.py)
        def edge_weight(u, v, edge_data):
            pheromone = edge_data['pheromone']
            distance = edge_data['distance']
            congestion = edge_data['congestion']
            weight = (1 / pheromone) * distance * (1 + congestion)
            return weight
        
        # Find path using Dijkstra's algorithm
        dijkstra_path = nx.shortest_path(self.graph, source, destination, weight=edge_weight)
        
        # Create main HTML report
        html_report_path = os.path.join(output_dir, f'path_analysis_{source}_to_{destination}.html')
        
        # Create detailed CSV for each step
        csv_path = os.path.join(output_dir, f'path_metrics_{source}_to_{destination}.csv')
        
        # Analyze each node's decision in the PAMR path
        step_by_step_analysis = []
        
        for i in range(len(pamr_path) - 1):
            current = pamr_path[i]
            next_node = pamr_path[i + 1]
            
            # Analyze options at this step
            neighbors = list(self.graph.successors(current))
            valid_neighbors = [n for n in neighbors if n not in pamr_path[:i+1]]  # Exclude already visited
            
            options_data = []
            probabilities = []
            
            for neighbor in valid_neighbors:
                # Extract edge attributes
                edge_data = self.graph[current][neighbor]
                pheromone = edge_data['pheromone']
                distance = edge_data['distance']
                congestion = edge_data['congestion']
                traffic = edge_data.get('traffic', 0)
                
                # Calculate PAMR factors
                pheromone_factor = pheromone ** self.alpha
                distance_factor = (1.0 / distance) ** self.beta
                congestion_factor = (1.0 - min(congestion, 0.99)) ** self.gamma
                
                # Calculate desirability and edge weight
                desirability = pheromone_factor * distance_factor * congestion_factor
                weight = (1 / pheromone) * distance * (1 + congestion)
                
                # Store probability components
                probabilities.append(desirability)
                
                # Determine if this was the chosen option
                is_chosen = (neighbor == next_node)
                
                # Append data
                options_data.append({
                    'step': i,
                    'current_node': current,
                    'candidate_node': neighbor,
                    'pheromone': pheromone,
                    'distance': distance,
                    'congestion': congestion,
                    'traffic': traffic,
                    'pheromone_factor': pheromone_factor,
                    'distance_factor': distance_factor,
                    'congestion_factor': congestion_factor,
                    'desirability': desirability,
                    'edge_weight': weight,
                    'is_chosen': is_chosen
                })
            
            # Normalize probabilities
            total = sum(probabilities)
            normalized_probs = [p / total for p in probabilities] if total > 0 else [1.0/len(probabilities) for _ in probabilities]
            
            # Add probability to options data
            for j, prob in enumerate(normalized_probs):
                options_data[j]['probability'] = prob
            
            step_by_step_analysis.append(options_data)
        
        # Flatten list for CSV export
        flat_data = [item for sublist in step_by_step_analysis for item in sublist]
        df = pd.DataFrame(flat_data)
        df.to_csv(csv_path, index=False)
        
        # Generate HTML report
        self._generate_html_report(
            html_report_path,
            source,
            destination,
            pamr_path,
            dijkstra_path,
            all_simple_paths,
            step_by_step_analysis,
            path_quality
        )
        
        # Generate decision tree visualization
        tree_path = os.path.join(output_dir, f'decision_tree_{source}_to_{destination}.png')
        self._visualize_decision_tree(pamr_path, step_by_step_analysis, tree_path)
        
        # Generate path comparison visualization
        comparison_path = os.path.join(output_dir, f'path_comparison_{source}_to_{destination}.png')
        self._visualize_path_comparison(source, destination, pamr_path, dijkstra_path, comparison_path)
        
        return {
            'html_report': html_report_path,
            'csv_data': csv_path,
            'decision_tree': tree_path,
            'path_comparison': comparison_path
        }
    
    def _generate_html_report(self, output_path, source, destination, pamr_path, dijkstra_path, 
                             all_paths, step_analysis, path_quality):
        """Generate detailed HTML report of the path selection process."""
        
        # Compare different path options
        path_comparison = []
        for path in all_paths:
            metrics = self._calculate_path_metrics(path)
            metrics['path'] = '→'.join(map(str, path))
            metrics['is_pamr'] = (path == pamr_path)
            metrics['is_dijkstra'] = (path == dijkstra_path)
            path_comparison.append(metrics)
        
        # Sort paths by their selection likelihood
        path_comparison.sort(key=lambda x: x['quality'], reverse=True)
        
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Path Analysis from {source} to {destination}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr.selected {{ background-color: #d4edda; }}
                tr.alternative {{ background-color: #f8d7da; }}
                .decision-step {{ margin-bottom: 30px; border: 1px solid #ddd; border-radius: 5px; padding: 15px; }}
                .metric-item {{ display: inline-block; margin-right: 15px; padding: 5px 10px; background-color: #f8f9fa; border-radius: 3px; }}
                .analysis-container {{ display: flex; margin-top: 20px; }}
                .analysis-text {{ flex: 1; padding-right: 20px; }}
                .analysis-chart {{ flex: 1; }}
                .highlight {{ background-color: #fff3cd; padding: 2px 5px; border-radius: 3px; }}
                .factor-bar {{ height: 20px; background-color: #007bff; margin-bottom: 5px; }}
                .chosen {{ font-weight: bold; color: #28a745; }}
                .factor-high {{ color: #28a745; }}
                .factor-medium {{ color: #fd7e14; }}
                .factor-low {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <h1>Path Selection Analysis: Node {source} to Node {destination}</h1>
            
            <h2>Summary</h2>
            <div>
                <p>
                    <strong>PAMR Selected Path:</strong> {' → '.join(map(str, pamr_path))} 
                    <span class="metric-item">Quality: {path_quality:.4f}</span>
                    <span class="metric-item">Hops: {len(pamr_path)}</span>
                </p>
                <p>
                    <strong>Dijkstra Selected Path:</strong> {' → '.join(map(str, dijkstra_path))}
                    <span class="metric-item">Quality: {self._calculate_path_metrics(dijkstra_path)['quality']:.4f}</span>
                    <span class="metric-item">Hops: {len(dijkstra_path)}</span>
                </p>
            </div>
            
            <h2>Path Comparison</h2>
            <table>
                <tr>
                    <th>Path</th>
                    <th>Path Quality</th>
                    <th>Total Distance</th>
                    <th>Max Congestion</th>
                    <th>Avg Pheromone</th>
                    <th>Hops</th>
                    <th>Algorithm</th>
                </tr>
        """
        
        # Add path comparison rows
        for path_data in path_comparison:
            is_pamr = path_data['is_pamr']
            is_dijkstra = path_data['is_dijkstra']
            row_class = "selected" if is_pamr or is_dijkstra else ""
            algorithm = []
            if is_pamr:
                algorithm.append("PAMR")
            if is_dijkstra:
                algorithm.append("Dijkstra")
            
            html += f"""
                <tr class="{row_class}">
                    <td>{path_data['path']}</td>
                    <td>{path_data['quality']:.4f}</td>
                    <td>{path_data['total_distance']:.2f}</td>
                    <td>{path_data['max_congestion']:.2f}</td>
                    <td>{path_data['avg_pheromone']:.2f}</td>
                    <td>{path_data['hops']}</td>
                    <td>{', '.join(algorithm) if algorithm else 'Alternative'}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Step-by-Step Decision Process</h2>
        """
        
        # Add step-by-step analysis
        for i, step_data in enumerate(step_analysis):
            current_node = pamr_path[i]
            next_node = pamr_path[i+1]
            
            html += f"""
            <div class="decision-step">
                <h3>Step {i+1}: Node {current_node}</h3>
                <p>Current position: Node {current_node}</p>
                <p>Destination: Node {destination}</p>
                <table>
                    <tr>
                        <th>Candidate Node</th>
                        <th>Pheromone</th>
                        <th>Distance</th>
                        <th>Congestion</th>
                        <th>Pheromone Factor<br><small>(P^{self.alpha:.1f})</small></th>
                        <th>Distance Factor<br><small>(1/D)^{self.beta:.1f}</small></th>
                        <th>Congestion Factor<br><small>(1-C)^{self.gamma:.1f}</small></th>
                        <th>Desirability</th>
                        <th>Probability</th>
                    </tr>
            """
            
            # Sort candidates by probability
            step_data.sort(key=lambda x: x['probability'], reverse=True)
            
            for option in step_data:
                node = option['candidate_node']
                is_chosen = option['is_chosen']
                row_class = "selected" if is_chosen else ""
                
                # Determine factor magnitude classes
                ph_class = self._get_magnitude_class(option['pheromone_factor'])
                dist_class = self._get_magnitude_class(option['distance_factor'])
                cong_class = self._get_magnitude_class(option['congestion_factor'])
                
                html += f"""
                    <tr class="{row_class}">
                        <td>{'→ ' if is_chosen else ''}{node}</td>
                        <td>{option['pheromone']:.3f}</td>
                        <td>{option['distance']:.2f}</td>
                        <td>{option['congestion']:.2f}</td>
                        <td class="{ph_class}">{option['pheromone_factor']:.3f}</td>
                        <td class="{dist_class}">{option['distance_factor']:.3f}</td>
                        <td class="{cong_class}">{option['congestion_factor']:.3f}</td>
                        <td>{option['desirability']:.4f}</td>
                        <td>{option['probability']*100:.1f}%</td>
                    </tr>
                """
            
            html += """
                </table>
            """
            
            # Add explanation for this step
            chosen_option = next(option for option in step_data if option['is_chosen'])
            major_factor = self._determine_major_factor(chosen_option)
            
            html += f"""
                <div class="analysis-container">
                    <div class="analysis-text">
                        <h4>Decision Analysis</h4>
                        <p>
                            Node {current_node} selected Node {next_node} as the next hop with a 
                            {chosen_option['probability']*100:.1f}% probability. This decision was 
                            primarily influenced by <span class="highlight">{major_factor}</span>.
                        </p>
                        <p>
                            <strong>Decision Factors:</strong>
                            <ul>
                                <li><strong>Pheromone ({chosen_option['pheromone']:.3f}):</strong> 
                                   Indicates previous successful routing through this path. 
                                   {"High pheromone levels strongly favor this path." if chosen_option['pheromone'] > 0.8 else
                                    "Moderate pheromone levels suggest this is a viable path." if chosen_option['pheromone'] > 0.4 else
                                    "Low pheromone levels indicate this path is less traveled."}
                                </li>
                                <li><strong>Distance ({chosen_option['distance']:.2f}):</strong> 
                                   {"This link offers a shorter distance compared to alternatives." if chosen_option['distance_factor'] > 0.6 else
                                    "This link has moderate distance compared to alternatives." if chosen_option['distance_factor'] > 0.3 else
                                    "This link has a longer distance but other factors compensate."}
                                </li>
                                <li><strong>Congestion ({chosen_option['congestion']:.2f}):</strong> 
                                   {"This link has very low congestion, making it highly desirable." if chosen_option['congestion'] < 0.2 else
                                    "This link has moderate congestion, acceptable for routing." if chosen_option['congestion'] < 0.6 else
                                    "This link has high congestion but other factors outweigh this disadvantage."}
                                </li>
                            </ul>
                        </p>
                    </div>
                </div>
            </div>
            """
        
        # Finish HTML
        html += """
            <h2>Routing Formula Explanation</h2>
            <div>
                <p>
                    <strong>PAMR Algorithm Weights:</strong><br>
                    α (pheromone importance) = {self.alpha}<br>
                    β (distance importance) = {self.beta}<br>
                    γ (congestion importance) = {self.gamma}
                </p>
                <p>
                    <strong>Desirability Formula:</strong><br>
                    Desirability = (Pheromone^α) × (1/Distance)^β × (1-Congestion)^γ
                </p>
                <p>
                    <strong>Path Selection:</strong><br>
                    At each step, the next node is selected with probability proportional to its desirability compared to other options.
                </p>
                <p>
                    <strong>Path Quality:</strong><br>
                    Path Quality = 1 / (Total Distance × (1 + Max Congestion))
                </p>
            </div>
        </body>
        </html>
        """.format(self=self)
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _calculate_path_metrics(self, path):
        """Calculate comprehensive metrics for a given path."""
        if len(path) < 2:
            return {
                'quality': 0,
                'total_distance': 0,
                'max_congestion': 0,
                'avg_pheromone': 0,
                'hops': len(path)
            }
        
        total_distance = 0
        total_pheromone = 0
        max_congestion = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_distance += self.graph[u][v]['distance']
            total_pheromone += self.graph[u][v]['pheromone']
            max_congestion = max(max_congestion, self.graph[u][v]['congestion'])
        
        # Calculate path quality (same as in PAMRRouter)
        path_quality = 1.0 / (total_distance * (1 + max_congestion))
        
        return {
            'quality': path_quality,
            'total_distance': total_distance,
            'max_congestion': max_congestion,
            'avg_pheromone': total_pheromone / (len(path) - 1),
            'hops': len(path) - 1
        }
    
    def _determine_major_factor(self, option):
        """Determine which factor had the greatest influence on the decision."""
        factors = {
            'pheromone': option['pheromone_factor'],
            'distance': option['distance_factor'],
            'congestion': option['congestion_factor']
        }
        
        # Normalize the factors to compare their relative influence
        max_val = max(factors.values())
        if max_val == 0:
            return "random selection (all factors are zero)"
        
        normalized = {k: v/max_val for k, v in factors.items()}
        
        # Find the dominant factor
        major_factor = max(normalized.items(), key=lambda x: x[1])
        
        if major_factor[0] == 'pheromone':
            return f"high pheromone level ({option['pheromone']:.2f})"
        elif major_factor[0] == 'distance':
            return f"short distance ({option['distance']:.2f})"
        else:
            return f"low congestion ({option['congestion']:.2f})"
    
    def _get_magnitude_class(self, value):
        """Get CSS class based on the magnitude of a factor."""
        if value > 0.7:
            return "factor-high"
        elif value > 0.3:
            return "factor-medium"
        else:
            return "factor-low"
    
    def _visualize_decision_tree(self, path, step_analysis, output_path):
        """Create a decision tree visualization showing the path selection process."""
        plt.figure(figsize=(12, len(path) * 2))
        
        # Create a directional graph for the decisions
        G = nx.DiGraph()
        
        # Add nodes and edges based on the analysis
        for i, step_data in enumerate(step_analysis):
            current_node = path[i]
            next_node = path[i+1]
            
            # Add the current node to the graph
            G.add_node(f"{current_node}", level=i)
            
            # Add candidate nodes
            for option in step_data:
                candidate = option['candidate_node']
                probability = option['probability']
                is_chosen = option['is_chosen']
                
                # Only add high probability or chosen nodes to avoid clutter
                if probability > 0.1 or is_chosen:
                    # Create a unique node ID for this candidate at this step
                    candidate_id = f"{candidate}_{i+1}"
                    G.add_node(candidate_id, level=i+1, 
                              is_chosen=is_chosen, 
                              actual_node=candidate)
                    
                    # Add edge with probability as weight
                    G.add_edge(f"{current_node}", candidate_id, 
                              probability=probability,
                              pheromone=option['pheromone'],
                              distance=option['distance'], 
                              congestion=option['congestion'],
                              is_chosen=is_chosen)
        
        # Create positions for nodes
        pos = {}
        for node in G.nodes():
            level = G.nodes[node]['level']
            # Distribute nodes horizontally at each level
            same_level_nodes = [n for n in G.nodes() if G.nodes[n]['level'] == level]
            idx = same_level_nodes.index(node)
            pos[node] = (idx - len(same_level_nodes)/2, -level)
        
        # Draw the graph
        plt.figure(figsize=(10, len(path) * 1.5))
        
        # Draw edges with varying thickness based on probability
        for u, v, data in G.edges(data=True):
            prob = data['probability']
            is_chosen = data['is_chosen']
            
            # Scale line width based on probability
            width = 1 + 5 * prob
            
            # Color based on whether this was the chosen path
            color = 'red' if is_chosen else 'gray'
            alpha = 0.9 if is_chosen else 0.5
            
            # Draw the edge
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, 
                                  alpha=alpha, edge_color=color, arrows=True,
                                  arrowstyle='->', arrowsize=15)
            
            # Add edge label with probability percentage
            label = f"{prob*100:.0f}%"
            edge_labels = {(u, v): label}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                        font_size=8, label_pos=0.3)
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            if 'is_chosen' in G.nodes[node] and G.nodes[node]['is_chosen']:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color=node_colors, alpha=0.8)
        
        # Draw node labels
        labels = {}
        for node in G.nodes():
            if '_' in node:  # This is a candidate node
                actual_node = G.nodes[node]['actual_node']
                labels[node] = f"{actual_node}"
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
        
        # Add title and adjust layout
        plt.title(f"Path Decision Tree: Node {path[0]} to Node {path[-1]}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_path_comparison(self, source, destination, pamr_path, dijkstra_path, output_path):
        """Create a network visualization comparing different path options."""
        plt.figure(figsize=(12, 10))
        
        # Get positions for all nodes
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw the entire network in light gray
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, edge_color='gray')
        
        # Convert node lists to strings for better labels
        node_labels = {node: str(node) for node in self.graph.nodes()}
        
        # Draw regular nodes as light blue
        nx.draw_networkx_nodes(self.graph, pos, node_size=300, alpha=0.6, 
                              node_color='lightblue')
        
        # Draw source and destination nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[source, destination], 
                              node_size=500, node_color=['green', 'red'])
        
        # Prepare edge lists for PAMR and Dijkstra paths
        pamr_edges = [(pamr_path[i], pamr_path[i+1]) for i in range(len(pamr_path)-1)]
        dijkstra_edges = [(dijkstra_path[i], dijkstra_path[i+1]) for i in range(len(dijkstra_path)-1)]
        
        # Draw PAMR path in bold red
        nx.draw_networkx_edges(self.graph, pos, edgelist=pamr_edges, width=2.5, 
                              alpha=1.0, edge_color='red', arrows=True)
        
        # Draw Dijkstra path in bold blue (if different from PAMR)
        if pamr_path != dijkstra_path:
            nx.draw_networkx_edges(self.graph, pos, edgelist=dijkstra_edges, width=2.5, 
                                  alpha=0.7, edge_color='blue', arrows=True, style='dashed')
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)
        
        # Add edge labels for selected paths
        edge_labels = {}
        
        for i in range(len(pamr_path) - 1):
            u, v = pamr_path[i], pamr_path[i+1]
            pheromone = self.graph[u][v]['pheromone']
            distance = self.graph[u][v]['distance']
            congestion = self.graph[u][v]['congestion']
            edge_labels[(u, v)] = f"P:{pheromone:.1f}\nD:{distance:.1f}\nC:{congestion:.1f}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, 
                                    font_size=8)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2.5, label='PAMR Path'),
            Line2D([0], [0], color='blue', lw=2.5, linestyle='--', label='Dijkstra Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='Source Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Destination Node'),
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        # Add title
        plt.title(f"Path Comparison from Node {source} to Node {destination}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()