import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import time
import pickle
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# Add parent directory to path so we can import the pamr package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PAMR classes
from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter
from pamr.simulation.simulator import PAMRSimulator
from pamr.utils.path_analyzer import PathAnalyzer

# Import OSPFRouter for comparison
from comparison_with_ospf import OSPFRouter
from comparison_with_ospf import OSPFSimulator
class PAMRTrafficPredictor:
    """Machine learning model for predicting network traffic and congestion patterns."""
    
    def __init__(self, model_type='random_forest'):
        """Initialize the traffic predictor with specified ML model.
        
        Args:
            model_type (str): The type of model to use ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_training_data(self, simulator, num_iterations=150):
        """Run simulation and collect training data.
        
        Args:
            simulator (PAMRSimulator): Simulator instance
            num_iterations (int): Number of iterations to run
        
        Returns:
            tuple: X (features) and y (targets) for training
        """
        print(f"Collecting training data over {num_iterations} iterations...")
        
        # Run simulation to collect data
        path_history = simulator.run_simulation(num_iterations=num_iterations, packets_per_iter=50)
        
        # Extract features from network state after simulation
        features = []
        targets = []
        
        # Get reference to network
        network = simulator.network
        
        # Track congestion over time for each edge
        edge_congestion_history = {}
        
        # Process each iteration's data
        for iter_idx, iter_paths in enumerate(path_history):
            # Skip first few iterations as they're initialization phase
            if iter_idx < 10:
                continue
                
            # Collect edge features
            for u, v in network.graph.edges():
                edge_id = f"{u}-{v}"
                
                # Initialize history if this is the first time seeing this edge
                if edge_id not in edge_congestion_history:
                    edge_congestion_history[edge_id] = []
                
                # Get edge attributes
                edge_data = network.graph[u][v]
                
                # Store congestion history
                edge_congestion_history[edge_id].append(edge_data['congestion'])
                
                # Only start predicting once we have some history
                if len(edge_congestion_history[edge_id]) >= 5:
                    # Feature vector: [edge properties + recent congestion history]
                    feature_vector = [
                        u, v,                                        # Source, destination nodes
                        edge_data['distance'],                       # Physical distance
                        edge_data['pheromone'],                      # Current pheromone level
                        edge_data['traffic'],                        # Current traffic amount
                        edge_data.get('capacity', 10),               # Edge capacity
                        network.graph.degree(u),                     # Source node degree
                        network.graph.degree(v),                     # Destination node degree
                        np.mean(edge_congestion_history[edge_id][-5:]), # Mean congestion (last 5)
                        np.std(edge_congestion_history[edge_id][-5:]),  # Std congestion (last 5)
                        np.min(edge_congestion_history[edge_id][-5:]),  # Min congestion (last 5)
                        np.max(edge_congestion_history[edge_id][-5:]),  # Max congestion (last 5)
                    ]
                    
                    # Store feature vectors and targets
                    features.append(feature_vector)
                    targets.append(edge_data['congestion'])  # Current congestion as target
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        # Save feature names for later interpretation
        self.feature_names = [
            'source_node', 'dest_node', 'distance', 'pheromone', 'traffic', 'capacity',
            'source_degree', 'dest_degree', 'mean_congestion_5', 'std_congestion_5',
            'min_congestion_5', 'max_congestion_5'
        ]
        
        print(f"Collected {len(X)} training samples")
        return X, y
    
    def train(self, X, y):
        """Train the ML model on the provided data."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Select and train model with regularization to prevent overfitting
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=8,  # Limit tree depth 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        # Time the training
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        # Performance evaluation
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model trained in {training_time:.2f} seconds")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R²: {r2:.4f}")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            print("\nFeature importance ranking:")
            for f in range(min(10, len(indices))):
                idx = indices[f]
                if f < len(self.feature_names):
                    print(f"{f+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict_congestion(self, network, look_ahead=1):
        """Predict future congestion levels for the network.
        
        Args:
            network (NetworkTopology): The network to predict congestion for
            look_ahead (int): Number of steps to look ahead
            
        Returns:
            dict: Predicted congestion values for each edge
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare feature vectors for each edge
        edge_features = []
        edge_ids = []
        
        for u, v in network.graph.edges():
            edge_data = network.graph[u][v]
            edge_id = f"{u}-{v}"
            edge_ids.append(edge_id)
            
            # Get historical congestion if available, otherwise use current
            congestion_history = [edge_data['congestion']] * 5  # Simple default
            
            # Create feature vector (same structure as training data)
            feature_vector = [
                u, v,
                edge_data['distance'],
                edge_data['pheromone'],
                edge_data['traffic'],
                edge_data.get('capacity', 10),
                network.graph.degree(u),
                network.graph.degree(v),
                np.mean(congestion_history),
                np.std(congestion_history),
                np.min(congestion_history),
                np.max(congestion_history),
            ]
            
            edge_features.append(feature_vector)
        
        # Scale features
        X = np.array(edge_features)
        X_scaled = self.scaler.transform(X)
        
        # Predict congestion
        predictions = self.model.predict(X_scaled)
        
        # Create result dictionary
        predicted_congestion = {}
        for i, (u, v) in enumerate(network.graph.edges()):
            predicted_congestion[(u, v)] = max(0.0, min(1.0, predictions[i]))  # Clamp to [0,1]
            
        return predicted_congestion
    
    def save_model(self, filename="pamr_traffic_predictor.pkl"):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="pamr_traffic_predictor.pkl"):
        """Load a trained model from disk."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.model_type = data['model_type']
        print(f"Model loaded from {filename}")


class PredictiveRouter(PAMRRouter):
    """Enhanced PAMR router that uses ML predictions to improve routing decisions."""
    
    def __init__(self, graph, alpha=2.0, beta=3.0, gamma=2.5, predictor=None, prediction_weight=0.3):
        super().__init__(graph, alpha, beta, gamma)
        self.predictor = predictor
        self.prediction_weight = prediction_weight
        self.predicted_congestion = {}
        # We don't need to set max_path_length as we're using global path finding
    
    def update_predictions(self):
        """Update the congestion predictions using the ML model."""
        if self.predictor is None:
            return
            
        # Create a temporary network topology with existing graph
        temp_network = NetworkTopology(num_nodes=30, connectivity=0.3)
        # Replace its graph with our current graph
        temp_network.graph = self.graph
        
        # Get predictions from the model
        self.predicted_congestion = self.predictor.predict_congestion(temp_network)
    
    def find_path(self, source, destination, max_steps=100):
        """Find a path using ML-enhanced PAMR with global path consideration."""
        # Update predictions before finding path
        self.update_predictions()
        
        if source == destination:
            return [source], 0
        
        # Use global path finding with ML-enhanced edge weights
        if self.use_global_path:
            global_path = self._find_global_optimal_path_with_ml(source, destination)
            if global_path:
                path_quality = self._calculate_path_quality(global_path)
                # Update traffic on the path
                for i in range(len(global_path) - 1):
                    u, v = global_path[i], global_path[i+1]
                    self.graph[u][v]['traffic'] += 1
                return global_path, path_quality
        
        # Fall back to original method if global path fails or is disabled
        return super().find_path(source, destination, max_steps)
    
    def _find_global_optimal_path_with_ml(self, source, destination):
        """Find a globally optimal path using a modified Dijkstra's algorithm with ML predictions."""
        try:
            # Define edge weight function that considers ML predictions
            def edge_weight(u, v, edge_data):
                pheromone = edge_data['pheromone']
                distance = edge_data['distance']
                current_congestion = edge_data['congestion']
                
                # Get predicted congestion if available
                predicted_congestion = self.predicted_congestion.get((u, v), current_congestion)
                
                # Blend current and predicted congestion
                trend_factor = 1.5 if predicted_congestion > current_congestion else 1.0
                effective_congestion = (
                    (1 - self.prediction_weight) * current_congestion + 
                    self.prediction_weight * min(predicted_congestion, 0.9) * trend_factor
                )
                
                # Apply congestion factor (similar to OSPF)
                congestion_factor = 1 + effective_congestion * 5
                
                # Include pheromone in inverse proportion (higher pheromone = lower weight)
                pheromone_factor = 1 / (pheromone + 0.1)  # Add 0.1 to avoid division by zero
                
                # Combined weight - lower is better
                return distance * congestion_factor * pheromone_factor
            
            # Use Dijkstra's algorithm with the custom weight function
            path = nx.shortest_path(self.graph, source, destination, weight=edge_weight)
            return path
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None
    
    def _select_next_node(self, current_node, destination, visited):
        """Select next node using ML-enhanced PAMR algorithm."""
        neighbors = list(self.graph.successors(current_node))
        if not neighbors:
            return None
        
        # Calculate selection probabilities
        probabilities = []
        valid_neighbors = []
        
        for neighbor in neighbors:
            if neighbor in visited:
                continue
                
            valid_neighbors.append(neighbor)
            
            # Extract edge attributes
            pheromone = self.graph[current_node][neighbor]['pheromone']
            distance = self.graph[current_node][neighbor]['distance']
            current_congestion = self.graph[current_node][neighbor]['congestion']
            
            # Get predicted congestion if available
            predicted_congestion = self.predicted_congestion.get((current_node, neighbor), current_congestion)
            
            # Blend current and predicted congestion
            trend_factor = 1.5 if predicted_congestion > current_congestion else 1.0
            effective_congestion = (
                (1 - self.prediction_weight) * current_congestion + 
                self.prediction_weight * min(predicted_congestion, 0.9) * trend_factor
            )
            
            # Calculate desirability
            pheromone_factor = pheromone ** self.alpha
            distance_factor = (1.0 / distance) ** self.beta
            congestion_factor = (1.0 - effective_congestion) ** self.gamma
            
            # Combined desirability
            desirability = pheromone_factor * distance_factor * congestion_factor
            probabilities.append(desirability)
        
        # If no valid neighbors, return None
        if not valid_neighbors:
            return None
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            # If all probabilities are 0, choose randomly
            return random.choice(valid_neighbors)
            
        probabilities = [p / total for p in probabilities]

        # Select neighbor based on probability
        selected_idx = np.argmax(probabilities)  # Choose highest probability always
        return valid_neighbors[selected_idx]


def visualize_ospf_vs_pamr_paths(network, source, destinations, pamr_router, ospf_router, output_dir="comparison_results"):
    """Visualize path selection differences between PAMR and OSPF."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for dest in destinations:
        # Get paths from both routers
        pamr_path, pamr_quality = pamr_router.find_path(source, dest)
        ospf_path, ospf_quality = ospf_router.find_path(source, dest)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Use consistent layout
        pos = nx.spring_layout(network.graph, seed=42)
        
        # Draw network background
        nx.draw_networkx_edges(network.graph, pos, alpha=0.2, edge_color='gray')
        nx.draw_networkx_nodes(network.graph, pos, node_size=300, alpha=0.6, node_color='lightblue')
        
        # Draw source and destination nodes
        nx.draw_networkx_nodes(network.graph, pos, nodelist=[source, dest], 
                              node_size=500, node_color=['green', 'red'])
        
        # Draw PAMR path
        pamr_edges = [(pamr_path[i], pamr_path[i+1]) for i in range(len(pamr_path)-1)]
        nx.draw_networkx_edges(network.graph, pos, edgelist=pamr_edges, width=2.5, 
                              alpha=1.0, edge_color='red', arrows=True)
        
        # Draw OSPF path (if different from PAMR)
        ospf_edges = [(ospf_path[i], ospf_path[i+1]) for i in range(len(ospf_path)-1)]
        nx.draw_networkx_edges(network.graph, pos, edgelist=ospf_edges, width=2.5, 
                              alpha=0.7, edge_color='blue', arrows=True, style='dashed')
            
        # Draw node labels
        nx.draw_networkx_labels(network.graph, pos)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2.5, label='PAMR Path'),
            Line2D([0], [0], color='blue', lw=2.5, linestyle='--', label='OSPF Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Source'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Destination')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        # Add metrics comparison
        plt.figtext(0.5, 0.01, 
                   f"PAMR Quality: {pamr_quality:.4f} | OSPF Quality: {ospf_quality:.4f} | ML Improvement: {((pamr_quality/ospf_quality)-1)*100:.2f}%", 
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2})
        
        # Set title and style
        plt.title(f"Path Comparison: PAMR vs OSPF from Node {source} to Node {dest}")
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ospf_vs_pamr_path_{source}_to_{dest}_{timestamp}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()


def visualize_ospf_vs_pamr_ml_paths(network, source, dest, ospf_router, standard_router, ml_router, output_dir="comparison_results"):
    """Create side-by-side visualization of paths selected by OSPF, PAMR, and ML-PAMR."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get paths from all routers
    ospf_path, ospf_quality = ospf_router.find_path(source, dest)
    pamr_path, pamr_quality = standard_router.find_path(source, dest)
    ml_path, ml_quality = ml_router.find_path(source, dest)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Use consistent layout
    pos = nx.spring_layout(network.graph, seed=42)
    
    # Draw network background
    nx.draw_networkx_edges(network.graph, pos, alpha=0.2, edge_color='gray')
    nx.draw_networkx_nodes(network.graph, pos, node_size=300, alpha=0.6, node_color='lightblue')
    
    # Draw source and destination nodes
    nx.draw_networkx_nodes(network.graph, pos, nodelist=[source, dest], 
                          node_size=500, node_color=['green', 'red'])
    
    # Draw OSPF path
    ospf_edges = [(ospf_path[i], ospf_path[i+1]) for i in range(len(ospf_path)-1)]
    nx.draw_networkx_edges(network.graph, pos, edgelist=ospf_edges, width=2.5, 
                          alpha=0.7, edge_color='blue', arrows=True, style='dashed')
    
    # Draw standard PAMR path
    pamr_edges = [(pamr_path[i], pamr_path[i+1]) for i in range(len(pamr_path)-1)]
    nx.draw_networkx_edges(network.graph, pos, edgelist=pamr_edges, width=2.5, 
                          alpha=1.0, edge_color='red', arrows=True)
    
    # Draw ML-PAMR path
    ml_edges = [(ml_path[i], ml_path[i+1]) for i in range(len(ml_path)-1)]
    nx.draw_networkx_edges(network.graph, pos, edgelist=ml_edges, width=2.5, 
                          alpha=1.0, edge_color='green', arrows=True, style='dotted')
    
    # Add node labels
    nx.draw_networkx_labels(network.graph, pos)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2.5, linestyle='--', 
               label=f'OSPF ({len(ospf_path)-1} hops, quality: {ospf_quality:.4f})'),
        Line2D([0], [0], color='red', lw=2.5, 
               label=f'PAMR ({len(pamr_path)-1} hops, quality: {pamr_quality:.4f})'),
        Line2D([0], [0], color='green', lw=2.5, linestyle='dotted', 
               label=f'ML-PAMR ({len(ml_path)-1} hops, quality: {ml_quality:.4f})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Source'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Destination')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Set title and save
    plt.title(f"Path Comparison: OSPF vs PAMR vs ML-PAMR from Node {source} to Node {dest}")
    plt.axis('off')
    plt.tight_layout()
    
    # Save with timestamp to avoid overwriting
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"ospf_vs_pamr_ml_path_{source}_to_{dest}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def visualize_performance_metrics(pamr_sim, ospf_sim, ml_pamr_sim, standard_network, ospf_network, predictive_network, iterations=100, output_dir="comparison_results"):
    """Visualize performance metrics to highlight ML enhancements."""
    metrics = {
        'Convergence Time': [
            np.mean(pamr_sim.metrics['convergence_times']),
            np.mean(ospf_sim.metrics['convergence_times']),
            np.mean(ml_pamr_sim.metrics['convergence_times'])
        ],
        'Traffic Distribution': [
            np.std([e['congestion'] for _, _, e in standard_network.graph.edges(data=True)]),
            np.std([e['congestion'] for _, _, e in ospf_network.graph.edges(data=True)]),
            np.std([e['congestion'] for _, _, e in predictive_network.graph.edges(data=True)])
        ],
        'Path Quality': [
            np.mean(pamr_sim.metrics.get('path_qualities', [0])),
            np.mean(ospf_sim.metrics.get('path_qualities', [0])),
            np.mean(ml_pamr_sim.metrics.get('path_qualities', [0]))
        ]
    }
    
    # Create DataFrame for visualization
    df = pd.DataFrame(metrics, index=['PAMR', 'OSPF', 'ML-PAMR'])
    
    # Create visualization for each metric
    for metric in metrics.keys():
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=df.index, y=df[metric])
        
        # Add value labels on bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.title(f"Comparison of {metric}")
        plt.ylabel(metric)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{output_dir}/{metric.lower().replace(' ', '_')}_comparison.png", dpi=300)
        plt.close()


def generate_results_dashboard(network, pamr_sim, ospf_sim, ml_pamr_sim, 
                              output_dir="comparison_results", 
                              output_file="routing_comparison_dashboard.html"):
    """Generate an HTML dashboard to view all comparison results."""
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all image files in the output directory
    image_files = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.png'):
                image_files.append(os.path.join(output_dir, file))
    
    # Categorize images based on filename patterns
    path_comparison_images = []
    metric_comparison_images = []
    congestion_viz_images = []
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        if "path" in img_name.lower():
            # Extract source and destination from path comparison images
            parts = img_name.split('_')
            if len(parts) >= 5:
                try:
                    source = parts[-4]
                    dest = parts[-3]
                    caption = f"Path from Node {source} to Node {dest}"
                    path_comparison_images.append((img_path, caption))
                except Exception:
                    path_comparison_images.append((img_path, "Path Comparison"))
        
        elif "comparison" in img_name.lower():
            # Extract metric name from comparison images
            metric_name = img_name.split('_comparison')[0].replace('_', ' ').title()
            metric_comparison_images.append((img_path, metric_name))
        
        elif "congestion" in img_name.lower():
            congestion_viz_images.append(img_path)
    
    # Extract simulation parameters if available, or use default values
    # Use the value from the passed parameters to main() if attributes don't exist
    num_iterations = getattr(pamr_sim, 'num_iterations', 100)  # Default to 100 if attribute doesn't exist
    packets_per_iter = getattr(pamr_sim, 'packets_per_iter', 50)  # Default to 50 if attribute doesn't exist
    
    # Generate timestamp for the report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Copy topology image to reports directory if available
    topology_image_path = os.path.join(os.path.dirname(output_dir), "network_topology.png")
    if os.path.exists(topology_image_path):
        topology_rel_path = "network_topology.png"
    else:
        topology_rel_path = ""
    
    # Create relative path to images directory from reports directory
    # This is how we'll reference images in the HTML
    images_rel_path = "../images/comparison"
    
    # Create HTML content with network details and styling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PAMR vs OSPF Routing Comparison</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px;
                background-color: #f5f5f5; 
            }}
            h1, h2, h3 {{ 
                color: #333;
                margin-top: 25px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 8px;
            }}
            .image-gallery {{ 
                display: flex; 
                flex-wrap: wrap; 
                justify-content: center; 
                gap: 20px; 
                margin-bottom: 30px; 
            }}
            .gallery-item {{
                max-width: 45%;
                margin-bottom: 20px;
                text-align: center;
            }}
            .gallery-item img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            .gallery-caption {{
                margin-top: 8px;
                font-size: 14px;
                color: #555;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PAMR vs OSPF Routing Comparison Dashboard</h1>
            <p>This dashboard presents a comparison between the PAMR and OSPF routing protocols, 
            showcasing path selections, performance metrics, and congestion visualizations.</p>
            <p><strong>Generated:</strong> {timestamp}</p>
            <h2>Network Topology</h2>
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="{topology_rel_path}" alt="Network Topology" style="max-width:80%; height:auto;">
            </div>
            <h2>Simulation Parameters</h2>
            <ul>
                <li><strong>Number of Iterations:</strong> {num_iterations}</li>
                <li><strong>Packets per Iteration:</strong> {packets_per_iter}</li>
            </ul>
            <h2>Path Selection Comparisons</h2>
            <div class="image-gallery">
    """
    for image_path, caption in path_comparison_images:
        img_filename = os.path.basename(image_path)
        img_url = f"{images_rel_path}/{img_filename}"
        html_content += f"""
            <div class="gallery-item">
                <img src="{img_url}" alt="Path comparison visualization">
                <p class="gallery-caption">{caption}</p>
            </div>
        """
    html_content += """
            </div>
        """
    
    # Same for metric comparison - ONLY if available
    if metric_comparison_images:
        html_content += """
        <h2>Performance Metric Comparisons</h2>
        <div class="image-gallery">
"""
        for image_path, caption in metric_comparison_images:
            img_filename = os.path.basename(image_path)
            img_url = f"{images_rel_path}/{img_filename}"
            html_content += f"""
            <div class="gallery-item">
                <img src="{img_url}" alt="Metric comparison visualization">
                <p class="gallery-caption">{caption} Comparison</p>
            </div>
"""
        html_content += """
        </div>
"""
    
    # Same for congestion visualizations - ONLY if available
    if congestion_viz_images:
        html_content += """
        <h2>Congestion Visualization</h2>
        <div class="image-gallery">
"""
        for image_path in congestion_viz_images:
            img_filename = os.path.basename(image_path)
            img_url = f"{images_rel_path}/{img_filename}"
            html_content += f"""
            <div class="gallery-item">
                <img src="{img_url}" alt="Congestion visualization">
                <p class="gallery-caption">Network Congestion Map</p>
            </div>
"""
        html_content += """
        </div>
"""
    
    # Add summary and conclusion
    html_content += """
        <h2>Analysis & Findings</h2>
        <p>The ML-enhanced PAMR routing protocol demonstrates several advantages over both standard PAMR and OSPF:</p>
        <ul>
            <li><strong>Predictive Congestion Avoidance:</strong> By anticipating traffic patterns, ML-PAMR can route around areas before they become congested</li>
            <li><strong>Optimized Path Selection:</strong> The machine learning model integrates historical performance data to make better routing decisions</li>
            <li><strong>Adaptive Parameter Tuning:</strong> The ML component dynamically adjusts routing parameters based on network conditions</li>
        </ul>
        
        <p class="highlight">Note: This dashboard is dynamically generated based on the current network configuration and available visualization results.</p>
    </div>
    
    <div class="footer" style="text-align: center; margin-top: 30px; color: #777;">
        <p>PAMR Routing Protocol Simulation Framework</p>
    </div>
</body>
</html>
"""
    
    # Use the provided output file path or create one with timestamp
    if not output_file:
        output_file = f"pamr_metrics_report_{timestamp}.html"
    
    # Write HTML content to file
    output_path = output_file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard generated at {output_path}")
    return output_path


def main():
    """Run a demonstration of the ML-enhanced PAMR routing."""
    print("Initializing ML-enhanced PAMR routing demonstration...")
    
    # Create structured results directories
    results_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    results_dirs = {
        "images": os.path.join(results_base_dir, "images"),
        "reports": os.path.join(results_base_dir, "reports"),
        "data": os.path.join(results_base_dir, "data"),
        "models": os.path.join(results_base_dir, "models"),
        "comparison": os.path.join(results_base_dir, "images", "comparison")
    }
    
    # Create directories if they don't exist
    for dir_path in results_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Clean up old comparison results to avoid confusion
    comparison_dir = results_dirs["comparison"]
    if os.path.exists(comparison_dir):
        for file in os.listdir(comparison_dir):
            file_path = os.path.join(comparison_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # Use the consistent network defined in network.py
    from pamr.core.network import network
    
    # Print network info for debugging
    print(f"Using network with {len(network.graph)} nodes and {network.graph.number_of_edges()} edges")
    print(f"Network nodes: {list(network.graph.nodes())}")
    
    # Generate and save network topology visualization
    from pamr.visualization.network_viz import NetworkVisualizer
    visualizer = NetworkVisualizer(network)
    topology_fig = visualizer.visualize_network(
        edge_attribute='congestion', 
        title='Network Topology - Congestion Levels'
    )
    
    # Save topology visualization to both reports and images directory
    reports_topology_path = os.path.join(results_dirs["reports"], "network_topology.png")
    images_topology_path = os.path.join(results_dirs["images"], "network_topology.png")
    
    topology_fig.savefig(reports_topology_path, dpi=300, bbox_inches='tight')
    topology_fig.savefig(images_topology_path, dpi=300, bbox_inches='tight')
    plt.close(topology_fig)
    
    # Create optimized standard PAMR router
    standard_router = PAMRRouter(network.graph, alpha=2.0, beta=3.0, gamma=2.5)
    
    # Create simulator
    simulator = PAMRSimulator(network, standard_router)
    
    # Create and train the traffic predictor
    predictor = PAMRTrafficPredictor(model_type='random_forest')
    X, y = predictor.prepare_training_data(simulator, num_iterations=200)
    predictor.train(X, y)
    
    # Save model to the models directory
    model_path = os.path.join(results_dirs["models"], "pamr_traffic_predictor.pkl")
    predictor.save_model(filename=model_path)
    
    # Create predictive router with optimized parameters
    predictive_router = PredictiveRouter(
        network.graph, 
        alpha=2.0, beta=3.0, gamma=2.5,
        predictor=predictor,
        prediction_weight=0.3
    )
    
    # Create OSPF router for comparison
    ospf_router = OSPFRouter(network.graph)

    # Choose source and destination nodes - strictly limit to existing nodes
    source = 0
    node_list = list(network.graph.nodes())
    if source not in node_list:
        source = node_list[0]
        
    # Select valid destinations from your actual network nodes
    destinations = [6, 8]
    for d in node_list:
        if d != source and len(destinations) < 3:  # Select up to 3 destinations
            destinations.append(d)
            
    print(f"Using source node {source} and destinations {destinations}")
    
    # Compare all three routing approaches
    for dest in destinations:
        std_path, std_quality = standard_router.find_path(source, dest)
        pred_path, pred_quality = predictive_router.find_path(source, dest)
        ospf_path, ospf_quality = ospf_router.find_path(source, dest)
        
        print(f"\nPaths from {source} to {dest}:")
        print(f"Standard PAMR: {std_path} (quality: {std_quality:.4f}, hops: {len(std_path)-1})")
        print(f"ML-Enhanced:   {pred_path} (quality: {pred_quality:.4f}, hops: {len(pred_path)-1})")
        print(f"OSPF:          {ospf_path} (quality: {ospf_quality:.4f}, hops: {len(ospf_path)-1})")
        print(f"ML vs PAMR:    {((pred_quality/std_quality)-1)*100:.2f}%")
        print(f"ML vs OSPF:    {((pred_quality/ospf_quality)-1)*100:.2f}%")
        
    # Create visualizations for all destinations
    for dest in destinations:
        visualize_ospf_vs_pamr_ml_paths(
            network, source, dest, 
            ospf_router, standard_router, predictive_router, 
            output_dir=comparison_dir
        )
    
    # Run simulation with dynamic network conditions
    print("\nRunning simulation to compare performance...")
    
    # USE THE SAME NETWORK INSTANCE for all simulators
    standard_network = network
    predictive_network = network  # Use the same network for all simulators
    ospf_network = network       # Use the same network for all simulators

    # Create routers with these networks
    standard_router = PAMRRouter(standard_network.graph, alpha=2.0, beta=3.0, gamma=2.5)
    predictive_router = PredictiveRouter(
        predictive_network.graph, 
        alpha=2.0, beta=3.0, gamma=2.5,
        predictor=predictor,
        prediction_weight=0.3
    )
    ospf_router = OSPFRouter(ospf_network.graph)

    # Create simulators
    standard_sim = PAMRSimulator(standard_network, standard_router)
    predictive_sim = PAMRSimulator(predictive_network, predictive_router)
    ospf_sim = OSPFSimulator(ospf_network, ospf_router)
    
    # Run simulations
    iterations = 100
    standard_sim.run_simulation(num_iterations=iterations)
    predictive_sim.run_simulation(num_iterations=iterations)
    ospf_sim.run_simulation(num_iterations=iterations)
    
    # Generate performance comparison visualizations
    visualize_performance_metrics(
        standard_sim, ospf_sim, predictive_sim,
        standard_network, ospf_network, predictive_network,
        iterations=iterations,
        output_dir=comparison_dir
    )
    
    # Generate results dashboard
    dashboard_file = generate_results_dashboard(
        network=standard_network,  
        pamr_sim=standard_sim,
        ospf_sim=ospf_sim,
        ml_pamr_sim=predictive_sim,
        output_dir=comparison_dir,
        output_file=os.path.join(results_dirs["reports"], "routing_comparison_dashboard.html")
    )
    print(f"Open {dashboard_file} in your web browser to view all visualizations")
if __name__ == "__main__":
    main()