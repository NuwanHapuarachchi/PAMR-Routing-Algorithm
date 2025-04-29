import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import time
from datetime import datetime
import csv
import networkx as nx
from tabulate import tabulate

# Add parent directory to path to import pamr package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PAMR router and network
from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter

# Simple RIP Router implementation
class RIPRouter:
    """Simplified implementation of RIP routing."""
    
    def __init__(self, topology):
        self.topology = topology
        self.routing_tables = {}
        self.update_tables()
        
    def update_tables(self):
        """Update routing tables based on current topology."""
        nodes = list(self.topology.keys())
        
        # Initialize routing tables
        for node in nodes:
            self.routing_tables[node] = {}
            for dest in nodes:
                if node == dest:
                    # Route to self
                    self.routing_tables[node][dest] = ([node], 0)
                elif dest in self.topology[node]:
                    # Direct neighbor
                    self.routing_tables[node][dest] = ([node, dest], self.topology[node][dest])
                else:
                    # Unknown route
                    self.routing_tables[node][dest] = (None, float('infinity'))
        
        # Run Bellman-Ford algorithm (simplified RIP)
        changed = True
        max_iterations = 10  # Limit iterations for performance
        iterations = 0
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for node in nodes:
                for neighbor in self.topology[node]:
                    neighbor_table = self.routing_tables[neighbor]
                    
                    for dest in nodes:
                        if dest == node:
                            continue
                            
                        # Current distance
                        _, current_dist = self.routing_tables[node][dest]
                        
                        # Distance through neighbor
                        neighbor_path, neighbor_dist = neighbor_table[dest]
                        
                        if neighbor_path is not None:
                            total_dist = self.topology[node][neighbor] + neighbor_dist
                            
                            if total_dist < current_dist:
                                # Found better path
                                new_path = [node]
                                if neighbor_path:
                                    new_path.extend(neighbor_path[1:])  # Skip the neighbor
                                self.routing_tables[node][dest] = (new_path, total_dist)
                                changed = True
    
    def find_path(self, source, destination):
        """Find the path from source to destination."""
        if source not in self.routing_tables or destination not in self.routing_tables[source]:
            self.update_tables()
            
        path, metric = self.routing_tables[source][destination]
        
        if path is None:
            return [], 0
            
        # Calculate quality using the same formula as PAMR for fair comparison
        quality = 1.0 / (metric + 1.0) if metric > 0 else 0
        
        return path, quality

# Simple OSPF Router implementation
class OSPFRouter:
    """Simplified implementation of OSPF routing algorithm."""
    
    def __init__(self, topology):
        self.topology = topology
        self.routing_tables = {}
        self.update_link_state()
    
    def update_link_state(self):
        """Update the link state database (recalculate shortest paths)."""
        self.routing_tables = {}
        
        # For each node, calculate shortest paths to all other nodes
        for source in self.topology:
            self.routing_tables[source] = {}
            distances = {node: float('infinity') for node in self.topology}
            distances[source] = 0
            predecessors = {node: None for node in self.topology}
            unvisited = list(self.topology.keys())
            
            while unvisited:
                # Find closest unvisited node
                current = min(unvisited, key=lambda x: distances[x])
                
                if distances[current] == float('infinity'):
                    break
                    
                unvisited.remove(current)
                
                # Check neighbors
                for neighbor, cost in self.topology[current].items():
                    new_distance = distances[current] + cost
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        predecessors[neighbor] = current
            
            # Reconstruct paths
            for destination in self.topology:
                if destination == source:
                    self.routing_tables[source][destination] = [source]
                    continue
                    
                if predecessors[destination] is None:
                    # No path found
                    self.routing_tables[source][destination] = []
                    continue
                    
                # Reconstruct path
                path = [destination]
                current = destination
                
                while current != source:
                    current = predecessors[current]
                    path.append(current)
                    
                path.reverse()
                self.routing_tables[source][destination] = path
    
    def find_path(self, source, destination):
        """Find the path from source to destination using OSPF routing."""
        if source not in self.routing_tables or destination not in self.routing_tables[source]:
            self.update_link_state()
            
        path = self.routing_tables[source][destination]
        
        if not path:
            return [], 0
            
        # Calculate path quality (same as in PAMR for comparison)
        total_distance = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_distance += self.topology[u][v]
            
        quality = 1.0 / (total_distance + 1.0) if total_distance > 0 else 0
        
        return path, quality

# Create a function to simulate congestion patterns
def simulate_congestion_pattern(network, pattern="random", intensity=0.5, target_nodes=None):
    """Simulate different congestion patterns in the network.
    
    Args:
        network: NetworkTopology instance
        pattern: Type of congestion pattern ("random", "hotspot", "bottleneck", "cascade")
        intensity: Intensity of congestion (0.0-1.0)
        target_nodes: Specific nodes to target (for hotspot pattern)
    """
    if pattern == "random":
        # Random congestion across the network
        for u, v in network.graph.edges():
            if random.random() < intensity:
                traffic = random.uniform(0, network.graph[u][v]['capacity'] * 0.8)
                network.graph[u][v]['traffic'] = traffic
                network.graph[u][v]['congestion'] = traffic / network.graph[u][v]['capacity']
    
    elif pattern == "hotspot":
        # Concentrated congestion around specific nodes
        if target_nodes is None:
            # Pick random nodes if not specified
            nodes = list(network.graph.nodes())
            target_nodes = random.sample(nodes, max(1, int(len(nodes) * 0.2)))
            
        for node in target_nodes:
            for neighbor in network.graph.neighbors(node):
                traffic = random.uniform(
                    network.graph[node][neighbor]['capacity'] * 0.5, 
                    network.graph[node][neighbor]['capacity'] * 0.9
                ) * intensity
                network.graph[node][neighbor]['traffic'] = traffic
                network.graph[node][neighbor]['congestion'] = traffic / network.graph[node][neighbor]['capacity']
    
    elif pattern == "bottleneck":
        # Identify critical paths and congest them
        centrality = sorted(
            network.graph.edges(), 
            key=lambda e: len(list(network.graph.neighbors(e[0]))) + len(list(network.graph.neighbors(e[1]))),
            reverse=True
        )
        
        # Congest top 10% of central edges
        bottleneck_edges = centrality[:max(1, int(len(centrality) * 0.1))]
        for u, v in bottleneck_edges:
            traffic = network.graph[u][v]['capacity'] * 0.8 * intensity
            network.graph[u][v]['traffic'] = traffic
            network.graph[u][v]['congestion'] = traffic / network.graph[u][v]['capacity']
    
    elif pattern == "cascade":
        # Start with a few congested edges and create cascade effects
        edges = list(network.graph.edges())
        initial_edges = random.sample(edges, max(1, int(len(edges) * 0.05)))
        
        # Set initial congestion
        for u, v in initial_edges:
            traffic = network.graph[u][v]['capacity'] * 0.9 * intensity
            network.graph[u][v]['traffic'] = traffic
            network.graph[u][v]['congestion'] = traffic / network.graph[u][v]['capacity']
        
        # Propagate congestion to adjacent edges (cascade effect)
        for _ in range(3):  # 3 levels of propagation
            propagation_edges = []
            for u, v in initial_edges:
                for neighbor in network.graph.neighbors(v):
                    if neighbor != u and (v, neighbor) in network.graph.edges():
                        propagation_edges.append((v, neighbor))
            
            # Apply reduced congestion to propagation edges
            intensity *= 0.7  # Reduce intensity with each step
            for u, v in propagation_edges:
                traffic = network.graph[u][v]['capacity'] * 0.7 * intensity
                network.graph[u][v]['traffic'] = traffic
                network.graph[u][v]['congestion'] = traffic / network.graph[u][v]['capacity']
            
            initial_edges = propagation_edges

# Function to demonstrate PAMR's adaptive routing capabilities
def demonstrate_adaptive_routing(output_file="adaptive_routing_comparison.csv"):
    """
    Simulate a scenario where congestion builds up on the preferred path
    and demonstrate how PAMR adapts while RIP and OSPF don't.
    """
    # Create a controlled network for the demonstration
    num_nodes = 15
    connectivity = 0.3
    seed = 42
    
    print("\n" + "="*80)
    print("DEMONSTRATING PAMR'S ADAPTIVE ROUTING CAPABILITIES")
    print("="*80)
    print(f"Network Parameters: {num_nodes} nodes, {connectivity:.2f} connectivity factor, seed {seed}")
    
    # Create network topology with controlled layout
    network = NetworkTopology(num_nodes=num_nodes, connectivity=connectivity, seed=seed)
    
    # Count connections in the network
    num_connections = network.graph.number_of_edges()
    print(f"Total number of connections in the network: {num_connections}")
    
    # Convert the NetworkX graph to a format compatible with RIP and OSPF
    topology = {}
    for node in network.graph.nodes():
        topology[node] = {}
        for neighbor in network.graph.neighbors(node):
            topology[node][neighbor] = network.graph[node][neighbor]['distance']
    
    # Initialize routers
    pamr_router = PAMRRouter(network.graph, alpha=2.0, beta=3.0, gamma=2.5)
    rip_router = RIPRouter(topology)
    ospf_router = OSPFRouter(topology)
    
    # Find suitable source-destination pairs with multiple possible paths
    # We'll pick pairs where the shortest path is at least 3 hops
    test_pairs = []
    for source in range(min(5, num_nodes)):
        for destination in range(num_nodes - 5, num_nodes):
            if source != destination:
                # Find all paths between source and destination
                try:
                    all_paths = list(nx.all_simple_paths(network.graph, source, destination, cutoff=10))
                    if len(all_paths) >= 2 and min(len(path) for path in all_paths) >= 4:
                        test_pairs.append((source, destination))
                        if len(test_pairs) >= 3:  # Limit to 3 pairs for clarity
                            break
                except nx.NetworkXNoPath:
                    continue
        if len(test_pairs) >= 3:
            break
    
    if not test_pairs:
        # If no ideal pairs found, just pick any distant pair
        source = 0
        destination = num_nodes - 1
        test_pairs = [(source, destination)]
    
    # Storage for path history
    path_history = {
        'iterations': [],
        'PAMR': {pair: [] for pair in test_pairs},
        'RIP': {pair: [] for pair in test_pairs},
        'OSPF': {pair: [] for pair in test_pairs},
        'congestion': {pair: {} for pair in test_pairs}
    }
    
    # Run the simulation for several iterations
    num_iterations = 25
    print(f"\nRunning adaptive routing simulation for {num_iterations} iterations...")
    
    # Initial path finding - measure convergence time
    convergence_times = {}
    for protocol in ['PAMR', 'RIP', 'OSPF']:
        start_time = time.time()
        if protocol == 'PAMR':
            for pair in test_pairs:
                source, destination = pair
                pamr_router.find_path(source, destination)
        elif protocol == 'RIP':
            for pair in test_pairs:
                source, destination = pair
                rip_router.find_path(source, destination)
        elif protocol == 'OSPF':
            for pair in test_pairs:
                source, destination = pair
                ospf_router.find_path(source, destination)
        convergence_times[protocol] = time.time() - start_time
    
    print("\nInitial convergence times:")
    for protocol, time_taken in convergence_times.items():
        print(f"{protocol}: {time_taken:.6f} seconds")
    
    # Get all edges in the network
    all_edges = list(network.graph.edges())
    
    # For each test pair, find the initial PAMR path
    # and gradually increase congestion along that path
    paths_to_congest = {}
    
    for iteration in range(num_iterations):
        path_history['iterations'].append(iteration)
        
        # Simulate gradually increasing congestion on the main path
        for pair in test_pairs:
            source, destination = pair
            
            if iteration == 0:
                # First iteration - find initial paths
                pamr_path, pamr_quality = pamr_router.find_path(source, destination)
                rip_path, rip_quality = rip_router.find_path(source, destination)
                ospf_path, ospf_quality = ospf_router.find_path(source, destination)
                
                # Store the initial PAMR path to congest
                paths_to_congest[pair] = pamr_path
                
                # Initialize congestion history for each edge in this path
                for i in range(len(pamr_path) - 1):
                    u, v = pamr_path[i], pamr_path[i+1]
                    path_history['congestion'][pair][(u, v)] = []
            else:
                # Apply increasing congestion to the initial path
                # Every 5 iterations, dramatically increase congestion on the path
                path_to_congest = paths_to_congest[pair]
                
                for i in range(len(path_to_congest) - 1):
                    u, v = path_to_congest[i], path_to_congest[i+1]
                    
                    # Increase traffic on this edge
                    traffic_increase = 0
                    if iteration % 5 == 0:
                        # Major traffic spike every 5 iterations
                        traffic_increase = network.graph[u][v]['capacity'] * 0.3
                    else:
                        # Gradual increase otherwise
                        traffic_increase = network.graph[u][v]['capacity'] * 0.05
                    
                    # Add traffic
                    network.graph[u][v]['traffic'] += traffic_increase
                    
                    # Recalculate congestion
                    congestion = min(0.95, network.graph[u][v]['traffic'] / network.graph[u][v]['capacity'])
                    network.graph[u][v]['congestion'] = congestion
                    
                    # Store congestion level
                    path_history['congestion'][pair][(u, v)].append(congestion)
                
                # Slightly decay traffic on other edges
                for u, v in all_edges:
                    edge_in_path = False
                    for i in range(len(path_to_congest) - 1):
                        if (u, v) == (path_to_congest[i], path_to_congest[i+1]):
                            edge_in_path = True
                            break
                    
                    if not edge_in_path:
                        network.graph[u][v]['traffic'] *= 0.9
                        network.graph[u][v]['congestion'] = network.graph[u][v]['traffic'] / network.graph[u][v]['capacity']
                
                # Update topology for RIP and OSPF
                for u, v in network.graph.edges():
                    topology[u][v] = network.graph[u][v]['distance']
                
                # Find new paths with updated congestion
                pamr_path, pamr_quality = pamr_router.find_path(source, destination)
                rip_path, rip_quality = rip_router.find_path(source, destination)
                ospf_path, ospf_quality = ospf_router.find_path(source, destination)
            
            # Store paths for this iteration
            path_history['PAMR'][pair].append({
                'path': pamr_path,
                'quality': pamr_quality,
                'length': len(pamr_path) - 1 if pamr_path else 0
            })
            
            path_history['RIP'][pair].append({
                'path': rip_path,
                'quality': rip_quality,
                'length': len(rip_path) - 1 if rip_path else 0
            })
            
            path_history['OSPF'][pair].append({
                'path': ospf_path,
                'quality': ospf_quality,
                'length': len(ospf_path) - 1 if ospf_path else 0
            })
    
    # Analyze and display the results
    print("\nAnalyzing path changes over time...")
    
    for pair in test_pairs:
        source, destination = pair
        print(f"\nSource {source} to Destination {destination}:")
        
        # Count path changes for each protocol
        pamr_path_changes = 0
        rip_path_changes = 0
        ospf_path_changes = 0
        
        for i in range(1, num_iterations):
            if str(path_history['PAMR'][pair][i]['path']) != str(path_history['PAMR'][pair][i-1]['path']):
                pamr_path_changes += 1
            if str(path_history['RIP'][pair][i]['path']) != str(path_history['RIP'][pair][i-1]['path']):
                rip_path_changes += 1
            if str(path_history['OSPF'][pair][i]['path']) != str(path_history['OSPF'][pair][i-1]['path']):
                ospf_path_changes += 1
        
        print(f"  Number of path changes: PAMR={pamr_path_changes}, RIP={rip_path_changes}, OSPF={ospf_path_changes}")
        
        # Show key iterations where PAMR adapted to changing conditions
        pamr_change_iterations = []
        pamr_paths = []
        pamr_qualities = []
        
        for i in range(1, num_iterations):
            if str(path_history['PAMR'][pair][i]['path']) != str(path_history['PAMR'][pair][i-1]['path']):
                pamr_change_iterations.append(i)
                pamr_paths.append(path_history['PAMR'][pair][i]['path'])
                pamr_qualities.append(path_history['PAMR'][pair][i]['quality'])
        
        if pamr_change_iterations:
            print("\n  Key PAMR path changes:")
            table_data = []
            for i, path, quality in zip(pamr_change_iterations, pamr_paths, pamr_qualities):
                # Get congestion levels on the original path at this iteration
                avg_congestion = 0
                num_edges = 0
                
                # Use the initial congested path for reference
                path_to_congest = paths_to_congest[pair]
                
                for j in range(len(path_to_congest) - 1):
                    u, v = path_to_congest[j], path_to_congest[j+1]
                    edge_congestion_history = path_history['congestion'][pair].get((u, v), [])
                    
                    if i - 1 < len(edge_congestion_history):  # -1 because the first iteration has no history
                        avg_congestion += edge_congestion_history[i - 1]
                        num_edges += 1
                
                if num_edges > 0:
                    avg_congestion /= num_edges
                
                table_data.append([
                    i, 
                    '→'.join(map(str, path)), 
                    len(path) - 1, 
                    quality, 
                    f"{avg_congestion:.4f}"
                ])
            
            headers = ["Iteration", "New Path", "Hops", "Quality", "Avg Congestion on Original Path"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Show final paths for comparison
            print("\n  Final paths after congestion buildup:")
            final_data = []
            
            final_pamr = path_history['PAMR'][pair][-1]
            final_rip = path_history['RIP'][pair][-1]
            final_ospf = path_history['OSPF'][pair][-1]
            
            final_data.append([
                "PAMR", 
                '→'.join(map(str, final_pamr['path'])), 
                final_pamr['length'], 
                final_pamr['quality']
            ])
            
            final_data.append([
                "RIP", 
                '→'.join(map(str, final_rip['path'])), 
                final_rip['length'], 
                final_rip['quality']
            ])
            
            final_data.append([
                "OSPF", 
                '→'.join(map(str, final_ospf['path'])), 
                final_ospf['length'], 
                final_ospf['quality']
            ])
            
            headers = ["Protocol", "Final Path", "Hops", "Quality"]
            print(tabulate(final_data, headers=headers, tablefmt="grid"))
            
            # Calculate quality improvement of PAMR's adaptivity
            initial_pamr_quality = path_history['PAMR'][pair][0]['quality']
            final_pamr_quality = path_history['PAMR'][pair][-1]['quality']
            pamr_quality_change = ((final_pamr_quality / initial_pamr_quality) - 1) * 100
            
            initial_rip_quality = path_history['RIP'][pair][0]['quality']
            final_rip_quality = path_history['RIP'][pair][-1]['quality']
            rip_quality_change = ((final_rip_quality / initial_rip_quality) - 1) * 100
            
            initial_ospf_quality = path_history['OSPF'][pair][0]['quality']
            final_ospf_quality = path_history['OSPF'][pair][-1]['quality']
            ospf_quality_change = ((final_ospf_quality / initial_ospf_quality) - 1) * 100
            
            print(f"\n  Quality change due to congestion:")
            print(f"    PAMR: {pamr_quality_change:.2f}% change")
            print(f"    RIP:  {rip_quality_change:.2f}% change")
            print(f"    OSPF: {ospf_quality_change:.2f}% change")
            
            print(f"\n  Adaptive Routing Advantage:")
            print(f"    PAMR vs RIP:  {pamr_quality_change - rip_quality_change:.2f}% better adaptation")
            print(f"    PAMR vs OSPF: {pamr_quality_change - ospf_quality_change:.2f}% better adaptation")
    
    # Create a more complete summary
    print("\n" + "="*80)
    print("ADAPTIVE ROUTING SUMMARY")
    print("="*80)
    
    print(f"Network Characteristics:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of connections: {num_connections}")
    print(f"  Average node degree: {num_connections / num_nodes:.2f}")
    
    # Calculate average path lengths
    avg_path_lengths = {protocol: 0 for protocol in ['PAMR', 'RIP', 'OSPF']}
    count = 0
    
    for pair in test_pairs:
        for protocol in ['PAMR', 'RIP', 'OSPF']:
            for iteration in range(num_iterations):
                avg_path_lengths[protocol] += path_history[protocol][pair][iteration]['length']
                count += 1
    
    for protocol in avg_path_lengths:
        avg_path_lengths[protocol] /= count if count > 0 else 1
    
    print(f"\nPath Efficiency:")
    print(f"  Average path length (hops): PAMR={avg_path_lengths['PAMR']:.2f}, RIP={avg_path_lengths['RIP']:.2f}, OSPF={avg_path_lengths['OSPF']:.2f}")
    
    # Summarize convergence times
    print(f"\nConvergence Performance:")
    for protocol, time_taken in convergence_times.items():
        print(f"  {protocol} convergence time: {time_taken:.6f} seconds")
    
    # Count total path changes for each protocol
    total_pamr_changes = 0
    total_rip_changes = 0
    total_ospf_changes = 0
    
    for pair in test_pairs:
        for i in range(1, num_iterations):
            if str(path_history['PAMR'][pair][i]['path']) != str(path_history['PAMR'][pair][i-1]['path']):
                total_pamr_changes += 1
            if str(path_history['RIP'][pair][i]['path']) != str(path_history['RIP'][pair][i-1]['path']):
                total_rip_changes += 1
            if str(path_history['OSPF'][pair][i]['path']) != str(path_history['OSPF'][pair][i-1]['path']):
                total_ospf_changes += 1
    
    print(f"\nAdaptivity to Changing Conditions:")
    print(f"  Total path changes: PAMR={total_pamr_changes}, RIP={total_rip_changes}, OSPF={total_ospf_changes}")
    adaptivity_factor_rip = total_pamr_changes / (total_rip_changes + 1)  # Avoid division by zero
    adaptivity_factor_ospf = total_pamr_changes / (total_ospf_changes + 1)
    print(f"  Adaptivity factor: PAMR is {adaptivity_factor_rip:.1f}x more adaptive than RIP")
    print(f"  Adaptivity factor: PAMR is {adaptivity_factor_ospf:.1f}x more adaptive than OSPF")
    
    # Create and save the congestion vs time graph for the first test pair
    if test_pairs:
        pair = test_pairs[0]
        source, destination = pair
        
        path_changes = []
        for i in range(1, num_iterations):
            if str(path_history['PAMR'][pair][i]['path']) != str(path_history['PAMR'][pair][i-1]['path']):
                path_changes.append(i)
        
        plt.figure(figsize=(12, 8))
        
        # Plot congestion levels
        for edge, congestion_history in path_history['congestion'][pair].items():
            if congestion_history:  # Only if there's data
                plt.plot(range(len(congestion_history)), congestion_history, 
                        label=f"Edge {edge[0]}->{edge[1]} Congestion", alpha=0.7)
        
        # Plot vertical lines at path change points
        for change_point in path_changes:
            plt.axvline(x=change_point, color='r', linestyle='--', alpha=0.5)
            plt.text(change_point, 0.1, f"Path\nChange", rotation=90, verticalalignment='bottom')
        
        # Plot quality over time for each protocol
        pamr_quality = [entry['quality'] for entry in path_history['PAMR'][pair]]
        rip_quality = [entry['quality'] for entry in path_history['RIP'][pair]]
        ospf_quality = [entry['quality'] for entry in path_history['OSPF'][pair]]
        
        plt.plot(range(num_iterations), pamr_quality, 'g-', linewidth=2, label="PAMR Path Quality")
        plt.plot(range(num_iterations), rip_quality, 'b-', linewidth=2, label="RIP Path Quality")
        plt.plot(range(num_iterations), ospf_quality, 'c-', linewidth=2, label="OSPF Path Quality")
        
        plt.xlabel('Iteration')
        plt.ylabel('Congestion / Quality')
        plt.title(f'PAMR Adaptive Routing: Source {source} to Destination {destination}')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        filename = f"adaptive_routing_{source}_to_{destination}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nAdaptive routing visualization saved to {filename}")
    
    return network, test_pairs, path_history

# Create a function to run the simulation with different parameters
def compare_routing_algorithms(network_params, test_iterations=10, traffic_pattern="random", 
                              traffic_intensity=0.5, verbose=True):
    """Run comparison with specific network parameters and congestion pattern.
    
    Args:
        network_params: Dictionary with network parameters (nodes, connectivity, etc.)
        test_iterations: Number of test iterations
        traffic_pattern: Pattern of traffic to simulate
        traffic_intensity: Intensity of traffic (0.0-1.0)
        verbose: Whether to print detailed output
        
    Returns:
        DataFrame with comparison results
    """
    # Extract network parameters
    num_nodes = network_params.get('num_nodes', 15)
    connectivity = network_params.get('connectivity', 0.3)
    seed = network_params.get('seed', 42)
    
    # Create network topology for simulation
    network = NetworkTopology(num_nodes=num_nodes, connectivity=connectivity, seed=seed)
    
    # Apply congestion pattern
    simulate_congestion_pattern(network, pattern=traffic_pattern, intensity=traffic_intensity)
    
    # Convert the NetworkX graph to a format compatible with RIP and OSPF
    topology = {}
    for node in network.graph.nodes():
        topology[node] = {}
        for neighbor in network.graph.neighbors(node):
            topology[node][neighbor] = network.graph[node][neighbor]['distance']
    
    # Initialize routers
    start_time = time.time()
    pamr_router = PAMRRouter(network.graph, alpha=2.0, beta=3.0, gamma=2.5)
    rip_router = RIPRouter(topology)
    ospf_router = OSPFRouter(topology)
    if verbose:
        print(f"Router initialization took {time.time() - start_time:.2f} seconds")
    
    # Initialize data collection
    data = []
    
    # Select test cases - random source-destination pairs
    test_cases = []
    for _ in range(test_iterations):
        source, destination = random.sample(list(network.graph.nodes()), 2)
        test_cases.append((source, destination))
    
    # Measure convergence time
    convergence_times = {
        'PAMR': 0,
        'RIP': 0,
        'OSPF': 0
    }
    
    # Test path quality and stability
    for i, (source, destination) in enumerate(test_cases):
        iter_start = time.time()
        
        # Update network conditions
        network.update_dynamic_metrics()
        
        # Update topology for RIP and OSPF
        for u, v in network.graph.edges():
            topology[u][v] = network.graph[u][v]['distance']
        
        # Measure PAMR path finding time
        pamr_start = time.time()
        pamr_path, pamr_metric = pamr_router.find_path(source, destination)
        pamr_time = time.time() - pamr_start
        
        # Measure RIP path finding time
        rip_start = time.time()
        rip_path, rip_metric = rip_router.find_path(source, destination)
        rip_time = time.time() - rip_start
        
        # Measure OSPF path finding time
        ospf_start = time.time()
        ospf_path, ospf_metric = ospf_router.find_path(source, destination)
        ospf_time = time.time() - ospf_start
        
        # Update convergence times
        convergence_times['PAMR'] += pamr_time
        convergence_times['RIP'] += rip_time
        convergence_times['OSPF'] += ospf_time
        
        # Calculate average congestion along each path
        pamr_congestion = 0
        rip_congestion = 0
        ospf_congestion = 0
        
        # Calculate PAMR path congestion
        if pamr_path and len(pamr_path) > 1:
            for i in range(len(pamr_path) - 1):
                u, v = pamr_path[i], pamr_path[i+1]
                pamr_congestion += network.graph[u][v].get('congestion', 0)
            pamr_congestion /= (len(pamr_path) - 1)
        
        # Calculate RIP path congestion
        if rip_path and len(rip_path) > 1:
            for i in range(len(rip_path) - 1):
                u, v = rip_path[i], rip_path[i+1]
                if u in topology and v in topology[u]:
                    # Find the edge in the original network
                    if network.graph.has_edge(u, v):
                        rip_congestion += network.graph[u][v].get('congestion', 0)
            rip_congestion /= (len(rip_path) - 1) if len(rip_path) > 1 else 1
        
        # Calculate OSPF path congestion
        if ospf_path and len(ospf_path) > 1:
            for i in range(len(ospf_path) - 1):
                u, v = ospf_path[i], ospf_path[i+1]
                if u in topology and v in topology[u]:
                    # Find the edge in the original network
                    if network.graph.has_edge(u, v):
                        ospf_congestion += network.graph[u][v].get('congestion', 0)
            ospf_congestion /= (len(ospf_path) - 1) if len(ospf_path) > 1 else 1
        
        if verbose:
            print(f"Test case {source} → {destination}:")
            print(f"  PAMR: {pamr_path} (Quality: {pamr_metric:.4f}, Congestion: {pamr_congestion:.4f})")
            print(f"  RIP:  {rip_path} (Quality: {rip_metric:.4f}, Congestion: {rip_congestion:.4f})")
            print(f"  OSPF: {ospf_path} (Quality: {ospf_metric:.4f}, Congestion: {ospf_congestion:.4f})")
            print(f"  Iteration took {time.time() - iter_start:.4f} seconds")
            print("-" * 50)
        
        # Collect metrics
        data.append({
            'Source': source,
            'Destination': destination,
            'PAMR Path': pamr_path,
            'PAMR Path Length': len(pamr_path) - 1 if pamr_path else 0,  
            'PAMR Quality': pamr_metric,
            'PAMR Congestion': pamr_congestion,
            'PAMR Time': pamr_time,
            'RIP Path': rip_path,
            'RIP Path Length': len(rip_path) - 1 if rip_path else 0,
            'RIP Quality': rip_metric,
            'RIP Congestion': rip_congestion,
            'RIP Time': rip_time,
            'OSPF Path': ospf_path,
            'OSPF Path Length': len(ospf_path) - 1 if ospf_path else 0,
            'OSPF Quality': ospf_metric,
            'OSPF Congestion': ospf_congestion,
            'OSPF Time': ospf_time
        })
    
    # Calculate average convergence times
    for protocol in convergence_times:
        convergence_times[protocol] /= len(test_cases)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add summary row with average metrics
    summary = {
        'Network Size': num_nodes,
        'Connectivity': connectivity,
        'Traffic Pattern': traffic_pattern,
        'Traffic Intensity': traffic_intensity,
        'PAMR Avg Quality': df['PAMR Quality'].mean(),
        'RIP Avg Quality': df['RIP Quality'].mean(),
        'OSPF Avg Quality': df['OSPF Quality'].mean(),
        'PAMR Avg Congestion': df['PAMR Congestion'].mean(),
        'RIP Avg Congestion': df['RIP Congestion'].mean(),
        'OSPF Avg Congestion': df['OSPF Congestion'].mean(),
        'PAMR Avg Path Length': df['PAMR Path Length'].mean(),
        'RIP Avg Path Length': df['RIP Path Length'].mean(),
        'OSPF Avg Path Length': df['OSPF Path Length'].mean(),
        'PAMR Convergence Time': convergence_times['PAMR'],
        'RIP Convergence Time': convergence_times['RIP'],
        'OSPF Convergence Time': convergence_times['OSPF']
    }
    
    # Calculate improvement percentages
    summary['PAMR vs RIP Quality %'] = ((summary['PAMR Avg Quality'] / summary['RIP Avg Quality']) - 1) * 100
    summary['PAMR vs OSPF Quality %'] = ((summary['PAMR Avg Quality'] / summary['OSPF Avg Quality']) - 1) * 100
    summary['PAMR vs RIP Congestion %'] = ((summary['RIP Avg Congestion'] / summary['PAMR Avg Congestion']) - 1) * 100
    summary['PAMR vs OSPF Congestion %'] = ((summary['OSPF Avg Congestion'] / summary['PAMR Avg Congestion']) - 1) * 100
    
    return df, summary

# Run comprehensive testing across different network configurations
def run_comprehensive_testing(output_file="protocol_comparison_results.csv"):
    """Run comprehensive testing of routing protocols across different scenarios."""
    # Define network configurations to test
    network_configs = [
        {'num_nodes': 10, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Small Network'},
        {'num_nodes': 10, 'connectivity': 0.5, 'seed': 42, 'name': 'Dense Small Network'},
        {'num_nodes': 20, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Medium Network'},
        {'num_nodes': 20, 'connectivity': 0.4, 'seed': 42, 'name': 'Dense Medium Network'},
        {'num_nodes': 30, 'connectivity': 0.1, 'seed': 42, 'name': 'Sparse Large Network'},
        {'num_nodes': 30, 'connectivity': 0.3, 'seed': 42, 'name': 'Dense Large Network'}
    ]
    
    # Define traffic patterns to test
    traffic_patterns = [
        {'pattern': 'random', 'intensity': 0.3, 'name': 'Light Random Traffic'},
        {'pattern': 'random', 'intensity': 0.7, 'name': 'Heavy Random Traffic'},
        {'pattern': 'hotspot', 'intensity': 0.5, 'name': 'Hotspot Traffic'},
        {'pattern': 'bottleneck', 'intensity': 0.6, 'name': 'Bottleneck Traffic'},
        {'pattern': 'cascade', 'intensity': 0.8, 'name': 'Cascade Failure'}
    ]
    
    # Initialize results storage
    all_summaries = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROUTING PROTOCOL COMPARISON")
    print("="*80)
    
    # Run comparison for each combination
    for net_config in network_configs:
        for traffic in traffic_patterns:
            config_name = f"{net_config['name']} with {traffic['name']}"
            print(f"\nTesting configuration: {config_name}")
            print("-"*80)
            
            # Run comparison
            _, summary = compare_routing_algorithms(
                network_params=net_config,
                test_iterations=5,  # Reduced iterations for faster total testing
                traffic_pattern=traffic['pattern'],
                traffic_intensity=traffic['intensity'],
                verbose=False
            )
            
            # Add configuration name
            summary['Configuration'] = config_name
            all_summaries.append(summary)
            
            # Print key results
            print(f"PAMR Quality: {summary['PAMR Avg Quality']:.4f}, RIP Quality: {summary['RIP Avg Quality']:.4f}, OSPF Quality: {summary['OSPF Avg Quality']:.4f}")
            print(f"PAMR vs RIP Quality: +{summary['PAMR vs RIP Quality %']:.2f}%, PAMR vs OSPF Quality: +{summary['PAMR vs OSPF Quality %']:.2f}%")
            print(f"PAMR Congestion: {summary['PAMR Avg Congestion']:.4f}, RIP Congestion: {summary['RIP Avg Congestion']:.4f}, OSPF Congestion: {summary['OSPF Avg Congestion']:.4f}")
            print(f"PAMR vs RIP Congestion Reduction: {summary['PAMR vs RIP Congestion %']:.2f}%, PAMR vs OSPF Congestion Reduction: {summary['PAMR vs OSPF Congestion %']:.2f}%")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    avg_pamr_vs_rip_quality = summary_df['PAMR vs RIP Quality %'].mean()
    avg_pamr_vs_ospf_quality = summary_df['PAMR vs OSPF Quality %'].mean()
    avg_pamr_vs_rip_congestion = summary_df['PAMR vs RIP Congestion %'].mean()
    avg_pamr_vs_ospf_congestion = summary_df['PAMR vs OSPF Congestion %'].mean()
    
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs RIP: +{avg_pamr_vs_rip_quality:.2f}%")
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs OSPF: +{avg_pamr_vs_ospf_quality:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs RIP: {avg_pamr_vs_rip_congestion:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs OSPF: {avg_pamr_vs_ospf_congestion:.2f}%")
    
    # Find best and worst scenarios for PAMR
    best_vs_rip = summary_df.loc[summary_df['PAMR vs RIP Quality %'].idxmax()]
    best_vs_ospf = summary_df.loc[summary_df['PAMR vs OSPF Quality %'].idxmax()]
    
    print("\nBEST PAMR PERFORMANCE vs RIP:")
    print(f"Configuration: {best_vs_rip['Configuration']}")
    print(f"Quality Improvement: +{best_vs_rip['PAMR vs RIP Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_rip['PAMR vs RIP Congestion %']:.2f}%")
    
    print("\nBEST PAMR PERFORMANCE vs OSPF:")
    print(f"Configuration: {best_vs_ospf['Configuration']}")
    print(f"Quality Improvement: +{best_vs_ospf['PAMR vs OSPF Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_ospf['PAMR vs OSPF Congestion %']:.2f}%")
    
    print(f"\nDetailed results saved to {output_file}")
    
    return summary_df

# Run comprehensive testing across different network configurations
def run_comprehensive_testing(output_file="protocol_comparison_results.csv"):
    """Run comprehensive testing of routing protocols across different scenarios."""
    # Define network configurations to test
    network_configs = [
        {'num_nodes': 10, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Small Network'},
        {'num_nodes': 10, 'connectivity': 0.5, 'seed': 42, 'name': 'Dense Small Network'},
        {'num_nodes': 20, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Medium Network'},
        {'num_nodes': 20, 'connectivity': 0.4, 'seed': 42, 'name': 'Dense Medium Network'},
        {'num_nodes': 30, 'connectivity': 0.1, 'seed': 42, 'name': 'Sparse Large Network'},
        {'num_nodes': 30, 'connectivity': 0.3, 'seed': 42, 'name': 'Dense Large Network'}
    ]
    
    # Define traffic patterns to test
    traffic_patterns = [
        {'pattern': 'random', 'intensity': 0.3, 'name': 'Light Random Traffic'},
        {'pattern': 'random', 'intensity': 0.7, 'name': 'Heavy Random Traffic'},
        {'pattern': 'hotspot', 'intensity': 0.5, 'name': 'Hotspot Traffic'},
        {'pattern': 'bottleneck', 'intensity': 0.6, 'name': 'Bottleneck Traffic'},
        {'pattern': 'cascade', 'intensity': 0.8, 'name': 'Cascade Failure'}
    ]
    
    # Initialize results storage
    all_summaries = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROUTING PROTOCOL COMPARISON")
    print("="*80)
    
    # Run comparison for each combination
    for net_config in network_configs:
        for traffic in traffic_patterns:
            config_name = f"{net_config['name']} with {traffic['name']}"
            print(f"\nTesting configuration: {config_name}")
            print("-"*80)
            
            # Run comparison
            _, summary = compare_routing_algorithms(
                network_params=net_config,
                test_iterations=5,  # Reduced iterations for faster total testing
                traffic_pattern=traffic['pattern'],
                traffic_intensity=traffic['intensity'],
                verbose=False
            )
            
            # Add configuration name
            summary['Configuration'] = config_name
            all_summaries.append(summary)
            
            # Print key results
            print(f"PAMR Quality: {summary['PAMR Avg Quality']:.4f}, RIP Quality: {summary['RIP Avg Quality']:.4f}, OSPF Quality: {summary['OSPF Avg Quality']:.4f}")
            print(f"PAMR vs RIP Quality: +{summary['PAMR vs RIP Quality %']:.2f}%, PAMR vs OSPF Quality: +{summary['PAMR vs OSPF Quality %']:.2f}%")
            print(f"PAMR Congestion: {summary['PAMR Avg Congestion']:.4f}, RIP Congestion: {summary['RIP Avg Congestion']:.4f}, OSPF Congestion: {summary['OSPF Avg Congestion']:.4f}")
            print(f"PAMR vs RIP Congestion Reduction: {summary['PAMR vs RIP Congestion %']:.2f}%, PAMR vs OSPF Congestion Reduction: {summary['PAMR vs OSPF Congestion %']:.2f}%")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    avg_pamr_vs_rip_quality = summary_df['PAMR vs RIP Quality %'].mean()
    avg_pamr_vs_ospf_quality = summary_df['PAMR vs OSPF Quality %'].mean()
    avg_pamr_vs_rip_congestion = summary_df['PAMR vs RIP Congestion %'].mean()
    avg_pamr_vs_ospf_congestion = summary_df['PAMR vs OSPF Congestion %'].mean()
    
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs RIP: +{avg_pamr_vs_rip_quality:.2f}%")
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs OSPF: +{avg_pamr_vs_ospf_quality:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs RIP: {avg_pamr_vs_rip_congestion:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs OSPF: {avg_pamr_vs_ospf_congestion:.2f}%")
    
    # Find best and worst scenarios for PAMR
    best_vs_rip = summary_df.loc[summary_df['PAMR vs RIP Quality %'].idxmax()]
    best_vs_ospf = summary_df.loc[summary_df['PAMR vs OSPF Quality %'].idxmax()]
    
    print("\nBEST PAMR PERFORMANCE vs RIP:")
    print(f"Configuration: {best_vs_rip['Configuration']}")
    print(f"Quality Improvement: +{best_vs_rip['PAMR vs RIP Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_rip['PAMR vs RIP Congestion %']:.2f}%")
    
    print("\nBEST PAMR PERFORMANCE vs OSPF:")
    print(f"Configuration: {best_vs_ospf['Configuration']}")
    print(f"Quality Improvement: +{best_vs_ospf['PAMR vs OSPF Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_ospf['PAMR vs OSPF Congestion %']:.2f}%")
    
    print(f"\nDetailed results saved to {output_file}")
    
    return summary_df

# Run comprehensive testing across different network configurations
def run_comprehensive_testing(output_file="protocol_comparison_results.csv"):
    """Run comprehensive testing of routing protocols across different scenarios."""
    # Define network configurations to test
    network_configs = [
        {'num_nodes': 10, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Small Network'},
        {'num_nodes': 10, 'connectivity': 0.5, 'seed': 42, 'name': 'Dense Small Network'},
        {'num_nodes': 20, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Medium Network'},
        {'num_nodes': 20, 'connectivity': 0.4, 'seed': 42, 'name': 'Dense Medium Network'},
        {'num_nodes': 30, 'connectivity': 0.1, 'seed': 42, 'name': 'Sparse Large Network'},
        {'num_nodes': 30, 'connectivity': 0.3, 'seed': 42, 'name': 'Dense Large Network'}
    ]
    
    # Define traffic patterns to test
    traffic_patterns = [
        {'pattern': 'random', 'intensity': 0.3, 'name': 'Light Random Traffic'},
        {'pattern': 'random', 'intensity': 0.7, 'name': 'Heavy Random Traffic'},
        {'pattern': 'hotspot', 'intensity': 0.5, 'name': 'Hotspot Traffic'},
        {'pattern': 'bottleneck', 'intensity': 0.6, 'name': 'Bottleneck Traffic'},
        {'pattern': 'cascade', 'intensity': 0.8, 'name': 'Cascade Failure'}
    ]
    
    # Initialize results storage
    all_summaries = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROUTING PROTOCOL COMPARISON")
    print("="*80)
    
    # Run comparison for each combination
    for net_config in network_configs:
        for traffic in traffic_patterns:
            config_name = f"{net_config['name']} with {traffic['name']}"
            print(f"\nTesting configuration: {config_name}")
            print("-"*80)
            
            # Run comparison
            _, summary = compare_routing_algorithms(
                network_params=net_config,
                test_iterations=5,  # Reduced iterations for faster total testing
                traffic_pattern=traffic['pattern'],
                traffic_intensity=traffic['intensity'],
                verbose=False
            )
            
            # Add configuration name
            summary['Configuration'] = config_name
            all_summaries.append(summary)
            
            # Print key results
            print(f"PAMR Quality: {summary['PAMR Avg Quality']:.4f}, RIP Quality: {summary['RIP Avg Quality']:.4f}, OSPF Quality: {summary['OSPF Avg Quality']:.4f}")
            print(f"PAMR vs RIP Quality: +{summary['PAMR vs RIP Quality %']:.2f}%, PAMR vs OSPF Quality: +{summary['PAMR vs OSPF Quality %']:.2f}%")
            print(f"PAMR Congestion: {summary['PAMR Avg Congestion']:.4f}, RIP Congestion: {summary['RIP Avg Congestion']:.4f}, OSPF Congestion: {summary['OSPF Avg Congestion']:.4f}")
            print(f"PAMR vs RIP Congestion Reduction: {summary['PAMR vs RIP Congestion %']:.2f}%, PAMR vs OSPF Congestion Reduction: {summary['PAMR vs OSPF Congestion %']:.2f}%")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    avg_pamr_vs_rip_quality = summary_df['PAMR vs RIP Quality %'].mean()
    avg_pamr_vs_ospf_quality = summary_df['PAMR vs OSPF Quality %'].mean()
    avg_pamr_vs_rip_congestion = summary_df['PAMR vs RIP Congestion %'].mean()
    avg_pamr_vs_ospf_congestion = summary_df['PAMR vs OSPF Congestion %'].mean()
    
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs RIP: +{avg_pamr_vs_rip_quality:.2f}%")
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs OSPF: +{avg_pamr_vs_ospf_quality:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs RIP: {avg_pamr_vs_rip_congestion:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs OSPF: {avg_pamr_vs_ospf_congestion:.2f}%")
    
    # Find best and worst scenarios for PAMR
    best_vs_rip = summary_df.loc[summary_df['PAMR vs RIP Quality %'].idxmax()]
    best_vs_ospf = summary_df.loc[summary_df['PAMR vs OSPF Quality %'].idxmax()]
    
    print("\nBEST PAMR PERFORMANCE vs RIP:")
    print(f"Configuration: {best_vs_rip['Configuration']}")
    print(f"Quality Improvement: +{best_vs_rip['PAMR vs RIP Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_rip['PAMR vs RIP Congestion %']:.2f}%")
    
    print("\nBEST PAMR PERFORMANCE vs OSPF:")
    print(f"Configuration: {best_vs_ospf['Configuration']}")
    print(f"Quality Improvement: +{best_vs_ospf['PAMR vs OSPF Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_ospf['PAMR vs OSPF Congestion %']:.2f}%")
    
    print(f"\nDetailed results saved to {output_file}")
    
    return summary_df

# Run comprehensive testing across different network configurations
def run_comprehensive_testing(output_file="protocol_comparison_results.csv"):
    """Run comprehensive testing of routing protocols across different scenarios."""
    # Define network configurations to test
    network_configs = [
        {'num_nodes': 10, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Small Network'},
        {'num_nodes': 10, 'connectivity': 0.5, 'seed': 42, 'name': 'Dense Small Network'},
        {'num_nodes': 20, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Medium Network'},
        {'num_nodes': 20, 'connectivity': 0.4, 'seed': 42, 'name': 'Dense Medium Network'},
        {'num_nodes': 30, 'connectivity': 0.1, 'seed': 42, 'name': 'Sparse Large Network'},
        {'num_nodes': 30, 'connectivity': 0.3, 'seed': 42, 'name': 'Dense Large Network'}
    ]
    
    # Define traffic patterns to test
    traffic_patterns = [
        {'pattern': 'random', 'intensity': 0.3, 'name': 'Light Random Traffic'},
        {'pattern': 'random', 'intensity': 0.7, 'name': 'Heavy Random Traffic'},
        {'pattern': 'hotspot', 'intensity': 0.5, 'name': 'Hotspot Traffic'},
        {'pattern': 'bottleneck', 'intensity': 0.6, 'name': 'Bottleneck Traffic'},
        {'pattern': 'cascade', 'intensity': 0.8, 'name': 'Cascade Failure'}
    ]
    
    # Initialize results storage
    all_summaries = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROUTING PROTOCOL COMPARISON")
    print("="*80)
    
    # Run comparison for each combination
    for net_config in network_configs:
        for traffic in traffic_patterns:
            config_name = f"{net_config['name']} with {traffic['name']}"
            print(f"\nTesting configuration: {config_name}")
            print("-"*80)
            
            # Run comparison
            _, summary = compare_routing_algorithms(
                network_params=net_config,
                test_iterations=5,  # Reduced iterations for faster total testing
                traffic_pattern=traffic['pattern'],
                traffic_intensity=traffic['intensity'],
                verbose=False
            )
            
            # Add configuration name
            summary['Configuration'] = config_name
            all_summaries.append(summary)
            
            # Print key results
            print(f"PAMR Quality: {summary['PAMR Avg Quality']:.4f}, RIP Quality: {summary['RIP Avg Quality']:.4f}, OSPF Quality: {summary['OSPF Avg Quality']:.4f}")
            print(f"PAMR vs RIP Quality: +{summary['PAMR vs RIP Quality %']:.2f}%, PAMR vs OSPF Quality: +{summary['PAMR vs OSPF Quality %']:.2f}%")
            print(f"PAMR Congestion: {summary['PAMR Avg Congestion']:.4f}, RIP Congestion: {summary['RIP Avg Congestion']:.4f}, OSPF Congestion: {summary['OSPF Avg Congestion']:.4f}")
            print(f"PAMR vs RIP Congestion Reduction: {summary['PAMR vs RIP Congestion %']:.2f}%, PAMR vs OSPF Congestion Reduction: {summary['PAMR vs OSPF Congestion %']:.2f}%")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    avg_pamr_vs_rip_quality = summary_df['PAMR vs RIP Quality %'].mean()
    avg_pamr_vs_ospf_quality = summary_df['PAMR vs OSPF Quality %'].mean()
    avg_pamr_vs_rip_congestion = summary_df['PAMR vs RIP Congestion %'].mean()
    avg_pamr_vs_ospf_congestion = summary_df['PAMR vs OSPF Congestion %'].mean()
    
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs RIP: +{avg_pamr_vs_rip_quality:.2f}%")
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs OSPF: +{avg_pamr_vs_ospf_quality:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs RIP: {avg_pamr_vs_rip_congestion:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs OSPF: {avg_pamr_vs_ospf_congestion:.2f}%")
    
    # Find best and worst scenarios for PAMR
    best_vs_rip = summary_df.loc[summary_df['PAMR vs RIP Quality %'].idxmax()]
    best_vs_ospf = summary_df.loc[summary_df['PAMR vs OSPF Quality %'].idxmax()]
    
    print("\nBEST PAMR PERFORMANCE vs RIP:")
    print(f"Configuration: {best_vs_rip['Configuration']}")
    print(f"Quality Improvement: +{best_vs_rip['PAMR vs RIP Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_rip['PAMR vs RIP Congestion %']:.2f}%")
    
    print("\nBEST PAMR PERFORMANCE vs OSPF:")
    print(f"Configuration: {best_vs_ospf['Configuration']}")
    print(f"Quality Improvement: +{best_vs_ospf['PAMR vs OSPF Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_ospf['PAMR vs OSPF Congestion %']:.2f}%")
    
    print(f"\nDetailed results saved to {output_file}")
    
    return summary_df

# Run comprehensive testing across different network configurations
def run_comprehensive_testing(output_file="protocol_comparison_results.csv"):
    """Run comprehensive testing of routing protocols across different scenarios."""
    # Define network configurations to test
    network_configs = [
        {'num_nodes': 10, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Small Network'},
        {'num_nodes': 10, 'connectivity': 0.5, 'seed': 42, 'name': 'Dense Small Network'},
        {'num_nodes': 20, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Medium Network'},
        {'num_nodes': 20, 'connectivity': 0.4, 'seed': 42, 'name': 'Dense Medium Network'},
        {'num_nodes': 30, 'connectivity': 0.1, 'seed': 42, 'name': 'Sparse Large Network'},
        {'num_nodes': 30, 'connectivity': 0.3, 'seed': 42, 'name': 'Dense Large Network'}
    ]
    
    # Define traffic patterns to test
    traffic_patterns = [
        {'pattern': 'random', 'intensity': 0.3, 'name': 'Light Random Traffic'},
        {'pattern': 'random', 'intensity': 0.7, 'name': 'Heavy Random Traffic'},
        {'pattern': 'hotspot', 'intensity': 0.5, 'name': 'Hotspot Traffic'},
        {'pattern': 'bottleneck', 'intensity': 0.6, 'name': 'Bottleneck Traffic'},
        {'pattern': 'cascade', 'intensity': 0.8, 'name': 'Cascade Failure'}
    ]
    
    # Initialize results storage
    all_summaries = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROUTING PROTOCOL COMPARISON")
    print("="*80)
    
    # Run comparison for each combination
    for net_config in network_configs:
        for traffic in traffic_patterns:
            config_name = f"{net_config['name']} with {traffic['name']}"
            print(f"\nTesting configuration: {config_name}")
            print("-"*80)
            
            # Run comparison
            _, summary = compare_routing_algorithms(
                network_params=net_config,
                test_iterations=5,  # Reduced iterations for faster total testing
                traffic_pattern=traffic['pattern'],
                traffic_intensity=traffic['intensity'],
                verbose=False
            )
            
            # Add configuration name
            summary['Configuration'] = config_name
            all_summaries.append(summary)
            
            # Print key results
            print(f"PAMR Quality: {summary['PAMR Avg Quality']:.4f}, RIP Quality: {summary['RIP Avg Quality']:.4f}, OSPF Quality: {summary['OSPF Avg Quality']:.4f}")
            print(f"PAMR vs RIP Quality: +{summary['PAMR vs RIP Quality %']:.2f}%, PAMR vs OSPF Quality: +{summary['PAMR vs OSPF Quality %']:.2f}%")
            print(f"PAMR Congestion: {summary['PAMR Avg Congestion']:.4f}, RIP Congestion: {summary['RIP Avg Congestion']:.4f}, OSPF Congestion: {summary['OSPF Avg Congestion']:.4f}")
            print(f"PAMR vs RIP Congestion Reduction: {summary['PAMR vs RIP Congestion %']:.2f}%, PAMR vs OSPF Congestion Reduction: {summary['PAMR vs OSPF Congestion %']:.2f}%")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    avg_pamr_vs_rip_quality = summary_df['PAMR vs RIP Quality %'].mean()
    avg_pamr_vs_ospf_quality = summary_df['PAMR vs OSPF Quality %'].mean()
    avg_pamr_vs_rip_congestion = summary_df['PAMR vs RIP Congestion %'].mean()
    avg_pamr_vs_ospf_congestion = summary_df['PAMR vs OSPF Congestion %'].mean()
    
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs RIP: +{avg_pamr_vs_rip_quality:.2f}%")
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs OSPF: +{avg_pamr_vs_ospf_quality:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs RIP: {avg_pamr_vs_rip_congestion:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs OSPF: {avg_pamr_vs_ospf_congestion:.2f}%")
    
    # Find best and worst scenarios for PAMR
    best_vs_rip = summary_df.loc[summary_df['PAMR vs RIP Quality %'].idxmax()]
    best_vs_ospf = summary_df.loc[summary_df['PAMR vs OSPF Quality %'].idxmax()]
    
    print("\nBEST PAMR PERFORMANCE vs RIP:")
    print(f"Configuration: {best_vs_rip['Configuration']}")
    print(f"Quality Improvement: +{best_vs_rip['PAMR vs RIP Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_rip['PAMR vs RIP Congestion %']:.2f}%")
    
    print("\nBEST PAMR PERFORMANCE vs OSPF:")
    print(f"Configuration: {best_vs_ospf['Configuration']}")
    print(f"Quality Improvement: +{best_vs_ospf['PAMR vs OSPF Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_ospf['PAMR vs OSPF Congestion %']:.2f}%")
    
    print(f"\nDetailed results saved to {output_file}")
    
    return summary_df

# Run comprehensive testing across different network configurations
def run_comprehensive_testing(output_file="protocol_comparison_results.csv"):
    """Run comprehensive testing of routing protocols across different scenarios."""
    # Define network configurations to test
    network_configs = [
        {'num_nodes': 10, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Small Network'},
        {'num_nodes': 10, 'connectivity': 0.5, 'seed': 42, 'name': 'Dense Small Network'},
        {'num_nodes': 20, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Medium Network'},
        {'num_nodes': 20, 'connectivity': 0.4, 'seed': 42, 'name': 'Dense Medium Network'},
        {'num_nodes': 30, 'connectivity': 0.1, 'seed': 42, 'name': 'Sparse Large Network'},
        {'num_nodes': 30, 'connectivity': 0.3, 'seed': 42, 'name': 'Dense Large Network'}
    ]
    
    # Define traffic patterns to test
    traffic_patterns = [
        {'pattern': 'random', 'intensity': 0.3, 'name': 'Light Random Traffic'},
        {'pattern': 'random', 'intensity': 0.7, 'name': 'Heavy Random Traffic'},
        {'pattern': 'hotspot', 'intensity': 0.5, 'name': 'Hotspot Traffic'},
        {'pattern': 'bottleneck', 'intensity': 0.6, 'name': 'Bottleneck Traffic'},
        {'pattern': 'cascade', 'intensity': 0.8, 'name': 'Cascade Failure'}
    ]
    
    # Initialize results storage
    all_summaries = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROUTING PROTOCOL COMPARISON")
    print("="*80)
    
    # Run comparison for each combination
    for net_config in network_configs:
        for traffic in traffic_patterns:
            config_name = f"{net_config['name']} with {traffic['name']}"
            print(f"\nTesting configuration: {config_name}")
            print("-"*80)
            
            # Run comparison
            _, summary = compare_routing_algorithms(
                network_params=net_config,
                test_iterations=5,  # Reduced iterations for faster total testing
                traffic_pattern=traffic['pattern'],
                traffic_intensity=traffic['intensity'],
                verbose=False
            )
            
            # Add configuration name
            summary['Configuration'] = config_name
            all_summaries.append(summary)
            
            # Print key results
            print(f"PAMR Quality: {summary['PAMR Avg Quality']:.4f}, RIP Quality: {summary['RIP Avg Quality']:.4f}, OSPF Quality: {summary['OSPF Avg Quality']:.4f}")
            print(f"PAMR vs RIP Quality: +{summary['PAMR vs RIP Quality %']:.2f}%, PAMR vs OSPF Quality: +{summary['PAMR vs OSPF Quality %']:.2f}%")
            print(f"PAMR Congestion: {summary['PAMR Avg Congestion']:.4f}, RIP Congestion: {summary['RIP Avg Congestion']:.4f}, OSPF Congestion: {summary['OSPF Avg Congestion']:.4f}")
            print(f"PAMR vs RIP Congestion Reduction: {summary['PAMR vs RIP Congestion %']:.2f}%, PAMR vs OSPF Congestion Reduction: {summary['PAMR vs OSPF Congestion %']:.2f}%")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    avg_pamr_vs_rip_quality = summary_df['PAMR vs RIP Quality %'].mean()
    avg_pamr_vs_ospf_quality = summary_df['PAMR vs OSPF Quality %'].mean()
    avg_pamr_vs_rip_congestion = summary_df['PAMR vs RIP Congestion %'].mean()
    avg_pamr_vs_ospf_congestion = summary_df['PAMR vs OSPF Congestion %'].mean()
    
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs RIP: +{avg_pamr_vs_rip_quality:.2f}%")
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs OSPF: +{avg_pamr_vs_ospf_quality:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs RIP: {avg_pamr_vs_rip_congestion:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs OSPF: {avg_pamr_vs_ospf_congestion:.2f}%")
    
    # Find best and worst scenarios for PAMR
    best_vs_rip = summary_df.loc[summary_df['PAMR vs RIP Quality %'].idxmax()]
    best_vs_ospf = summary_df.loc[summary_df['PAMR vs OSPF Quality %'].idxmax()]
    
    print("\nBEST PAMR PERFORMANCE vs RIP:")
    print(f"Configuration: {best_vs_rip['Configuration']}")
    print(f"Quality Improvement: +{best_vs_rip['PAMR vs RIP Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_rip['PAMR vs RIP Congestion %']:.2f}%")
    
    print("\nBEST PAMR PERFORMANCE vs OSPF:")
    print(f"Configuration: {best_vs_ospf['Configuration']}")
    print(f"Quality Improvement: +{best_vs_ospf['PAMR vs OSPF Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_ospf['PAMR vs OSPF Congestion %']:.2f}%")
    
    print(f"\nDetailed results saved to {output_file}")
    
    return summary_df

# Run comprehensive testing across different network configurations
def run_comprehensive_testing(output_file="protocol_comparison_results.csv"):
    """Run comprehensive testing of routing protocols across different scenarios."""
    # Define network configurations to test
    network_configs = [
        {'num_nodes': 10, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Small Network'},
        {'num_nodes': 10, 'connectivity': 0.5, 'seed': 42, 'name': 'Dense Small Network'},
        {'num_nodes': 20, 'connectivity': 0.2, 'seed': 42, 'name': 'Sparse Medium Network'},
        {'num_nodes': 20, 'connectivity': 0.4, 'seed': 42, 'name': 'Dense Medium Network'},
        {'num_nodes': 30, 'connectivity': 0.1, 'seed': 42, 'name': 'Sparse Large Network'},
        {'num_nodes': 30, 'connectivity': 0.3, 'seed': 42, 'name': 'Dense Large Network'}
    ]
    
    # Define traffic patterns to test
    traffic_patterns = [
        {'pattern': 'random', 'intensity': 0.3, 'name': 'Light Random Traffic'},
        {'pattern': 'random', 'intensity': 0.7, 'name': 'Heavy Random Traffic'},
        {'pattern': 'hotspot', 'intensity': 0.5, 'name': 'Hotspot Traffic'},
        {'pattern': 'bottleneck', 'intensity': 0.6, 'name': 'Bottleneck Traffic'},
        {'pattern': 'cascade', 'intensity': 0.8, 'name': 'Cascade Failure'}
    ]
    
    # Initialize results storage
    all_summaries = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROUTING PROTOCOL COMPARISON")
    print("="*80)
    
    # Run comparison for each combination
    for net_config in network_configs:
        for traffic in traffic_patterns:
            config_name = f"{net_config['name']} with {traffic['name']}"
            print(f"\nTesting configuration: {config_name}")
            print("-"*80)
            
            # Run comparison
            _, summary = compare_routing_algorithms(
                network_params=net_config,
                test_iterations=5,  # Reduced iterations for faster total testing
                traffic_pattern=traffic['pattern'],
                traffic_intensity=traffic['intensity'],
                verbose=False
            )
            
            # Add configuration name
            summary['Configuration'] = config_name
            all_summaries.append(summary)
            
            # Print key results
            print(f"PAMR Quality: {summary['PAMR Avg Quality']:.4f}, RIP Quality: {summary['RIP Avg Quality']:.4f}, OSPF Quality: {summary['OSPF Avg Quality']:.4f}")
            print(f"PAMR vs RIP Quality: +{summary['PAMR vs RIP Quality %']:.2f}%, PAMR vs OSPF Quality: +{summary['PAMR vs OSPF Quality %']:.2f}%")
            print(f"PAMR Congestion: {summary['PAMR Avg Congestion']:.4f}, RIP Congestion: {summary['RIP Avg Congestion']:.4f}, OSPF Congestion: {summary['OSPF Avg Congestion']:.4f}")
            print(f"PAMR vs RIP Congestion Reduction: {summary['PAMR vs RIP Congestion %']:.2f}%, PAMR vs OSPF Congestion Reduction: {summary['PAMR vs OSPF Congestion %']:.2f}%")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    avg_pamr_vs_rip_quality = summary_df['PAMR vs RIP Quality %'].mean()
    avg_pamr_vs_ospf_quality = summary_df['PAMR vs OSPF Quality %'].mean()
    avg_pamr_vs_rip_congestion = summary_df['PAMR vs RIP Congestion %'].mean()
    avg_pamr_vs_ospf_congestion = summary_df['PAMR vs OSPF Congestion %'].mean()
    
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs RIP: +{avg_pamr_vs_rip_quality:.2f}%")
    print(f"AVERAGE PAMR QUALITY IMPROVEMENT vs OSPF: +{avg_pamr_vs_ospf_quality:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs RIP: {avg_pamr_vs_rip_congestion:.2f}%")
    print(f"AVERAGE PAMR CONGESTION REDUCTION vs OSPF: {avg_pamr_vs_ospf_congestion:.2f}%")
    
    # Find best and worst scenarios for PAMR
    best_vs_rip = summary_df.loc[summary_df['PAMR vs RIP Quality %'].idxmax()]
    best_vs_ospf = summary_df.loc[summary_df['PAMR vs OSPF Quality %'].idxmax()]
    
    print("\nBEST PAMR PERFORMANCE vs RIP:")
    print(f"Configuration: {best_vs_rip['Configuration']}")
    print(f"Quality Improvement: +{best_vs_rip['PAMR vs RIP Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_rip['PAMR vs RIP Congestion %']:.2f}%")
    
    print("\nBEST PAMR PERFORMANCE vs OSPF:")
    print(f"Configuration: {best_vs_ospf['Configuration']}")
    print(f"Quality Improvement: +{best_vs_ospf['PAMR vs OSPF Quality %']:.2f}%")
    print(f"Congestion Reduction: {best_vs_ospf['PAMR vs OSPF Congestion %']:.2f}%")
    
    print(f"\nDetailed results saved to {output_file}")
    
    return summary_df

if __name__ == "__main__":
    print("Starting adaptive routing demonstration to show PAMR's dynamic path selection")
    demonstrate_adaptive_routing(output_file="adaptive_routing_results.csv")
    
    print("\nRunning comprehensive comparison of PAMR, RIP, and OSPF across different network configurations")
    run_comprehensive_testing(output_file="protocol_comparison_results.csv")