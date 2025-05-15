#!/usr/bin/env python3
"""
PAMR vs Traditional Routing Protocols (RIP, OSPF) in Mininet
This script compares the PAMR routing protocol with traditional routing protocols
using Mininet to create a realistic network simulation environment.
"""

import sys
import os
import argparse
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path so we can import the pamr package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from pamr package
from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter
from pamr.simulation.mininet_sim import PAMRMininetSimulator, PAMRRIPExperiment
from mininet.log import setLogLevel

def run_comparison(topology_type='random', num_nodes=10, connectivity=0.3, 
                   num_iterations=100, packets_per_iter=10, output_dir=None):
    """Run a comparison between PAMR and traditional routing protocols.
    
    Args:
        topology_type: Type of topology to create ('random', 'ring', 'star', or 'mesh')
        num_nodes: Number of nodes in the network
        connectivity: Connectivity parameter for random topology
        num_iterations: Number of iterations to run the simulation
        packets_per_iter: Number of packets to send per iteration
        output_dir: Directory to save results (default: timestamp-based directory)
        
    Returns:
        Dictionary with comparison results
    """
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"mininet_comparison_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a network topology
    network = create_topology(topology_type, num_nodes, connectivity)
    
    # Generate some initial traffic in the network
    for _ in range(5):
        network.update_dynamic_metrics()
    
    # Create a mininet experiment for PAMR vs RIP
    experiment = PAMRRIPExperiment(network, output_dir=output_dir)
    
    # Select source-destination pairs for evaluation
    # Choose pairs with increasing hop distance
    source_dest_pairs = select_diverse_pairs(network.graph, 3)
    
    # Run the experiment
    results = experiment.run(source_dest_pairs=source_dest_pairs, num_packets=packets_per_iter)
    
    # Create summary of results
    create_summary_report(results, output_dir)
    
    return results

def create_topology(topology_type, num_nodes, connectivity):
    """Create a network topology of the specified type.
    
    Args:
        topology_type: Type of topology to create ('random', 'ring', 'star', or 'mesh')
        num_nodes: Number of nodes in the network
        connectivity: Connectivity parameter for random topology
        
    Returns:
        NetworkTopology instance
    """
    if topology_type == 'random':
        # Create a random topology with specified connectivity
        network = NetworkTopology(
            num_nodes=num_nodes,
            connectivity=connectivity,
            seed=42
        )
    elif topology_type == 'ring':
        # Create a ring topology
        G = nx.cycle_graph(num_nodes)
        G = nx.DiGraph(G)  # Convert to directed graph
        
        # Add random positions
        pos = {}
        for i in range(num_nodes):
            angle = 2 * np.pi * i / num_nodes
            pos[i] = (np.cos(angle), np.sin(angle))
        
        # Create network from the graph
        network = NetworkTopology(existing_graph=G, positions=pos)
    elif topology_type == 'star':
        # Create a star topology
        G = nx.star_graph(num_nodes - 1)
        G = nx.DiGraph(G)  # Convert to directed graph
        
        # Add random positions
        pos = {0: (0, 0)}  # Center node
        for i in range(1, num_nodes):
            angle = 2 * np.pi * (i-1) / (num_nodes-1)
            pos[i] = (np.cos(angle), np.sin(angle))
        
        # Create network from the graph
        network = NetworkTopology(existing_graph=G, positions=pos)
    elif topology_type == 'mesh':
        # Create a mesh (fully connected) topology
        G = nx.complete_graph(num_nodes)
        G = nx.DiGraph(G)  # Convert to directed graph
        
        # Add random positions
        pos = {}
        for i in range(num_nodes):
            angle = 2 * np.pi * i / num_nodes
            # Alternate radii for better visualization
            r = 0.8 if i % 2 == 0 else 1.0
            pos[i] = (r * np.cos(angle), r * np.sin(angle))
        
        # Create network from the graph
        network = NetworkTopology(existing_graph=G, positions=pos)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    # Initialize dynamic metrics for the network
    network.update_dynamic_metrics()
    
    return network

def select_diverse_pairs(graph, num_pairs):
    """Select a diverse set of source-destination pairs with varying hop distances.
    
    Args:
        graph: NetworkX graph
        num_pairs: Number of pairs to select
        
    Returns:
        List of (source, destination) tuples
    """
    pairs = []
    nodes = list(graph.nodes())
    
    # Calculate all-pairs shortest paths
    all_paths = dict(nx.all_pairs_shortest_path(graph))
    
    # Group paths by length
    paths_by_length = {}
    for src in all_paths:
        for dst, path in all_paths[src].items():
            if src != dst:
                length = len(path) - 1  # Number of hops
                if length not in paths_by_length:
                    paths_by_length[length] = []
                paths_by_length[length].append((src, dst))
    
    # Select pairs with increasing hop distances
    lengths = sorted(paths_by_length.keys())
    for i in range(min(num_pairs, len(lengths))):
        length = lengths[i % len(lengths)]
        pair = random.choice(paths_by_length[length])
        pairs.append(pair)
    
    # If we need more pairs, add random ones
    while len(pairs) < num_pairs:
        src = random.choice(nodes)
        dst = random.choice([n for n in nodes if n != src])
        if (src, dst) not in pairs:
            pairs.append((src, dst))
    
    return pairs

def create_summary_report(results, output_dir):
    """Create a summary report of comparison results.
    
    Args:
        results: Dictionary with experiment results
        output_dir: Directory to save the report
    """
    # Create a summary figure
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 5 * len(results)))
    
    # Ensure axes is a 2D array
    if len(results) == 1:
        axes = axes.reshape(1, 2)
    
    # Gather metrics for each source-destination pair
    for i, (pair_key, data) in enumerate(results.items()):
        src, dst = map(int, pair_key.split('-'))
        
        # Plot 1: RTT comparison
        ax1 = axes[i, 0]
        pamr_rtt = data['pamr']['rtt']
        rip_rtt = data['rip']['rtt']
        
        if pamr_rtt and rip_rtt:
            ax1.boxplot([pamr_rtt, rip_rtt], labels=['PAMR', 'RIP'])
            ax1.set_title(f'RTT Comparison: {src} to {dst}')
            ax1.set_ylabel('Round Trip Time (ms)')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add mean values as text
            pamr_mean = np.mean(pamr_rtt)
            rip_mean = np.mean(rip_rtt)
            ax1.text(1, max(pamr_rtt), f'Mean: {pamr_mean:.2f}ms', 
                    ha='center', va='bottom', fontweight='bold')
            ax1.text(2, max(rip_rtt), f'Mean: {rip_mean:.2f}ms', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Path length comparison
        ax2 = axes[i, 1]
        pamr_path_len = len(data['pamr']['path']) - 1 if data['pamr']['path'] else 0
        rip_path_len = len(data['rip']['path']) - 1 if data['rip']['path'] else 0
        
        ax2.bar([0, 1], [pamr_path_len, rip_path_len], color=['blue', 'red'])
        ax2.set_title(f'Path Length Comparison: {src} to {dst}')
        ax2.set_ylabel('Number of Hops')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['PAMR', 'RIP'])
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add hop count as text
        ax2.text(0, pamr_path_len, f'{pamr_path_len}', 
                ha='center', va='bottom', fontweight='bold')
        ax2.text(1, rip_path_len, f'{rip_path_len}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_results.png", bbox_inches='tight')
    plt.close()
    
    # Create a text summary
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write("PAMR vs Traditional Routing Protocols Comparison Summary\n")
        f.write("======================================================\n\n")
        
        for pair_key, data in results.items():
            src, dst = map(int, pair_key.split('-'))
            f.write(f"Path from {src} to {dst}:\n")
            f.write("-----------------------\n")
            
            # PAMR results
            pamr_rtt_mean = np.mean(data['pamr']['rtt']) if data['pamr']['rtt'] else 0
            pamr_path_len = len(data['pamr']['path']) - 1 if data['pamr']['path'] else 0
            pamr_loss = data['pamr']['loss']
            
            f.write(f"PAMR:\n")
            f.write(f"  - Path: {data['pamr']['path']}\n")
            f.write(f"  - Path Length: {pamr_path_len} hops\n")
            f.write(f"  - Average RTT: {pamr_rtt_mean:.2f} ms\n")
            f.write(f"  - Packet Loss: {pamr_loss:.1f}%\n\n")
            
            # RIP results
            rip_rtt_mean = np.mean(data['rip']['rtt']) if data['rip']['rtt'] else 0
            rip_path_len = len(data['rip']['path']) - 1 if data['rip']['path'] else 0
            rip_loss = data['rip']['loss']
            
            f.write(f"RIP:\n")
            f.write(f"  - Path: {data['rip']['path']}\n")
            f.write(f"  - Path Length: {rip_path_len} hops\n")
            f.write(f"  - Average RTT: {rip_rtt_mean:.2f} ms\n")
            f.write(f"  - Packet Loss: {rip_loss:.1f}%\n\n")
            
            # Comparison
            if pamr_rtt_mean > 0 and rip_rtt_mean > 0:
                rtt_improvement = (rip_rtt_mean - pamr_rtt_mean) / rip_rtt_mean * 100
                f.write(f"RTT Improvement: {rtt_improvement:.2f}%\n")
            
            if pamr_path_len > 0 and rip_path_len > 0:
                path_improvement = (rip_path_len - pamr_path_len) / rip_path_len * 100
                f.write(f"Path Length Improvement: {path_improvement:.2f}%\n")
            
            f.write("\n\n")

def main():
    """Parse command line arguments and run the comparison."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare PAMR with traditional routing protocols using Mininet')
    parser.add_argument('--topo', type=str, default='random', choices=['random', 'ring', 'star', 'mesh'],
                        help='Topology type (default: random)')
    parser.add_argument('--nodes', type=int, default=10,
                        help='Number of nodes in the network (default: 10)')
    parser.add_argument('--conn', type=float, default=0.3,
                        help='Connectivity parameter for random topology (default: 0.3)')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations (default: 100)')
    parser.add_argument('--packets', type=int, default=10,
                        help='Number of packets per iteration (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: timestamp-based)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set Mininet log level
    setLogLevel('info')
    
    # Run the comparison
    print(f"Starting comparison with topology: {args.topo}, nodes: {args.nodes}")
    results = run_comparison(
        topology_type=args.topo,
        num_nodes=args.nodes,
        connectivity=args.conn,
        num_iterations=args.iters,
        packets_per_iter=args.packets,
        output_dir=args.output
    )
    
    # Print completion message
    if args.output:
        output_dir = args.output
    else:
        # Find the directory that was just created (most recent)
        dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('mininet_comparison_')]
        output_dir = max(dirs, key=os.path.getctime) if dirs else "unknown"
    
    print(f"Comparison completed. Results saved to: {output_dir}")


if __name__ == '__main__':
    main() 