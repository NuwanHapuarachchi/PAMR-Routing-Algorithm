#!/usr/bin/env python3
"""
PAMR Protocol Implementation with Mininet
This module simulates the PAMR routing protocol in a realistic Mininet environment.
"""

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.topo import Topo
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import sys
import time
import random


class PAMRMininetSimulator:
    """Simulate PAMR routing protocol in a Mininet environment."""
    
    def __init__(self, network_topo=None, controller_ip='127.0.0.1', controller_port=6653):
        """Initialize the PAMR Mininet simulator.
        
        Args:
            network_topo: Optional custom Mininet topology
            controller_ip: IP address of the SDN controller
            controller_port: Port of the SDN controller
        """
        self.network_topo = network_topo
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        self.graph = nx.DiGraph()
        self.positions = {}
        
    def build_from_pamr_network(self, pamr_network):
        """Build a Mininet topology from a PAMR NetworkTopology instance.
        
        Args:
            pamr_network: An instance of pamr.core.network.NetworkTopology
        """
        # Create custom topology based on PAMR network
        class PAMRTopo(Topo):
            def build(self, pamr_net):
                # Create switches for each node in the PAMR network
                for node in pamr_net.graph.nodes():
                    self.addSwitch(f's{node}')
                
                # Create hosts for each node (for traffic generation)
                for node in pamr_net.graph.nodes():
                    host = self.addHost(f'h{node}', ip=f'10.0.0.{node+1}/24')
                    self.addLink(host, f's{node}')
                
                # Create links between switches based on PAMR network edges
                for u, v in pamr_net.graph.edges():
                    # Extract link properties from PAMR graph
                    delay = pamr_net.graph[u][v].get('distance', 1)
                    bw = pamr_net.graph[u][v].get('capacity', 10)
                    loss = pamr_net.graph[u][v].get('congestion', 0) * 100  # Convert to percentage
                    
                    # Ensure values are within reasonable ranges for Mininet
                    delay = min(max(delay, 1), 100)  # 1-100ms
                    bw = min(max(bw, 1), 1000)  # 1-1000 Mbps
                    loss = min(max(loss, 0), 50)  # 0-50% (avoid 100% loss)
                    
                    # Add bidirectional link with delay/bw/loss constraints
                    self.addLink(f's{u}', f's{v}', 
                                delay=f'{delay}ms',
                                bw=bw,
                                loss=loss,
                                max_queue_size=1000)
        
        # Store the PAMR network graph and positions
        self.graph = pamr_network.graph.copy()
        self.positions = pamr_network.positions.copy()
        
        # Create the Mininet topology
        self.network_topo = PAMRTopo(pamr_network)
    
    def start(self):
        """Start the Mininet network simulation."""
        if not self.network_topo:
            raise ValueError("Network topology not defined. Call build_from_pamr_network() first.")
        
        # Create Mininet network with the topology
        self.net = Mininet(
            topo=self.network_topo,
            switch=OVSKernelSwitch,
            controller=lambda name: RemoteController(name, ip=self.controller_ip, port=self.controller_port),
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        # Start the network
        self.net.start()
        info('*** Network started\n')
        
        # Show network information
        self.net.pingAll()  # Test connectivity
        
        # Return the network instance for direct interaction
        return self.net
    
    def stop(self):
        """Stop the Mininet network simulation."""
        if self.net:
            info('*** Stopping network\n')
            self.net.stop()
    
    def run_cli(self):
        """Run the Mininet CLI for interactive commands."""
        if self.net:
            info('*** Starting CLI\n')
            CLI(self.net)
    
    def run_experiment(self, source, destination, protocol='pamr', num_packets=10):
        """Run a routing experiment with specified parameters.
        
        Args:
            source: Source node
            destination: Destination node
            protocol: Routing protocol to use ('pamr', 'ospf', or 'rip')
            num_packets: Number of packets to send
            
        Returns:
            Dictionary with experiment results
        """
        if not self.net:
            raise ValueError("Network not started. Call start() first.")
        
        # Get host objects
        src_host = self.net.get(f'h{source}')
        dst_host = self.net.get(f'h{destination}')
        
        # Configure routing protocol
        if protocol == 'pamr':
            # Set up PAMR routing (in a real implementation, this would involve configuring the SDN controller)
            info(f'*** Setting up PAMR routing from {source} to {destination}\n')
            # In a real implementation, we'd configure the switches through the SDN controller
        elif protocol == 'ospf':
            # Set up OSPF routing
            info(f'*** Setting up OSPF routing from {source} to {destination}\n')
            # In a real implementation, we'd configure Quagga/FRR for OSPF
        elif protocol == 'rip':
            # Set up RIP routing
            info(f'*** Setting up RIP routing from {source} to {destination}\n')
            # In a real implementation, we'd configure Quagga/FRR for RIP
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
        
        # Run the experiment by sending packets
        info(f'*** Sending {num_packets} packets from {src_host.name} to {dst_host.name}\n')
        results = {
            'rtt': [],
            'loss': 0,
            'path': None
        }
        
        # Use ping to measure RTT
        ping_output = src_host.cmd(f'ping -c {num_packets} {dst_host.IP()}')
        info(ping_output)
        
        # Extract RTT from ping output
        rtt_lines = [line for line in ping_output.split('\n') if 'time=' in line]
        for line in rtt_lines:
            try:
                rtt = float(line.split('time=')[1].split(' ms')[0])
                results['rtt'].append(rtt)
            except (IndexError, ValueError):
                pass
        
        # Extract packet loss from ping output
        try:
            loss_line = [line for line in ping_output.split('\n') if 'packet loss' in line][0]
            loss_pct = float(loss_line.split('%')[0].split(' ')[-1])
            results['loss'] = loss_pct
        except (IndexError, ValueError):
            pass
        
        # In a real implementation, we'd extract the actual path from the SDN controller
        # For now, we'll use the NetworkX shortest path as a placeholder
        if nx.has_path(self.graph, source, destination):
            results['path'] = nx.shortest_path(self.graph, source, destination, weight='distance')
        
        return results
    
    def visualize_network(self, title="Mininet Network Topology", show_ips=True, 
                          highlight_path=None, save_path=None):
        """Visualize the network topology using NetworkX and Matplotlib.
        
        Args:
            title: Title for the visualization
            show_ips: Whether to show IP addresses
            highlight_path: Path to highlight (list of nodes)
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Draw the base network
        nx.draw_networkx_edges(
            self.graph, self.positions, 
            alpha=0.3, width=1.0, edge_color='gray'
        )
        
        # Draw hosts and switches
        switch_nodes = [f's{node}' for node in self.graph.nodes()]
        host_nodes = [f'h{node}' for node in self.graph.nodes()]
        
        # Position hosts slightly offset from their switches
        host_positions = {}
        for node in self.graph.nodes():
            x, y = self.positions[node]
            host_positions[f'h{node}'] = (x - 0.05, y - 0.05)
        
        # Draw switches
        nx.draw_networkx_nodes(
            self.graph, self.positions,
            nodelist=range(len(switch_nodes)),
            node_color='skyblue', 
            node_size=500,
            edgecolors='black'
        )
        
        # Draw hosts
        host_graph = nx.Graph()
        for node in self.graph.nodes():
            host_graph.add_node(f'h{node}')
        
        nx.draw_networkx_nodes(
            host_graph, host_positions,
            node_color='lightgreen', 
            node_size=300,
            edgecolors='black'
        )
        
        # Draw links between hosts and switches
        for node in self.graph.nodes():
            plt.plot(
                [self.positions[node][0], host_positions[f'h{node}'][0]],
                [self.positions[node][1], host_positions[f'h{node}'][1]],
                'k-', alpha=0.5, linewidth=1.0
            )
        
        # Highlight path if specified
        if highlight_path and len(highlight_path) > 1:
            path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
            nx.draw_networkx_edges(
                self.graph, self.positions,
                edgelist=path_edges,
                edge_color='red',
                width=3.0
            )
        
        # Add labels
        switch_labels = {node: f's{node}' for node in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph, self.positions,
            labels=switch_labels,
            font_size=10,
            font_weight='bold'
        )
        
        if show_ips:
            host_labels = {f'h{node}': f'h{node}\n10.0.0.{node+1}' for node in self.graph.nodes()}
            nx.draw_networkx_labels(
                host_graph, host_positions,
                labels=host_labels,
                font_size=8
            )
        
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            info(f'*** Saved visualization to {save_path}\n')
        
        return plt.gcf()


class PAMRRIPExperiment:
    """Compare PAMR and RIP in a Mininet environment."""
    
    def __init__(self, pamr_network, output_dir='mininet_results'):
        """Initialize the experiment.
        
        Args:
            pamr_network: PAMR network topology instance
            output_dir: Directory to save results
        """
        self.pamr_network = pamr_network
        self.output_dir = output_dir
        self.simulator = PAMRMininetSimulator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self, source_dest_pairs=None, num_packets=10):
        """Run the experiment comparing PAMR and RIP.
        
        Args:
            source_dest_pairs: List of (source, destination) tuples
            num_packets: Number of packets to send for each test
            
        Returns:
            Dictionary with experiment results
        """
        # Build Mininet topology from PAMR network
        self.simulator.build_from_pamr_network(self.pamr_network)
        
        # Start the Mininet simulation
        net = self.simulator.start()
        
        # If no source-destination pairs are provided, generate some random ones
        if not source_dest_pairs:
            nodes = list(self.pamr_network.graph.nodes())
            source_dest_pairs = []
            for _ in range(3):  # Generate 3 random pairs
                src = random.choice(nodes)
                dst = random.choice([n for n in nodes if n != src])
                source_dest_pairs.append((src, dst))
        
        # Run experiments for each protocol and source-destination pair
        results = {}
        for src, dst in source_dest_pairs:
            pair_key = f"{src}-{dst}"
            results[pair_key] = {}
            
            # Visualize the network for this pair
            self.simulator.visualize_network(
                title=f"Network Topology: Testing Path from {src} to {dst}",
                highlight_path=None,  # We don't know the path yet
                save_path=f"{self.output_dir}/topology_{src}_to_{dst}.png"
            )
            
            # Run PAMR experiment
            pamr_results = self.simulator.run_experiment(src, dst, protocol='pamr', num_packets=num_packets)
            results[pair_key]['pamr'] = pamr_results
            
            # Visualize PAMR path
            if pamr_results['path']:
                self.simulator.visualize_network(
                    title=f"PAMR Path from {src} to {dst}",
                    highlight_path=pamr_results['path'],
                    save_path=f"{self.output_dir}/pamr_path_{src}_to_{dst}.png"
                )
            
            # Run RIP experiment
            rip_results = self.simulator.run_experiment(src, dst, protocol='rip', num_packets=num_packets)
            results[pair_key]['rip'] = rip_results
            
            # Visualize RIP path
            if rip_results['path']:
                self.simulator.visualize_network(
                    title=f"RIP Path from {src} to {dst}",
                    highlight_path=rip_results['path'],
                    save_path=f"{self.output_dir}/rip_path_{src}_to_{dst}.png"
                )
            
            # Visualize comparison results
            self._visualize_comparison(src, dst, pamr_results, rip_results)
        
        # Stop the simulation
        self.simulator.stop()
        
        return results
    
    def _visualize_comparison(self, src, dst, pamr_results, rip_results):
        """Visualize the comparison results.
        
        Args:
            src: Source node
            dst: Destination node
            pamr_results: Results from PAMR experiment
            rip_results: Results from RIP experiment
        """
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: RTT comparison
        if pamr_results['rtt'] and rip_results['rtt']:
            ax1.boxplot([pamr_results['rtt'], rip_results['rtt']], labels=['PAMR', 'RIP'])
            ax1.set_title(f'RTT Comparison: {src} to {dst}')
            ax1.set_ylabel('Round Trip Time (ms)')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add mean values as text
            pamr_mean = np.mean(pamr_results['rtt'])
            rip_mean = np.mean(rip_results['rtt'])
            ax1.text(1, max(pamr_results['rtt']), f'Mean: {pamr_mean:.2f}ms', 
                    ha='center', va='bottom', fontweight='bold')
            ax1.text(2, max(rip_results['rtt']), f'Mean: {rip_mean:.2f}ms', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Path comparison
        if pamr_results['path'] and rip_results['path']:
            # Create a combined graph for visualization
            G = nx.DiGraph()
            
            # Add all nodes from both paths
            all_nodes = set(pamr_results['path']).union(set(rip_results['path']))
            for node in all_nodes:
                G.add_node(node)
            
            # Add edges from PAMR path
            for i in range(len(pamr_results['path']) - 1):
                u, v = pamr_results['path'][i], pamr_results['path'][i+1]
                G.add_edge(u, v, protocol='pamr')
            
            # Add edges from RIP path
            for i in range(len(rip_results['path']) - 1):
                u, v = rip_results['path'][i], rip_results['path'][i+1]
                if not G.has_edge(u, v):
                    G.add_edge(u, v, protocol='rip')
                elif G[u][v]['protocol'] == 'pamr':
                    G[u][v]['protocol'] = 'both'
            
            # Position nodes using the same layout as the network
            pos = {node: self.simulator.positions[node] for node in G.nodes()}
            
            # Draw the graph
            # Edges in both paths
            both_edges = [(u, v) for u, v in G.edges() if G[u][v]['protocol'] == 'both']
            nx.draw_networkx_edges(G, pos, edgelist=both_edges, ax=ax2,
                                 width=3.0, edge_color='purple', alpha=0.7)
            
            # PAMR-only edges
            pamr_edges = [(u, v) for u, v in G.edges() if G[u][v]['protocol'] == 'pamr']
            nx.draw_networkx_edges(G, pos, edgelist=pamr_edges, ax=ax2,
                                 width=2.0, edge_color='blue', style='solid')
            
            # RIP-only edges
            rip_edges = [(u, v) for u, v in G.edges() if G[u][v]['protocol'] == 'rip']
            nx.draw_networkx_edges(G, pos, edgelist=rip_edges, ax=ax2,
                                 width=2.0, edge_color='red', style='dashed')
            
            # Nodes
            nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=500, node_color='lightblue')
            
            # Highlight source and destination
            nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=[src, dst], 
                                 node_size=700, node_color=['green', 'red'])
            
            # Labels
            nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10)
            
            # Legend items
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', lw=2, label=f'PAMR Only ({len(pamr_results["path"])-1} hops)'),
                Line2D([0], [0], color='red', lw=2, linestyle='dashed', label=f'RIP Only ({len(rip_results["path"])-1} hops)'),
                Line2D([0], [0], color='purple', lw=3, label='Common Path Segments')
            ]
            ax2.legend(handles=legend_elements, loc='upper right')
            
            ax2.set_title(f'Path Comparison: {src} to {dst}')
            ax2.axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comparison_{src}_to_{dst}.png", bbox_inches='tight')
        plt.close()


def run_mininet_demo():
    """Run a simple demonstration of Mininet simulation with PAMR."""
    from pamr.core.network import NetworkTopology
    
    # Set log level for Mininet
    setLogLevel('info')
    
    # Create a sample PAMR network
    pamr_network = NetworkTopology(
        num_nodes=6,  # Small network for demo
        connectivity=0.5,
        seed=42
    )
    
    # Generate traffic in the network
    for _ in range(5):
        pamr_network.update_dynamic_metrics()
    
    # Create a Mininet simulator
    simulator = PAMRMininetSimulator()
    
    # Build Mininet topology from PAMR network
    simulator.build_from_pamr_network(pamr_network)
    
    # Visualize the network topology before starting
    simulator.visualize_network(
        title="PAMR Network in Mininet",
        save_path="mininet_results/mininet_topology.png"
    )
    
    # Start the simulation
    try:
        net = simulator.start()
        
        # Run a simple experiment
        result = simulator.run_experiment(0, 3, protocol='pamr', num_packets=5)
        
        # Visualize the result
        if result['path']:
            simulator.visualize_network(
                title=f"PAMR Path from 0 to 3",
                highlight_path=result['path'],
                save_path="mininet_results/example_path.png"
            )
        
        # Allow user to interact with the network
        simulator.run_cli()
        
    finally:
        # Stop the simulation
        simulator.stop()


if __name__ == '__main__':
    # Create results directory
    os.makedirs("mininet_results", exist_ok=True)
    
    # Run the demo
    run_mininet_demo() 