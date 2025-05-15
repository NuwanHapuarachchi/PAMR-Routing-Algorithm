#!/usr/bin/env python3
"""
Direct implementation of PAMR routing protocol in Mininet
with MiniEdit GUI visualization support

This script:
1. Loads your actual PAMR routing implementation from routing.py
2. Implements it as a controller in Mininet
3. Provides a bridge to MiniEdit for visualization
4. Allows real-time monitoring of routing decisions
"""

import sys
import os
import subprocess
import time
import signal
import networkx as nx
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Add parent directory to path so we can import the pamr package
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import your protocol implementation
from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter

# Mininet components
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.topo import Topo

# For saving topology for MiniEdit
import json
import tempfile

class PAMRController:
    """Implementation of PAMR routing as a controller logic for Mininet"""
    
    def __init__(self, pamr_network, router=None):
        """
        Initialize the PAMR controller
        
        Args:
            pamr_network: NetworkTopology instance from your implementation
            router: PAMRRouter instance (if None, will create one)
        """
        self.network = pamr_network
        self.graph = pamr_network.graph
        
        # Create router if not provided
        if router is None:
            self.router = PAMRRouter(
                self.graph,
                alpha=2.0,  # Pheromone importance 
                beta=3.0,   # Distance importance
                gamma=8.0,  # Congestion importance
                adapt_weights=True
            )
        else:
            self.router = router
            
        # Custom traffic decay rate for our implementation
        self.traffic_decay = 0.9
        
        # Track active flows
        self.active_flows = {}
        self.flow_stats = {}
        
    def get_route(self, src, dst):
        """
        Get routing path using the PAMR algorithm
        
        Args:
            src: Source node
            dst: Destination node
            
        Returns:
            List of nodes representing the path
        """
        path, quality = self.router.find_path(src, dst)
        flow_id = f"{src}-{dst}"
        
        # Store stats for this flow
        if flow_id not in self.flow_stats:
            self.flow_stats[flow_id] = {
                'paths': [],
                'qualities': [],
                'congestion': []
            }
        
        # Calculate max congestion on this path
        max_congestion = 0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            max_congestion = max(max_congestion, self.graph[u][v].get('congestion', 0))
            
        # Update flow stats
        self.flow_stats[flow_id]['paths'].append(path)
        self.flow_stats[flow_id]['qualities'].append(quality)
        self.flow_stats[flow_id]['congestion'].append(max_congestion)
        
        # Update active flows
        self.active_flows[flow_id] = path
        
        # Update edge traffic and congestion
        self._update_path_traffic(path)
        
        # Custom decay traffic (don't use router's method)
        self._decay_traffic()
        
        # Update router iteration count
        self.router.iteration += 1
        
        return path
    
    def _update_path_traffic(self, path):
        """Update traffic stats for a path."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Increase traffic on this edge
            self.graph[u][v]['traffic'] = self.graph[u][v].get('traffic', 0) + 1.0
            
            # Update congestion based on traffic/capacity
            capacity = self.graph[u][v].get('capacity', 10)
            self.graph[u][v]['congestion'] = min(0.9, self.graph[u][v]['traffic'] / capacity)
    
    def _decay_traffic(self):
        """Custom implementation of traffic decay for all edges"""
        # For each edge in the graph
        for u, v in self.graph.edges():
            # Decay traffic on this edge
            self.graph[u][v]['traffic'] = self.graph[u][v].get('traffic', 0) * self.traffic_decay
            
            # Update congestion based on traffic/capacity ratio
            capacity = self.graph[u][v].get('capacity', 10)
            self.graph[u][v]['congestion'] = min(0.9, self.graph[u][v]['traffic'] / capacity)
            
    def visualize_routes(self, output_dir="mininet_results"):
        """
        Visualize all active routes in the network
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Draw the base network
        nx.draw_networkx_edges(
            self.graph, self.network.positions, 
            alpha=0.3, width=1.0, edge_color='gray'
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, self.network.positions,
            node_size=500,
            node_color='lightblue',
            edgecolors='black'
        )
        
        # Draw active routes with different colors
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        for i, (flow_id, path) in enumerate(self.active_flows.items()):
            src, dst = map(int, flow_id.split('-'))
            
            # Highlight source and destination
            nx.draw_networkx_nodes(
                self.graph, self.network.positions,
                nodelist=[src],
                node_size=600,
                node_color='green',
                edgecolors='black'
            )
            nx.draw_networkx_nodes(
                self.graph, self.network.positions,
                nodelist=[dst],
                node_size=600,
                node_color='red',
                edgecolors='black'
            )
            
            # Draw the path
            color = colors[i % len(colors)]
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                nx.draw_networkx_edges(
                    self.graph, self.network.positions,
                    edgelist=[(u, v)],
                    width=2.0,
                    edge_color=color,
                    arrows=True
                )
        
        # Add node labels
        nx.draw_networkx_labels(
            self.graph, self.network.positions,
            font_size=10,
            font_weight='bold'
        )
        
        # Set title and remove axes
        plt.title("PAMR Active Routes")
        plt.axis('off')
        
        # Save the figure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{output_dir}/active_routes_{timestamp}.png", bbox_inches='tight')
        plt.close()
        
        return f"{output_dir}/active_routes_{timestamp}.png"

class PAMRMininetBridge:
    """Bridge between your PAMR implementation and Mininet"""
    
    def __init__(self, pamr_network, controller=None):
        """
        Initialize the bridge
        
        Args:
            pamr_network: NetworkTopology instance
            controller: PAMRController instance (if None, will create one)
        """
        self.pamr_network = pamr_network
        
        # Create controller if not provided
        if controller is None:
            self.controller = PAMRController(pamr_network)
        else:
            self.controller = controller
            
        # Create Mininet topology
        self.topo = self._create_mininet_topo()
        self.net = None
        
    def _create_mininet_topo(self):
        """Create a Mininet topology from the PAMR network"""
        class PAMRTopo(Topo):
            def build(self, pamr_net):
                # Create switches (routers) for each node
                for node in pamr_net.graph.nodes():
                    self.addSwitch(f's{node}')
                    
                # Create a host for each node for traffic generation
                for node in pamr_net.graph.nodes():
                    host = self.addHost(f'h{node}', ip=f'10.0.0.{node+1}/24')
                    self.addLink(host, f's{node}')
                    
                # Create links between switches based on PAMR network
                for u, v in pamr_net.graph.edges():
                    # Get edge properties
                    distance = pamr_net.graph[u][v].get('distance', 1)
                    capacity = pamr_net.graph[u][v].get('capacity', 10)
                    
                    # Convert to Mininet link parameters
                    delay = f"{max(1, min(100, distance))}ms"
                    bw = max(1, min(1000, capacity))
                    
                    # Add the link
                    self.addLink(f's{u}', f's{v}', delay=delay, bw=bw)
        
        # Create topology with PAMR network
        topo = PAMRTopo(self.pamr_network)
        return topo
    
    def start_mininet(self):
        """Start Mininet with the PAMR topology"""
        self.net = Mininet(
            topo=self.topo,
            switch=OVSKernelSwitch,
            link=TCLink,
            controller=None  # No controller, we'll implement our own
        )
        
        # Start the network
        self.net.start()
        
        # Configure routing based on PAMR
        self._configure_routing()
        
        # Test connectivity
        print("\nTesting basic connectivity (this will fail if topology isn't connected):")
        self.net.pingAll()
        
        return self.net
    
    def _configure_routing(self):
        """Configure routing in Mininet using PAMR logic"""
        # For each switch, configure routing based on PAMR
        for src_node in self.pamr_network.graph.nodes():
            src_switch = self.net.get(f's{src_node}')
            
            # Configure routing to every destination
            for dst_node in self.pamr_network.graph.nodes():
                if src_node != dst_node:
                    # Get the next hop using PAMR
                    path = self.controller.get_route(src_node, dst_node)
                    
                    if len(path) > 1:
                        next_hop = path[1]  # The next node in the path
                        dst_ip = f'10.0.0.{dst_node+1}/24'
                        
                        # Set up forwarding in Mininet (this is simplified)
                        # In a real implementation, you'd use OpenFlow
                        # This is just for demonstration
                        src_switch.cmd(f'ip route add {dst_ip} via 10.0.0.{next_hop+1}')
    
    def run_cli(self):
        """Run Mininet CLI for manual exploration"""
        if self.net:
            CLI(self.net)
    
    def stop_mininet(self):
        """Stop Mininet"""
        if self.net:
            self.net.stop()
    
    def export_to_miniedit(self, output_file=None):
        """
        Export the topology to a format that MiniEdit can load
        
        Args:
            output_file: Path to save MiniEdit-compatible file (if None, create a temp file)
            
        Returns:
            Path to the exported file
        """
        if output_file is None:
            fd, output_file = tempfile.mkstemp(suffix='.mn')
            os.close(fd)
        
        # Create a MiniEdit-compatible topology description
        topo_data = {
            "application": "miniedit",
            "version": "2.2.0.1",
            "hosts": [],
            "switches": [],
            "links": []
        }
        
        # Add host information
        for node in self.pamr_network.graph.nodes():
            host = {
                "number": f"{node}",
                "opts": {
                    "hostname": f"h{node}",
                    "ip": f"10.0.0.{node+1}/24",
                    "nodeNum": node,
                    "sched": "host"
                },
                "x": str(self.pamr_network.positions[node][0] * 100 + 500),
                "y": str(self.pamr_network.positions[node][1] * 100 + 500)
            }
            topo_data["hosts"].append(host)
        
        # Add switch information
        for node in self.pamr_network.graph.nodes():
            switch = {
                "number": f"{node}",
                "opts": {
                    "hostname": f"s{node}",
                    "nodeNum": node,
                    "switchType": "legacySwitch"
                },
                "x": str(self.pamr_network.positions[node][0] * 100 + 600),
                "y": str(self.pamr_network.positions[node][1] * 100 + 500)
            }
            topo_data["switches"].append(switch)
        
        # Add link information
        # Host to switch links
        for node in self.pamr_network.graph.nodes():
            link = {
                "dest": f"s{node}",
                "opts": {},
                "src": f"h{node}"
            }
            topo_data["links"].append(link)
        
        # Switch to switch links
        for u, v in self.pamr_network.graph.edges():
            # Get edge properties
            distance = self.pamr_network.graph[u][v].get('distance', 1)
            capacity = self.pamr_network.graph[u][v].get('capacity', 10)
            
            # Convert to Mininet link parameters
            link = {
                "dest": f"s{v}",
                "opts": {
                    "bw": max(1, min(1000, capacity)),
                    "delay": f"{max(1, min(100, distance))}ms"
                },
                "src": f"s{u}"
            }
            topo_data["links"].append(link)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(topo_data, f, indent=4)
        
        return output_file
    
    def launch_miniedit(self, topo_file=None):
        """
        Launch MiniEdit with the PAMR topology
        
        Args:
            topo_file: Path to MiniEdit topology file (if None, create one)
            
        Returns:
            MiniEdit process
        """
        if topo_file is None:
            topo_file = self.export_to_miniedit()
        
        # Find MiniEdit path
        miniedit_path = "/usr/lib/python3/dist-packages/mininet/examples/miniedit.py"
        
        # Launch MiniEdit
        cmd = [sys.executable, miniedit_path, "--topo", topo_file]
        process = subprocess.Popen(cmd)
        
        print(f"Launched MiniEdit with topology from: {topo_file}")
        print("Close MiniEdit window when done exploring the topology.")
        
        return process

def run_pamr_mininet_demo(num_nodes=10, connectivity=0.3, seed=42):
    """
    Run a demonstration of PAMR routing in Mininet with MiniEdit visualization
    
    Args:
        num_nodes: Number of nodes in the network
        connectivity: Connectivity parameter for network generation
        seed: Random seed for reproducibility
    """
    # Create output directory
    os.makedirs("mininet_results", exist_ok=True)
    
    # Create PAMR network
    pamr_network = NetworkTopology(
        num_nodes=num_nodes,
        connectivity=connectivity,
        seed=seed
    )
    
    # Initialize dynamic metrics
    for _ in range(5):
        pamr_network.update_dynamic_metrics()
    
    # Create the bridge between PAMR and Mininet
    bridge = PAMRMininetBridge(pamr_network)
    
    # Export topology to MiniEdit format
    topo_file = bridge.export_to_miniedit("mininet_results/pamr_topology.mn")
    print(f"Exported topology to: {topo_file}")
    
    # Start Mininet
    try:
        print("Starting Mininet with PAMR topology...")
        bridge.start_mininet()
        
        # Launch MiniEdit in a separate process
        miniedit_process = bridge.launch_miniedit(topo_file)
        
        # Visualize initial routes
        route_image = bridge.controller.visualize_routes()
        print(f"Initial routes visualization saved to: {route_image}")
        
        # Run some traffic to demonstrate PAMR routing
        print("\nRunning traffic tests with PAMR routing...")
        hosts = [bridge.net.get(f'h{i}') for i in range(min(5, num_nodes))]
        
        # Run ping tests between some hosts
        for i in range(min(3, len(hosts))):
            src = hosts[i]
            dst = hosts[(i + 2) % len(hosts)]
            print(f"\nTesting connection from {src.name} to {dst.name}:")
            
            # Run ping and capture output
            output = src.cmd(f'ping -c 5 {dst.IP()}')
            print(output)
            
            # Update route visualization after traffic
            route_image = bridge.controller.visualize_routes()
            print(f"Updated routes visualization saved to: {route_image}")
        
        print("\nTest traffic completed. You can explore the network in MiniEdit.")
        print("Starting Mininet CLI. Type 'exit' when done.")
        
        # Run Mininet CLI
        bridge.run_cli()
        
        # Wait for MiniEdit to be closed
        miniedit_process.wait()
        
    finally:
        # Stop Mininet
        print("Stopping Mininet...")
        bridge.stop_mininet()

def main():
    """Parse command line arguments and run the demo"""
    parser = ArgumentParser(description='PAMR Routing in Mininet with MiniEdit GUI')
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes (default: 10)')
    parser.add_argument('--connectivity', type=float, default=0.3, help='Connectivity parameter (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--miniedit-only', action='store_true', help='Only launch MiniEdit without Mininet')
    
    args = parser.parse_args()
    
    # Set log level for Mininet
    setLogLevel('info')
    
    if args.miniedit_only:
        # Just create the network and export to MiniEdit
        pamr_network = NetworkTopology(
            num_nodes=args.nodes,
            connectivity=args.connectivity,
            seed=args.seed
        )
        bridge = PAMRMininetBridge(pamr_network)
        topo_file = bridge.export_to_miniedit("mininet_results/pamr_topology.mn")
        bridge.launch_miniedit(topo_file)
    else:
        # Run the full demo
        run_pamr_mininet_demo(
            num_nodes=args.nodes,
            connectivity=args.connectivity,
            seed=args.seed
        )

if __name__ == '__main__':
    main() 