"""
PAMR Enterprise IP Routing Simulation
"""

import sys
import os
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import pickle
import datetime
from collections import defaultdict
import seaborn as sns
import threading
import ipaddress
import webbrowser
import inspect
import scipy.stats as stats
from matplotlib.ticker import PercentFormatter

# Add parent directory to path for importing the PAMR package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PAMR components
from pamr.core.routing import PAMRRouter, AdvancedMultiPathRouter
from pamr.simulation.simulator import PAMRSimulator
from pamr.visualization.network_viz import NetworkVisualizer

# Import comparison protocols implementation
try:
    from examples.comparison_with_ospf import OSPFRouter, OSPFSimulator
    from examples.comparison_with_rip import RIPRouter, RIPSimulator
except ImportError:
    print("Warning: Could not import comparison protocol classes. Continuing with PAMR only.")

class EnterpriseNetwork:
    """Enterprise network topology with IP address assignments and realistic traffic patterns."""
    
    def __init__(self, num_nodes=30, connectivity=0.05, seed=42, variation_factor=0.2):
        """Initialize the enterprise network.
        
        Args:
            num_nodes: Number of nodes in the network
            connectivity: Probability of edge creation between nodes
            seed: Random seed for reproducibility
            variation_factor: Factor for network metric variations
        """
        # Initialize network parameters
        self.num_nodes = num_nodes
        self.connectivity = connectivity
        self.seed = seed
        self.variation_factor = variation_factor
        self.iteration = 0
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize data structures for network properties
        self.departments = {}
        self.ip_addresses = {}
        self.subnets = {}  # Dictionary to store subnet information
        self.bgp_table = {}  # Dictionary to store BGP routing table
        self.traffic_history = defaultdict(list)
        self.congestion_history = defaultdict(list)
        
        # Create network topology
        self._create_network()
        
        # Assign departments to nodes
        self._assign_departments()
        
        # Assign IP addresses based on departments
        self._assign_ip_addresses()
        
        # Initialize edge attributes (distance, bandwidth, packet loss, etc.)
        self._initialize_edge_attributes()
        
        # Create BGP table for interdepartmental routing
        self._create_bgp_table()
        
        # Traffic class definitions with QoS parameters
        self.traffic_classes = {
            'voip': {
                'priority': 1,  # Highest priority (lower is higher)
                'latency_sensitivity': 0.9,  # Very sensitive to latency
                'bandwidth_requirement': 0.2,  # Low bandwidth requirement
                'packet_size': 0.1,  # Small packets
                'burst_factor': 0.3,  # Moderate bursts
                'jitter_sensitivity': 0.95,  # Very sensitive to jitter
                'dscp_value': 46,  # Expedited Forwarding
                'color': 'red'
            },
            'video': {
                'priority': 2,
                'latency_sensitivity': 0.7,
                'bandwidth_requirement': 0.8,  # High bandwidth
                'packet_size': 0.8,  # Large packets
                'burst_factor': 0.7,  # High bursts
                'jitter_sensitivity': 0.8,  # Sensitive to jitter
                'dscp_value': 34,  # Assured Forwarding 41
                'color': 'blue'
            },
            'data': {
                'priority': 3,
                'latency_sensitivity': 0.3,
                'bandwidth_requirement': 0.5,
                'packet_size': 0.6,
                'burst_factor': 0.4,
                'jitter_sensitivity': 0.2,
                'dscp_value': 0,  # Best Effort
                'color': 'green'
            },
            'backup': {
                'priority': 4,  # Lowest priority
                'latency_sensitivity': 0.1,
                'bandwidth_requirement': 0.9,  # Very high bandwidth
                'packet_size': 1.0,  # Largest packets
                'burst_factor': 0.1,  # Low bursts (constant flow)
                'jitter_sensitivity': 0.1,  # Not sensitive to jitter
                'dscp_value': 8,  # CS1 (Low Priority)
                'color': 'gray'
            }
        }
        
        # Packet ID counter for tracking individual packets
        self.packet_id_counter = 0
        
        # Active packets currently in the network
        self.active_packets = {}
        
        # History of all routed packets
        self.packet_history = []
        
        # Traffic patterns by time of day
        self.time_patterns = {
            'business_hours': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'lunch_hour': [12, 13],
            'after_hours': [18, 19, 20, 21, 22],
            'night': [23, 0, 1, 2, 3, 4, 5, 6, 7]
        }

    def _create_network(self):
        """Create a hierarchical enterprise network topology."""
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Use Barabasi-Albert model for scale-free network (more realistic for enterprise networks)
        # This creates networks with hub-and-spoke characteristics common in enterprises
        G = nx.barabasi_albert_graph(self.num_nodes, 3, seed=self.seed)
        
        # Convert to directed graph
        G = nx.DiGraph(G)
        
        # Ensure graph is strongly connected
        if not nx.is_strongly_connected(G):
            self._ensure_connectivity(G)
        
        self.graph = G
        
        # Generate positions for visualization
        self.positions = nx.spring_layout(G, seed=self.seed)
        
    def _ensure_connectivity(self, G):
        """Ensure the graph is connected by adding edges between components."""
        # Find strongly connected components
        components = list(nx.strongly_connected_components(G))
        
        if len(components) > 1:
            # Connect all components to the largest component
            largest_component = max(components, key=len)
            other_components = [c for c in components if c != largest_component]
            
            for component in other_components:
                # Pick random nodes from each component
                from_node = random.choice(list(component))
                to_node = random.choice(list(largest_component))
                
                # Add edges in both directions
                G.add_edge(from_node, to_node)
                G.add_edge(to_node, from_node)
    
    def _assign_departments(self):
        """Assign nodes to different departments (subnets) based on network topology."""
        # Create departments based on graph structure
        departments = {
            'Core': [],
            'IT': [],
            'Finance': [],
            'HR': [],
            'Marketing': [],
            'Engineering': [],
            'Operations': [],
            'Sales': [],
            'Executive': []
        }
        
        # Identify highly connected nodes as core infrastructure
        degree_centrality = nx.degree_centrality(self.graph)
        sorted_by_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Top 5% of nodes by centrality are core infrastructure
        core_count = max(3, int(0.05 * self.num_nodes))
        for node, _ in sorted_by_centrality[:core_count]:
            departments['Core'].append(node)
        
        # Assign remaining nodes to departments based on graph structure (using community detection)
        remaining_nodes = [node for node in self.graph.nodes() if node not in departments['Core']]
        remaining_graph = self.graph.subgraph(remaining_nodes)
        
        # Use community detection to find natural groupings in the network
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(remaining_graph.to_undirected())
            
            # Map each community to a department
            available_departments = list(departments.keys())[1:]  # Skip 'Core'
            for i, comm in enumerate(communities):
                dept = available_departments[i % len(available_departments)]
                departments[dept].extend(list(comm))
        except:
            # Fallback to random assignment if community detection fails
            for node in remaining_nodes:
                dept = random.choice(list(departments.keys())[1:])  # Skip 'Core'
                departments[dept].append(node)
        
        self.departments = departments
    
    def _assign_ip_addresses(self):
        """Assign IP addresses to nodes based on their departments."""
        # Create subnets for each department
        base_network = ipaddress.IPv4Network('10.0.0.0/8')
        subnets = list(base_network.subnets(prefixlen_diff=8))  # Creates /16 subnets
        
        # Reserve the last subnet for nodes without departments
        unassigned_subnet = subnets[-1]
        subnets = subnets[:-1]
        
        # Assign subnets to departments
        subnet_mapping = {}
        for i, dept in enumerate(self.departments.keys()):
            subnet_idx = i % len(subnets)
            subnet_mapping[dept] = subnets[subnet_idx]
            self.subnets[dept] = str(subnets[subnet_idx])
        
        # Track all nodes that have been assigned IP addresses
        assigned_nodes = set()
        
        # Assign IP addresses to nodes based on their department
        for dept, nodes in self.departments.items():
            subnet = subnet_mapping[dept]
            hosts = list(subnet.hosts())
            
            for i, node in enumerate(nodes):
                host_idx = i % len(hosts)
                self.ip_addresses[node] = str(hosts[host_idx])
                assigned_nodes.add(node)
        
        # Check for any nodes without IP addresses and assign them from the unassigned subnet
        unassigned_hosts = list(unassigned_subnet.hosts())
        unassigned_idx = 0
        
        for node in self.graph.nodes():
            if node not in assigned_nodes:
                # Assign from the unassigned subnet
                self.ip_addresses[node] = str(unassigned_hosts[unassigned_idx])
                unassigned_idx = (unassigned_idx + 1) % len(unassigned_hosts)
                print(f"Warning: Node {node} was not assigned to any department. Assigned IP {self.ip_addresses[node]}")
        
        # Verify all nodes have IP addresses
        for node in self.graph.nodes():
            if node not in self.ip_addresses:
                raise ValueError(f"Node {node} still does not have an IP address after assignment")
    
    def _initialize_edge_attributes(self):
        """Initialize edge attributes with enterprise-specific characteristics."""
        for u, v in self.graph.edges():
            # Set consistent initial values based on node indices
            edge_seed = u * self.num_nodes + v
            rng = random.Random(edge_seed + self.seed)
            
            # Enterprise network characteristics:
            # - Core links have higher capacity
            # - Links within same department have lower latency
            # - Cross-department links have higher latency
            
            # Basic attributes
            is_core_link = u in self.departments['Core'] or v in self.departments['Core']
            same_department = False
            for dept, nodes in self.departments.items():
                if u in nodes and v in nodes:
                    same_department = True
                    break
            
            # Set distance (latency) based on link type
            if is_core_link:
                distance = rng.uniform(1.0, 3.0)  # Core links have low latency
            elif same_department:
                distance = rng.uniform(1.0, 5.0)  # Intra-department links have moderate latency
            else:
                distance = rng.uniform(5.0, 15.0)  # Inter-department links have higher latency
            
            # Set capacity (bandwidth) based on link type
            if is_core_link:
                capacity = rng.uniform(80.0, 100.0)  # Core links have high capacity
            elif same_department:
                capacity = rng.uniform(30.0, 70.0)   # Intra-department links have moderate capacity
            else:
                capacity = rng.uniform(10.0, 30.0)   # Inter-department links have lower capacity
            
            # Initialize edge attributes
            self.graph[u][v]['distance'] = distance
            self.graph[u][v]['pheromone'] = 0.5  # Initial pheromone level
            self.graph[u][v]['capacity'] = capacity
            self.graph[u][v]['traffic'] = 0.0
            self.graph[u][v]['congestion'] = 0.0
            
            # Enterprise-specific attributes
            self.graph[u][v]['link_type'] = 'core' if is_core_link else ('internal' if same_department else 'external')
            self.graph[u][v]['qos_enabled'] = is_core_link or rng.random() < 0.3  # QoS enabled on core and some other links
            self.graph[u][v]['packet_loss'] = rng.uniform(0.0, 0.01)  # Initial packet loss rate (0-1%)
            
            # Store original values for controlled variations
            self.graph[u][v]['base_distance'] = distance
            self.graph[u][v]['base_capacity'] = capacity
    
    def _create_bgp_table(self):
        """Create a simplified BGP routing table for interdepartmental routing."""
        # In an enterprise network, BGP might be used for large networks or for connections to ISPs
        # This is a simplified representation for simulation purposes
        
        # Create AS numbers for each department
        as_numbers = {dept: 65000 + i for i, dept in enumerate(self.departments.keys())}
        
        # Initialize BGP table
        for dept_source, nodes_source in self.departments.items():
            source_as = as_numbers[dept_source]
            
            for dept_dest, nodes_dest in self.departments.items():
                if dept_source != dept_dest:
                    # Find border nodes (nodes with connections to other departments)
                    border_nodes_source = []
                    for node in nodes_source:
                        neighbors = list(self.graph.neighbors(node))
                        for neighbor in neighbors:
                            neighbor_dept = self._get_node_department(neighbor)
                            if neighbor_dept != dept_source:
                                border_nodes_source.append(node)
                                break
                    
                    # Use the most connected border node as the exit point
                    if border_nodes_source:
                        exit_node = max(border_nodes_source, key=lambda x: self.graph.degree(x))
                        
                        # Add BGP entry
                        dest_subnet = self.subnets[dept_dest]
                        self.bgp_table[(dept_source, dest_subnet)] = {
                            'next_hop': exit_node,
                            'as_path': [source_as, as_numbers[dept_dest]],
                            'med': 100,  # Multi-Exit Discriminator
                            'local_pref': 100  # Local preference
                        }
    
    def _get_node_department(self, node):
        """Get the department for a given node."""
        for dept, nodes in self.departments.items():
            if node in nodes:
                return dept
        return None
    
    def update_dynamic_metrics(self, traffic_decay=0.9):
        """Update network metrics with small variations to simulate real-world conditions."""
        self.iteration += 1
        
        # Apply traffic decay to simulate packets leaving the network
        for u, v in self.graph.edges():
            # Apply traffic decay
            self.graph[u][v]['traffic'] *= traffic_decay
            
            # Store traffic history
            self.traffic_history[(u, v)].append(self.graph[u][v]['traffic'])
            if len(self.traffic_history[(u, v)]) > 100:  # Keep last 100 iterations
                self.traffic_history[(u, v)].pop(0)
            
            # Update congestion based on traffic/capacity ratio
            capacity = self.graph[u][v]['capacity']
            self.graph[u][v]['congestion'] = self.graph[u][v]['traffic'] / capacity
            
            # Store congestion history
            self.congestion_history[(u, v)].append(self.graph[u][v]['congestion'])
            if len(self.congestion_history[(u, v)]) > 100:  # Keep last 100 iterations
                self.congestion_history[(u, v)].pop(0)
            
            # Add small random variations to simulate real-world changes
            # Distance (latency) variations due to network conditions
            base_distance = self.graph[u][v]['base_distance']
            variation = self.variation_factor * base_distance * (random.random() - 0.5)
            self.graph[u][v]['distance'] = max(0.1, base_distance + variation)
            
            # Update packet loss based on congestion with modified formula
            # Lower base packet loss and more gradual increase with congestion
            congestion = self.graph[u][v]['congestion']
            
            # Check if this is a high-quality link (core or internal departmental link)
            is_core_link = u in self.departments.get('Core', []) or v in self.departments.get('Core', [])
            same_department = False
            for dept, nodes in self.departments.items():
                if u in nodes and v in nodes:
                    same_department = True
                    break
            
            # Calculate base loss probability based on link type
            if is_core_link:
                # Core links have much lower base packet loss
                base_loss = 0.002  # 0.2% base loss
            elif same_department:
                # Departmental links have low-medium base packet loss
                base_loss = 0.005  # 0.5% base loss
            else:
                # Inter-departmental links have higher base packet loss
                base_loss = 0.008  # 0.8% base loss
                
            # Calculate congestion-based additional loss
            # More gradual increase and lower maximum
            if congestion < 0.5:
                # Below 50% congestion, minimal impact
                congestion_loss = congestion * 0.05
            elif congestion < 0.8:
                # Between 50-80% congestion, moderate impact
                congestion_loss = 0.025 + (congestion - 0.5) * 0.1
            else:
                # Above 80% congestion, significant impact
                congestion_loss = 0.055 + (congestion - 0.8) * 0.25
                
            # Calculate final packet loss (max 8% for most links)
            self.graph[u][v]['packet_loss'] = min(0.08, base_loss + congestion_loss)
    
    def generate_background_traffic(self):
        """Generate realistic background traffic in the network."""
        # Different traffic patterns based on time of day/week
        # This simulates regular enterprise traffic patterns
        
        # Get current time for time-of-day based traffic
        current_time = datetime.datetime.now()
        hour = current_time.hour
        weekday = current_time.weekday()  # 0-6 (Monday-Sunday)
        
        # Business hours factor (higher during business hours)
        business_hours = 1.0  # Default
        if 0 <= weekday <= 4:  # Weekday
            if 9 <= hour <= 17:  # Business hours
                business_hours = 2.0  # Double traffic during business hours
            elif 7 <= hour < 9 or 17 < hour <= 19:  # Commute hours
                business_hours = 1.5  # 50% more traffic during commute hours
            else:  # Night hours
                business_hours = 0.5  # Half traffic during night hours
        else:  # Weekend
            business_hours = 0.3  # 30% traffic during weekends
        
        # Generate background traffic for different traffic classes
        self._generate_traffic_class('voip', intensity=0.3 * business_hours, packets=5)
        self._generate_traffic_class('video', intensity=0.5 * business_hours, packets=8)
        self._generate_traffic_class('data', intensity=0.7 * business_hours, packets=15)
        self._generate_traffic_class('backup', intensity=0.2, packets=5)  # Backup traffic is more constant
    
    def _generate_traffic_class(self, traffic_class, intensity=1.0, packets=10):
        """Generate traffic for a specific traffic class."""
        # Different traffic classes have different patterns
        if traffic_class == 'voip':
            # VoIP traffic: primarily between random pairs of nodes
            for _ in range(int(packets * intensity)):
                source, dest = random.sample(list(self.graph.nodes()), 2)
                self._add_traffic_on_shortest_path(source, dest, amount=0.5, qos_required=True)
                
        elif traffic_class == 'video':
            # Video traffic: often from core/servers to end nodes
            for _ in range(int(packets * intensity)):
                if self.departments['Core'] and random.random() < 0.7:
                    # 70% of video traffic from core/servers
                    source = random.choice(self.departments['Core'])
                    # Only choose from departments that have nodes
                    valid_dest_depts = [d for d in self.departments.keys() 
                                     if d != 'Core' and self.departments[d]]
                    if valid_dest_depts:
                        dest_dept = random.choice(valid_dest_depts)
                        destination = random.choice(self.departments[dest_dept])
                    else:
                        # Fallback if no valid departments found
                        source, destination = random.sample(list(self.graph.nodes()), 2)
                else:
                    # 30% random video traffic
                    source, destination = random.sample(list(self.graph.nodes()), 2)
                
        elif traffic_class == 'data':
            # Data traffic: distributed across the network
            for _ in range(int(packets * intensity)):
                source, dest = random.sample(list(self.graph.nodes()), 2)
                self._add_traffic_on_shortest_path(source, dest, amount=1.0, qos_required=False)
                
        elif traffic_class == 'backup':
            # Backup traffic: often to core infrastructure
            for _ in range(int(packets * intensity)):
                if self.departments['Core'] and random.random() < 0.8:
                    # 80% of backup traffic to core/servers
                    destination = random.choice(self.departments['Core'])
                    # Only choose from departments that have nodes
                    valid_source_depts = [d for d in self.departments.keys() 
                                       if d != 'Core' and self.departments[d]]
                    if valid_source_depts:
                        source_dept = random.choice(valid_source_depts)
                        source = random.choice(self.departments[source_dept])
                    else:
                        # Fallback if no valid departments found
                        source, destination = random.sample(list(self.graph.nodes()), 2)
                else:
                    # 20% random backup traffic
                    source, destination = random.sample(list(self.graph.nodes()), 2)
                self._add_traffic_on_shortest_path(source, destination, amount=3.0, qos_required=False)
    
    def _add_traffic_on_shortest_path(self, source, dest, amount=1.0, qos_required=False):
        """Add traffic along a path between source and destination."""
        try:
            # Use distance as weight for shortest path calculation
            path = nx.shortest_path(self.graph, source, dest, weight='distance')
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Add traffic to this edge
                self.graph[u][v]['traffic'] += amount
                
                # Update congestion based on new traffic
                capacity = self.graph[u][v]['capacity']
                self.graph[u][v]['congestion'] = self.graph[u][v]['traffic'] / capacity
        except nx.NetworkXNoPath:
            # No path exists
            pass

class EnterpriseRoutingSimulator:
    """Simulator for enterprise routing with background traffic."""
    
    def __init__(self, network, router):
        """Initialize the simulator.
        
        Args:
            network: The network topology
            router: The routing algorithm to use
        """
        self.network = network
        self.router = router
        self.metrics = {
            'path_lengths': [],
            'convergence_times': [],
            'path_qualities': [],
            'congestion_levels': [],
            'packet_loss_rates': [],
            'traffic_classes': {
                'voip': {'latency': [], 'loss': [], 'jitter': []},
                'video': {'latency': [], 'loss': [], 'jitter': []},
                'data': {'latency': [], 'loss': [], 'jitter': []},
                'backup': {'latency': [], 'loss': [], 'jitter': []}
            }
        }
    
    def run_simulation(self, num_iterations=100, packets_per_iter=50):
        """Run the routing simulation with background traffic.
        
        Args:
            num_iterations: Number of simulation iterations
            packets_per_iter: Number of packets to route per iteration
            
        Returns:
            List of path history for analysis
        """
        path_history = []
        
        for i in range(num_iterations):
            # Generate background traffic
            self.network.generate_background_traffic()
            
            # Update network metrics to simulate dynamic conditions
            self.network.update_dynamic_metrics()
            
            # Route packets for different traffic classes
            iteration_paths = []
            start_time = time.time()
            
            # Initialize metrics for this iteration
            total_path_length = 0
            total_path_quality = 0
            max_congestion = 0
            total_packet_loss = 0
            
            # Simulate different traffic classes with different requirements
            traffic_classes = [
                {'name': 'voip', 'weight': 0.2, 'qos_required': True},
                {'name': 'video', 'weight': 0.3, 'qos_required': True},
                {'name': 'data', 'weight': 0.4, 'qos_required': False},
                {'name': 'backup', 'weight': 0.1, 'qos_required': False}
            ]
            
            # Route packets for each traffic class
            for traffic_class in traffic_classes:
                # Calculate number of packets for this class
                class_packets = int(packets_per_iter * traffic_class['weight'])
                class_metrics = {
                    'latencies': [],
                    'losses': [],
                    'jitters': []
                }
                
                for _ in range(class_packets):
                    # Pick random source and destination based on traffic class
                    if traffic_class['name'] == 'voip':
                        # VoIP traffic: random pairs
                        source, destination = random.sample(list(self.network.graph.nodes()), 2)
                    elif traffic_class['name'] == 'video':
                        # Video traffic: often from core to endpoints
                        if self.network.departments['Core'] and random.random() < 0.7:
                            source = random.choice(self.network.departments['Core'])
                            # Only choose from departments that have nodes
                            valid_dest_depts = [d for d in self.network.departments.keys() 
                                             if d != 'Core' and self.network.departments[d]]
                            if valid_dest_depts:
                                dest_dept = random.choice(valid_dest_depts)
                                destination = random.choice(self.network.departments[dest_dept])
                            else:
                                # Fallback if no valid departments found
                                source, destination = random.sample(list(self.network.graph.nodes()), 2)
                        else:
                            # 30% random video traffic
                            source, destination = random.sample(list(self.network.graph.nodes()), 2)
                    
                    # Data traffic: distributed across the network
                    elif traffic_class['name'] == 'data':
                        source, destination = random.sample(list(self.network.graph.nodes()), 2)
                    
                    # Backup traffic: often to core infrastructure
                    elif traffic_class['name'] == 'backup':
                        if self.network.departments['Core'] and random.random() < 0.8:
                            # 80% of backup traffic to core/servers
                            destination = random.choice(self.network.departments['Core'])
                            # Only choose from departments that have nodes
                            valid_source_depts = [d for d in self.network.departments.keys() 
                                               if d != 'Core' and self.network.departments[d]]
                            if valid_source_depts:
                                source_dept = random.choice(valid_source_depts)
                                source = random.choice(self.network.departments[source_dept])
                            else:
                                # Fallback if no valid departments found
                                source, destination = random.sample(list(self.network.graph.nodes()), 2)
                        else:
                            # 20% random backup traffic
                            source, destination = random.sample(list(self.network.graph.nodes()), 2)
                    
                    # Find path using the router with traffic class and QoS information
                    if hasattr(self.router, 'find_path') and len(inspect.signature(self.router.find_path).parameters) >= 4:
                        # Router supports traffic class and QoS parameters
                        path, quality = self.router.find_path(
                            source, 
                            destination, 
                            traffic_class=traffic_class['name'],
                            qos_required=traffic_class.get('qos_required', False)
                        )
                    else:
                        # Basic router interface
                        path, quality = self.router.find_path(source, destination)
                    
                    if path and len(path) > 1:
                        # Calculate metrics for this path
                        path_latency = 0
                        path_loss = 0
                        path_jitter = 0
                        prev_latency = None
                        
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            
                            # Track latency (distance)
                            edge_latency = self.network.graph[u][v]['distance']
                            path_latency += edge_latency
                            
                            # Track packet loss
                            edge_loss = self.network.graph[u][v]['packet_loss']
                            path_loss = path_loss + edge_loss - (path_loss * edge_loss)  # Compound probability
                            
                            # Calculate jitter (variation in latency)
                            if prev_latency is not None:
                                path_jitter += abs(edge_latency - prev_latency)
                            prev_latency = edge_latency
                            
                            # Update traffic and congestion
                            self.network.graph[u][v]['traffic'] += 1.0
                            capacity = self.network.graph[u][v]['capacity']
                            self.network.graph[u][v]['congestion'] = self.network.graph[u][v]['traffic'] / capacity
                            max_congestion = max(max_congestion, self.network.graph[u][v]['congestion'])
                        
                        # Store metrics for this traffic class
                        class_metrics['latencies'].append(path_latency)
                        class_metrics['losses'].append(path_loss)
                        class_metrics['jitters'].append(path_jitter)
                        
                        # Store path information
                        iteration_paths.append((source, destination, path, quality, traffic_class['name']))
                        
                        # Update overall metrics
                        total_path_length += len(path) - 1
                        total_path_quality += quality
                        total_packet_loss += path_loss
                
                # Calculate average metrics for this traffic class
                if class_metrics['latencies']:
                    avg_latency = sum(class_metrics['latencies']) / len(class_metrics['latencies'])
                    avg_loss = sum(class_metrics['losses']) / len(class_metrics['losses'])
                    avg_jitter = sum(class_metrics['jitters']) / len(class_metrics['jitters'])
                    
                    # Store in metrics
                    self.metrics['traffic_classes'][traffic_class['name']]['latency'].append(avg_latency)
                    self.metrics['traffic_classes'][traffic_class['name']]['loss'].append(avg_loss)
                    self.metrics['traffic_classes'][traffic_class['name']]['jitter'].append(avg_jitter)
            
            # Calculate overall metrics for this iteration
            num_routed_packets = 0
            for tc in self.metrics['traffic_classes']:
                if 'latency' in self.metrics['traffic_classes'][tc]:
                    num_routed_packets += len(self.metrics['traffic_classes'][tc]['latency'])
            
            if num_routed_packets > 0:
                avg_path_length = total_path_length / num_routed_packets
                avg_path_quality = total_path_quality / num_routed_packets
                avg_packet_loss = total_packet_loss / num_routed_packets
            else:
                avg_path_length = 0
                avg_path_quality = 0
                avg_packet_loss = 0
            
            # Store metrics for this iteration
            self.metrics['convergence_times'].append(time.time() - start_time)
            self.metrics['path_lengths'].append(avg_path_length)
            self.metrics['path_qualities'].append(avg_path_quality)
            self.metrics['congestion_levels'].append(max_congestion)
            self.metrics['packet_loss_rates'].append(avg_packet_loss)
            
            # Add this iteration's paths to history
            path_history.append(iteration_paths)
            
            # Print progress update every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations")
        
        return path_history

class AdvancedEnterpriseRoutingSimulator(EnterpriseRoutingSimulator):
    """Advanced simulator for enterprise routing with detailed packet-level analysis."""

    def __init__(self, network, router):
        super().__init__(network, router)
        self.packet_logs = []  # Store detailed logs for each packet

    def run_simulation(self, num_iterations=100, packets_per_iter=50):
        """Run the advanced routing simulation with detailed packet-level analysis."""
        path_history = []

        # Simulate different traffic classes with different requirements
        traffic_classes = [
            {'name': 'voip', 'weight': 0.2, 'qos_required': True},
            {'name': 'video', 'weight': 0.3, 'qos_required': True},
            {'name': 'data', 'weight': 0.4, 'qos_required': False},
            {'name': 'backup', 'weight': 0.1, 'qos_required': False}
        ]

        for i in range(num_iterations):
            # Generate background traffic
            self.network.generate_background_traffic()

            # Update network metrics to simulate dynamic conditions
            self.network.update_dynamic_metrics()

            # Route packets for different traffic classes
            iteration_paths = []
            start_time = time.time()

            # Initialize metrics for this iteration
            total_path_length = 0
            total_path_quality = 0
            max_congestion = 0
            total_packet_loss = 0

            # Process traffic for each traffic class
            for traffic_class in traffic_classes:
                # Calculate number of packets for this class
                class_packets = int(packets_per_iter * traffic_class['weight'])
                
                for _ in range(class_packets):
                    # Pick source and destination based on traffic class
                    if traffic_class['name'] == 'voip':
                        # VoIP traffic: random pairs
                        source, destination = random.sample(list(self.network.graph.nodes()), 2)
                    elif traffic_class['name'] == 'video':
                        # Video traffic: often from core to endpoints
                        if self.network.departments['Core'] and random.random() < 0.7:
                            source = random.choice(self.network.departments['Core'])
                            # Only choose from departments that have nodes
                            valid_dest_depts = [d for d in self.network.departments.keys() 
                                             if d != 'Core' and self.network.departments[d]]
                            if valid_dest_depts:
                                dest_dept = random.choice(valid_dest_depts)
                                destination = random.choice(self.network.departments[dest_dept])
                            else:
                                # Fallback if no valid departments found
                                source, destination = random.sample(list(self.network.graph.nodes()), 2)
                        else:
                            # 30% random video traffic
                            source, destination = random.sample(list(self.network.graph.nodes()), 2)
                    elif traffic_class['name'] == 'data':
                        # Data traffic: distributed across the network
                        source, destination = random.sample(list(self.network.graph.nodes()), 2)
                    else:  # backup
                        # Backup traffic: often to core infrastructure
                        if self.network.departments['Core'] and random.random() < 0.8:
                            # 80% of backup traffic to core/servers
                            destination = random.choice(self.network.departments['Core'])
                            # Only choose from departments that have nodes
                            valid_source_depts = [d for d in self.network.departments.keys() 
                                               if d != 'Core' and self.network.departments[d]]
                            if valid_source_depts:
                                source_dept = random.choice(valid_source_depts)
                                source = random.choice(self.network.departments[source_dept])
                            else:
                                # Fallback if no valid departments found
                                source, destination = random.sample(list(self.network.graph.nodes()), 2)
                        else:
                            # 20% random backup traffic
                            source, destination = random.sample(list(self.network.graph.nodes()), 2)

                    # Route packet using the router with traffic class info if supported
                    if hasattr(self.router, 'find_path') and len(inspect.signature(self.router.find_path).parameters) >= 4:
                        # Router supports traffic class and QoS parameters
                        path, quality = self.router.find_path(
                            source, 
                            destination, 
                            traffic_class=traffic_class['name'],
                            qos_required=traffic_class.get('qos_required', False)
                        )
                    else:
                        # Basic router interface
                        path, quality = self.router.find_path(source, destination)

                    if path and len(path) > 1:
                        # Log packet details
                        packet_log = {
                            'iteration': i + 1,
                            'source': source,
                            'destination': destination,
                            'path': path,
                            'quality': quality,
                            'traffic_class': traffic_class['name'],
                            'qos_required': traffic_class.get('qos_required', False),
                            'latency': 0,
                            'congestion': 0,
                            'packet_loss': 0
                        }

                        # Calculate metrics for the path
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j + 1]
                            edge = self.network.graph[u][v]

                            # Update latency
                            packet_log['latency'] += edge['distance']

                            # Update congestion
                            packet_log['congestion'] = max(packet_log['congestion'], edge['congestion'])

                            # Update packet loss using compound probability
                            edge_loss = edge['packet_loss']
                            current_loss = packet_log['packet_loss']
                            packet_log['packet_loss'] = current_loss + edge_loss - (current_loss * edge_loss)
                            
                            # Update traffic and congestion on this edge
                            self.network.graph[u][v]['traffic'] += 1.0
                            capacity = self.network.graph[u][v]['capacity']
                            self.network.graph[u][v]['congestion'] = self.network.graph[u][v]['traffic'] / capacity
                            max_congestion = max(max_congestion, self.network.graph[u][v]['congestion'])
                            
                        # Update totals for overall metrics
                        total_path_length += len(path) - 1
                        total_path_quality += quality
                        total_packet_loss += packet_log['packet_loss']

                        # Store packet log
                        self.packet_logs.append(packet_log)

                        # Store path information
                        iteration_paths.append(packet_log)

            # Calculate overall metrics for this iteration
            num_routed_packets = len(iteration_paths)
            if num_routed_packets > 0:
                avg_path_length = total_path_length / num_routed_packets
                avg_path_quality = total_path_quality / num_routed_packets
                avg_packet_loss = total_packet_loss / num_routed_packets
            else:
                avg_path_length = 0
                avg_path_quality = 0
                avg_packet_loss = 0
                
            # Store metrics for this iteration
            self.metrics['convergence_times'].append(time.time() - start_time)
            self.metrics['path_lengths'].append(avg_path_length)
            self.metrics['path_qualities'].append(avg_path_quality)
            self.metrics['congestion_levels'].append(max_congestion)
            self.metrics['packet_loss_rates'].append(avg_packet_loss)
            
            # Process metrics by traffic class
            traffic_class_metrics = {}
            for tc in traffic_classes:
                tc_name = tc['name']
                tc_logs = [log for log in iteration_paths if log['traffic_class'] == tc_name]
                
                if tc_logs:
                    avg_latency = sum(log['latency'] for log in tc_logs) / len(tc_logs)
                    avg_loss = sum(log['packet_loss'] for log in tc_logs) / len(tc_logs)
                    
                    # Calculate jitter as variation in latencies
                    latencies = [log['latency'] for log in tc_logs]
                    if len(latencies) > 1:
                        jitter = sum(abs(latencies[i] - latencies[i-1]) for i in range(1, len(latencies))) / (len(latencies) - 1)
                    else:
                        jitter = 0
                    
                    # Store in metrics
                    self.metrics['traffic_classes'][tc_name]['latency'].append(avg_latency)
                    self.metrics['traffic_classes'][tc_name]['loss'].append(avg_loss)
                    self.metrics['traffic_classes'][tc_name]['jitter'].append(jitter)
                
            # Add this iteration's paths to history
            path_history.append(iteration_paths)

            # Print progress update every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations")

        return path_history

class PacketTracker:
    """Advanced packet tracker for detailed routing analysis across protocols."""
    
    def __init__(self):
        """Initialize the packet tracker."""
        self.packet_id_counter = 0
        self.active_packets = {}  # Currently active packets
        self.completed_packets = {}  # Successfully delivered packets
        self.dropped_packets = {}  # Dropped packets
        
        # Traffic class metrics
        self.traffic_class_metrics = {
            'voip': {'latency': [], 'loss': [], 'jitter': [], 'path_lengths': []},
            'video': {'latency': [], 'loss': [], 'jitter': [], 'path_lengths': []},
            'data': {'latency': [], 'loss': [], 'jitter': [], 'path_lengths': []},
            'backup': {'latency': [], 'loss': [], 'jitter': [], 'path_lengths': []}
        }
        
        # Protocol comparison metrics
        self.protocol_comparison = {
            'PAMR': {'paths': {}, 'metrics': {}},
            'OSPF': {'paths': {}, 'metrics': {}},
            'RIP': {'paths': {}, 'metrics': {}}
        }
        
        # Path history for each source-destination pair
        self.path_history = {}
    
    def create_packet(self, source, destination, traffic_class='data', size=1.0):
        """Create a new packet and return its ID."""
        packet_id = self.packet_id_counter
        self.packet_id_counter += 1
        
        packet = {
            'id': packet_id,
            'source': source,
            'destination': destination,
            'traffic_class': traffic_class,
            'size': size,
            'creation_time': time.time(),
            'hops': [],
            'current_node': source,
            'status': 'active',
            'metrics': {
                'latency': 0.0,
                'jitter': [],
                'congestion': [],
                'packet_loss_prob': 0.0
            }
        }
        
        self.active_packets[packet_id] = packet
        return packet_id
    
    def update_packet_position(self, packet_id, next_node, edge_metrics):
        """Update a packet's position and metrics."""
        if packet_id not in self.active_packets:
            return False
        
        packet = self.active_packets[packet_id]
        current_node = packet['current_node']
        
        # Record hop
        hop_details = {
            'from': current_node,
            'to': next_node,
            'time': time.time(),
            'latency': edge_metrics.get('latency', 0.0),
            'congestion': edge_metrics.get('congestion', 0.0),
            'packet_loss': edge_metrics.get('packet_loss', 0.0)
        }
        
        packet['hops'].append(hop_details)
        packet['current_node'] = next_node
        
        # Update packet metrics
        packet['metrics']['latency'] += hop_details['latency']
        packet['metrics']['congestion'].append(hop_details['congestion'])
        
        # Calculate jitter if we have more than one hop
        if len(packet['hops']) > 1:
            prev_latency = packet['hops'][-2]['latency']
            current_latency = hop_details['latency']
            jitter = abs(current_latency - prev_latency)
            packet['metrics']['jitter'].append(jitter)
        
        # Check if packet has reached its destination
        if next_node == packet['destination']:
            self._complete_packet(packet_id)
            return True
        
        return True
    
    def _complete_packet(self, packet_id):
        """Mark a packet as completed and update metrics."""
        if packet_id not in self.active_packets:
            return False
        
        packet = self.active_packets[packet_id]
        packet['status'] = 'completed'
        packet['completion_time'] = time.time()
        packet['delivery_time'] = packet['completion_time'] - packet['creation_time']
        
        # Move to completed packets
        self.completed_packets[packet_id] = packet
        del self.active_packets[packet_id]
        
        # Update traffic class metrics
        tc = packet['traffic_class']
        if tc in self.traffic_class_metrics:
            self.traffic_class_metrics[tc]['latency'].append(packet['metrics']['latency'])
            self.traffic_class_metrics[tc]['path_lengths'].append(len(packet['hops']))
            
            if packet['metrics']['jitter']:
                avg_jitter = sum(packet['metrics']['jitter']) / len(packet['metrics']['jitter'])
                self.traffic_class_metrics[tc]['jitter'].append(avg_jitter)
        
        # Update path history
        path_key = (packet['source'], packet['destination'])
        if path_key not in self.path_history:
            self.path_history[path_key] = []
        
        path = [packet['source']] + [hop['to'] for hop in packet['hops']]
        self.path_history[path_key].append({
            'path': path,
            'traffic_class': tc,
            'metrics': packet['metrics'],
            'time': packet['completion_time']
        })
        
        return True
    
    def drop_packet(self, packet_id, reason='congestion'):
        """Mark a packet as dropped and update metrics."""
        if packet_id not in self.active_packets:
            return False
        
        packet = self.active_packets[packet_id]
        packet['status'] = 'dropped'
        packet['drop_time'] = time.time()
        packet['drop_reason'] = reason
        
        # Move to dropped packets
        self.dropped_packets[packet_id] = packet
        del self.active_packets[packet_id]
        
        # Update traffic class metrics for packet loss
        tc = packet['traffic_class']
        if tc in self.traffic_class_metrics:
            self.traffic_class_metrics[tc]['loss'].append(1.0)  # 100% loss for this packet
        
        return True
    
    def compare_protocol_paths(self, protocols, source, destination):
        """Compare paths chosen by different protocols for the same source-destination pair."""
        path_key = (source, destination)
        comparison = {protocol: None for protocol in protocols}
        
        for protocol in protocols:
            if protocol in self.protocol_comparison and path_key in self.protocol_comparison[protocol]['paths']:
                comparison[protocol] = self.protocol_comparison[protocol]['paths'][path_key]
        
        return comparison
    
    def record_protocol_path(self, protocol, source, destination, path, metrics):
        """Record a path chosen by a specific protocol."""
        if protocol not in self.protocol_comparison:
            self.protocol_comparison[protocol] = {'paths': {}, 'metrics': {}}
        
        path_key = (source, destination)
        self.protocol_comparison[protocol]['paths'][path_key] = {
            'path': path,
            'metrics': metrics,
            'time': time.time()
        }
    
    def get_traffic_class_metrics(self, traffic_class=None):
        """Get metrics for specific traffic class or all traffic classes."""
        if traffic_class and traffic_class in self.traffic_class_metrics:
            return self.traffic_class_metrics[traffic_class]
        return self.traffic_class_metrics
    
    def get_path_statistics(self, source, destination):
        """Get detailed statistics for paths between a source and destination."""
        path_key = (source, destination)
        if path_key not in self.path_history:
            return None
        
        path_data = self.path_history[path_key]
        
        # Calculate statistics
        path_lengths = [len(item['path']) - 1 for item in path_data]
        latencies = [item['metrics']['latency'] for item in path_data]
        
        # Group by traffic class
        by_tc = {}
        for item in path_data:
            tc = item['traffic_class']
            if tc not in by_tc:
                by_tc[tc] = []
            by_tc[tc].append(item)
        
        # Calculate statistics by traffic class
        tc_stats = {}
        for tc, items in by_tc.items():
            tc_latencies = [item['metrics']['latency'] for item in items]
            tc_path_lengths = [len(item['path']) - 1 for item in items]
            
            tc_stats[tc] = {
                'count': len(items),
                'avg_latency': sum(tc_latencies) / len(tc_latencies) if tc_latencies else 0,
                'avg_path_length': sum(tc_path_lengths) / len(tc_path_lengths) if tc_path_lengths else 0
            }
        
        return {
            'total_packets': len(path_data),
            'avg_path_length': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
            'by_traffic_class': tc_stats
        }
    
def visualize_enterprise_network(network, output_dir="visualization_results"):
    """Visualize the enterprise network with departments and IP assignments."""
    plt.figure(figsize=(12, 10))
    
    # Create position map
    pos = network.positions
    
    # Draw edges with congestion-based coloring
    edge_colors = []
    for u, v in network.graph.edges():
        congestion = network.graph[u][v]['congestion']
        # Color scale: green (low) to yellow to red (high congestion)
        if congestion < 0.3:
            edge_colors.append('green')
        elif congestion < 0.6:
            edge_colors.append('orange')
        else:
            edge_colors.append('red')
    
    # Draw edges
    nx.draw_networkx_edges(
        network.graph, pos, 
        edge_color=edge_colors,
        alpha=0.6,
        arrows=True,
        arrowsize=10,
        width=1.5
    )
    
    # Draw nodes with department-based coloring
    department_colors = {
        'Core': 'red',
        'IT': 'blue',
        'Finance': 'green',
        'HR': 'purple',
        'Marketing': 'orange',
        'Engineering': 'cyan',
        'Operations': 'magenta',
        'Sales': 'brown',
        'Executive': 'black'
    }
    
    # Create node color map
    node_colors = []
    for node in network.graph.nodes():
        for dept, nodes in network.departments.items():
            if node in nodes:
                node_colors.append(department_colors.get(dept, 'gray'))
                break
        else:
            node_colors.append('gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        network.graph, pos,
        node_color=node_colors,
        node_size=100,
        alpha=0.8
    )
    
    # Draw labels with IP addresses (only show a subset for clarity)
    labels = {}
    for dept, nodes in network.departments.items():
        if dept == 'Core':
            # Always show Core nodes
            for node in nodes:
                labels[node] = f"{node}\n{network.ip_addresses.get(node, '')}"
        else:
            # Show only a subset of nodes from other departments
            sample_size = min(2, len(nodes))
            for node in random.sample(nodes, sample_size):
                labels[node] = f"{node}\n{network.ip_addresses.get(node, '')}"
    
    nx.draw_networkx_labels(
        network.graph, pos,
        labels=labels,
        font_size=8,
        font_weight='bold'
    )
    
    # Create legend for departments
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10, label=dept)
                      for dept, color in department_colors.items() if dept in network.departments]
    
    # Create legend for congestion levels
    congestion_elements = [
        plt.Line2D([0], [0], color='green', lw=2, label='Low Congestion'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Medium Congestion'),
        plt.Line2D([0], [0], color='red', lw=2, label='High Congestion')
    ]
    
    # Add legends
    plt.legend(handles=legend_elements, title="Departments", loc='upper left')
    plt.legend(handles=congestion_elements, title="Congestion Levels", loc='upper right')
    
    # Set title and remove axis
    plt.title(f"Enterprise Network Topology with IP Assignments\n{len(network.graph.nodes())} nodes, {len(network.graph.edges())} links")
    plt.axis('off')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"enterprise_network_topology_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def visualize_routing_comparison(simulators, output_dir="visualization_results"):
    """Visualize comparison results between different routing algorithms."""
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot convergence times
    ax = axs[0, 0]
    for name, simulator in simulators.items():
        ax.plot(simulator.metrics['convergence_times'], label=name)
    ax.set_title('Convergence Time Comparison')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    ax.grid(True)
    
    # Plot path lengths
    ax = axs[0, 1]
    for name, simulator in simulators.items():
        ax.plot(simulator.metrics['path_lengths'], label=name)
    ax.set_title('Path Length Comparison')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Path Length')
    ax.legend()
    ax.grid(True)
    
    # Plot path qualities
    ax = axs[1, 0]
    for name, simulator in simulators.items():
        ax.plot(simulator.metrics['path_qualities'], label=name)
    ax.set_title('Path Quality Comparison')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Path Quality')
    ax.legend()
    ax.grid(True)
    
    # Plot congestion levels
    ax = axs[1, 1]
    for name, simulator in simulators.items():
        ax.plot(simulator.metrics['congestion_levels'], label=name)
    ax.set_title('Congestion Level Comparison')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Congestion')
    ax.legend()
    ax.grid(True)
    
    # Plot packet loss rates
    ax = axs[2, 0]
    for name, simulator in simulators.items():
        ax.plot(simulator.metrics['packet_loss_rates'], label=name)
    ax.set_title('Packet Loss Rate Comparison')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Packet Loss Rate')
    ax.legend()
    ax.grid(True)
    
    # Plot traffic class performance for PAMR (focused on latency)
    ax = axs[2, 1]
    traffic_classes = ['voip', 'video', 'data', 'backup']
    for tc in traffic_classes:
        if 'PAMR' in simulators and tc in simulators['PAMR'].metrics['traffic_classes']:
            ax.plot(simulators['PAMR'].metrics['traffic_classes'][tc]['latency'], label=f"{tc}")
    ax.set_title('PAMR Latency by Traffic Class')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Latency')
    ax.legend()
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"enterprise_routing_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def visualize_protocol_path_comparison(network, simulators, num_pairs=5, output_dir="visualization_results"):
    """Visualize the path selection differences between routing protocols for specific source-destination pairs."""
    # Choose some random source-destination pairs to analyze
    nodes = list(network.graph.nodes())
    pairs = []
    
    # Select some interesting pairs (higher traffic nodes)
    node_degrees = sorted([(n, network.graph.degree(n)) for n in nodes], key=lambda x: x[1], reverse=True)
    high_traffic_nodes = [n for n, _ in node_degrees[:min(10, len(node_degrees))]]
    
    # Generate some random pairs from high traffic nodes
    for _ in range(min(num_pairs, len(high_traffic_nodes))):
        source, dest = random.sample(high_traffic_nodes, 2)
        if source != dest and (source, dest) not in pairs:
            pairs.append((source, dest))
    
    # Add some random pairs if needed
    while len(pairs) < num_pairs:
        source, dest = random.sample(nodes, 2)
        if source != dest and (source, dest) not in pairs:
            pairs.append((source, dest))
    
    # Create visualizations for each pair
    viz_paths = []
    metrics_data = {}
    
    for source, dest in pairs:
        # Get paths from each protocol
        paths = {}
        for name, simulator in simulators.items():
            if hasattr(simulator.router, 'find_path'):
                path, quality = simulator.router.find_path(source, dest)
                if path and len(path) > 1:
                    paths[name] = (path, quality)
        
        if len(paths) < 2:
            continue  # Skip if fewer than 2 protocols found paths
        
        # Create visualization for this pair
        plt.figure(figsize=(14, 10))
        
        # Create subplot for path visualization
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        
        # Get network positions - use spring layout if not available
        if hasattr(network, 'positions') and network.positions:
            pos = network.positions
        else:
            pos = nx.spring_layout(network.graph, seed=42)
        
        # Draw base network structure
        nx.draw_networkx_edges(
            network.graph, pos,
            alpha=0.1,
            width=1,
            ax=ax1
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            network.graph, pos,
            node_size=50,
            alpha=0.3,
            ax=ax1
        )
        
        # Highlight source and destination
        nx.draw_networkx_nodes(
            network.graph, pos,
            nodelist=[source, dest],
            node_size=150,
            node_color='red',
            ax=ax1
        )
        
        # Draw labels for source and destination only
        nx.draw_networkx_labels(
            network.graph, pos,
            labels={n: str(n) for n in [source, dest]},
            font_size=10,
            font_weight='bold',
            ax=ax1
        )
        
        # Draw each protocol's path with different colors
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        line_styles = ['-', '--', '-.', ':', '-']
        
        # Track metrics for comparison
        path_metrics = {name: {} for name in paths}
        
        # Draw each path
        for i, (name, (path, quality)) in enumerate(paths.items()):
            # Create edges list for the path
            path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
            
            # Draw the path
            nx.draw_networkx_edges(
                network.graph, pos,
                edgelist=path_edges,
                edge_color=colors[i % len(colors)],
                width=2,
                alpha=0.8,
                style=line_styles[i % len(line_styles)],
                ax=ax1
            )
            
            # Calculate metrics for this path
            path_length = len(path) - 1
            total_distance = 0
            max_congestion = 0
            total_loss = 0
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                edge_data = network.graph[u][v]
                
                total_distance += edge_data.get('distance', 1.0)
                max_congestion = max(max_congestion, edge_data.get('congestion', 0.0))
                edge_loss = edge_data.get('packet_loss', 0.01)
                total_loss = total_loss + edge_loss - (total_loss * edge_loss)  # Compound probability
            
            # Store metrics for this protocol
            path_metrics[name] = {
                'length': path_length,
                'distance': total_distance,
                'congestion': max_congestion,
                'loss': total_loss,
                'quality': quality
            }
        
        # Set title and layout for path visualization
        ax1.set_title(f"Path Comparison: Node {source} to Node {dest}")
        ax1.axis('off')
        
        # Create subplot for metrics comparison
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        
        # Create bar chart comparing path metrics
        protocol_names = list(path_metrics.keys())
        metrics = ['distance', 'congestion', 'loss']
        
        # Create bar chart data
        bar_data = {metric: [path_metrics[name][metric] for name in protocol_names] for metric in metrics}
        
        # Create bar chart
        x = np.arange(len(protocol_names))
        width = 0.25
        
        # Plot bars for each metric
        ax2.bar(x - width, bar_data['distance'], width, label='Distance', alpha=0.7)
        ax2.bar(x, bar_data['congestion'], width, label='Congestion', alpha=0.7)
        ax2.bar(x + width, bar_data['loss'], width, label='Packet Loss', alpha=0.7)
        
        # Add labels and legend
        ax2.set_xlabel('Protocol')
        ax2.set_ylabel('Value')
        ax2.set_title('Path Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(protocol_names)
        ax2.legend()
        
        # Create subplot for path quality comparison
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        
        # Create bar chart comparing path quality
        quality_values = [path_metrics[name]['quality'] for name in protocol_names]
        
        # Plot bars for path quality
        bars = ax3.bar(protocol_names, quality_values, alpha=0.7)
        
        # Add labels
        ax3.set_xlabel('Protocol')
        ax3.set_ylabel('Path Quality')
        ax3.set_title('Path Quality Comparison')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"path_comparison_{source}_to_{dest}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_paths.append(output_path)
        metrics_data[(source, dest)] = path_metrics
    
    return viz_paths, metrics_data

def visualize_traffic_class_performance(simulators, output_dir="visualization_results"):
    """Visualize protocol performance across different traffic classes."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have traffic class data
    has_traffic_class_data = False
    for name, simulator in simulators.items():
        if simulator.metrics.get('traffic_classes'):
            has_traffic_class_data = True
            break
    
    if not has_traffic_class_data:
        print("No traffic class data available for visualization")
        return None
    
    # Create figure with multiple subplots for different metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Metrics to visualize
    metrics = ['latency', 'loss', 'jitter']
    traffic_classes = ['voip', 'video', 'data', 'backup']
    
    # Get last values for each metric and traffic class
    protocols = list(simulators.keys())
    
    # Create data structures for the plots
    latency_data = {}
    loss_data = {}
    jitter_data = {}
    
    # Collect data for each protocol
    for protocol in protocols:
        simulator = simulators[protocol]
        
        latency_data[protocol] = []
        loss_data[protocol] = []
        jitter_data[protocol] = []
        
        for tc in traffic_classes:
            # Get metrics for this traffic class
            tc_metrics = simulator.metrics.get('traffic_classes', {}).get(tc, {})
            
            # Get the average of last 5 values for each metric or 0 if not available
            if 'latency' in tc_metrics and tc_metrics['latency']:
                avg_latency = sum(tc_metrics['latency'][-5:]) / min(5, len(tc_metrics['latency']))
                latency_data[protocol].append(avg_latency)
            else:
                latency_data[protocol].append(0)
                
            if 'loss' in tc_metrics and tc_metrics['loss']:
                avg_loss = sum(tc_metrics['loss'][-5:]) / min(5, len(tc_metrics['loss']))
                loss_data[protocol].append(avg_loss)
            else:
                loss_data[protocol].append(0)
                
            if 'jitter' in tc_metrics and tc_metrics['jitter']:
                avg_jitter = sum(tc_metrics['jitter'][-5:]) / min(5, len(tc_metrics['jitter']))
                jitter_data[protocol].append(avg_jitter)
            else:
                jitter_data[protocol].append(0)
    
    # 1. Create group bar chart for latency by traffic class
    ax = axs[0, 0]
    x = np.arange(len(traffic_classes))
    width = 0.8 / len(protocols)
    
    for i, protocol in enumerate(protocols):
        offset = (i - len(protocols) / 2 + 0.5) * width
        ax.bar(x + offset, latency_data[protocol], width, label=protocol)
    
    ax.set_title('Average Latency by Traffic Class')
    ax.set_xticks(x)
    ax.set_xticklabels(traffic_classes)
    ax.set_ylabel('Latency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Create group bar chart for packet loss by traffic class
    ax = axs[0, 1]
    
    for i, protocol in enumerate(protocols):
        offset = (i - len(protocols) / 2 + 0.5) * width
        ax.bar(x + offset, loss_data[protocol], width, label=protocol)
    
    ax.set_title('Packet Loss Rate by Traffic Class')
    ax.set_xticks(x)
    ax.set_xticklabels(traffic_classes)
    ax.set_ylabel('Packet Loss Rate')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Create group bar chart for jitter by traffic class
    ax = axs[1, 0]
    
    for i, protocol in enumerate(protocols):
        offset = (i - len(protocols) / 2 + 0.5) * width
        ax.bar(x + offset, jitter_data[protocol], width, label=protocol)
    
    ax.set_title('Jitter by Traffic Class')
    ax.set_xticks(x)
    ax.set_xticklabels(traffic_classes)
    ax.set_ylabel('Jitter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Create box plots for overall performance comparison
    ax = axs[1, 1]
    
    # Collect all metric data for each protocol
    box_data = []
    
    for protocol in protocols:
        protocol_data = []
        
        # Normalize metrics to comparable ranges
        norm_latency = [val / max(max(latency_data[p]) for p in protocols) if max(max(latency_data[p]) for p in protocols) > 0 else 0 for val in latency_data[protocol]]
        norm_loss = loss_data[protocol]  # Already in 0-1 range
        norm_jitter = [val / max(max(jitter_data[p]) for p in protocols) if max(max(jitter_data[p]) for p in protocols) > 0 else 0 for val in jitter_data[protocol]]
        
        # Combine all normalized metrics
        protocol_data.extend(norm_latency)
        protocol_data.extend(norm_loss)
        protocol_data.extend(norm_jitter)
        
        box_data.append(protocol_data)
    
    # Create box plot
    ax.boxplot(box_data, labels=protocols, vert=True, patch_artist=True)
    ax.set_title('Overall Protocol Performance')
    ax.set_ylabel('Normalized Metric Value (lower is better)')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"traffic_class_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_packet_iteration_analysis(simulators, output_dir="visualization_results"):
    """Visualize packet performance across simulation iterations."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Extract iteration data from all simulators
    for name, simulator in simulators.items():
        if not simulator.metrics.get('packet_loss_rates', []):
            continue
            
        iterations = range(1, len(simulator.metrics['packet_loss_rates']) + 1)
        loss_rates = simulator.metrics['packet_loss_rates']
        
        # Plot packet loss rate over iterations
        plt.plot(iterations, loss_rates, label=f"{name}", linewidth=2)
    
    plt.title("Packet Loss Rate Over Simulation Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Packet Loss Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    output_path = os.path.join(output_dir, f"packet_iteration_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more detailed analysis with multiple metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Packet Loss Rate
    ax = axs[0, 0]
    for name, simulator in simulators.items():
        if simulator.metrics.get('packet_loss_rates', []):
            iterations = range(1, len(simulator.metrics['packet_loss_rates']) + 1)
            loss_rates = simulator.metrics['packet_loss_rates']
            ax.plot(iterations, loss_rates, label=name, linewidth=2)
    
    ax.set_title("Packet Loss Rate Over Time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Packet Loss Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Path Quality
    ax = axs[0, 1]
    for name, simulator in simulators.items():
        if simulator.metrics.get('path_qualities', []):
            iterations = range(1, len(simulator.metrics['path_qualities']) + 1)
            qualities = simulator.metrics['path_qualities']
            ax.plot(iterations, qualities, label=name, linewidth=2)
    
    ax.set_title("Path Quality Over Time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Path Quality")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Convergence Time
    ax = axs[1, 0]
    for name, simulator in simulators.items():
        if simulator.metrics.get('convergence_times', []):
            iterations = range(1, len(simulator.metrics['convergence_times']) + 1)
            times = simulator.metrics['convergence_times']
            ax.plot(iterations, times, label=name, linewidth=2)
    
    ax.set_title("Convergence Time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (seconds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Path Lengths
    ax = axs[1, 1]
    for name, simulator in simulators.items():
        if simulator.metrics.get('path_lengths', []):
            iterations = range(1, len(simulator.metrics['path_lengths']) + 1)
            lengths = simulator.metrics['path_lengths']
            ax.plot(iterations, lengths, label=name, linewidth=2)
    
    ax.set_title("Average Path Length")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hops")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save detailed figure
    detailed_path = os.path.join(output_dir, f"detailed_performance_analysis.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [output_path, detailed_path]

def run_enterprise_simulation(
    num_nodes=40, 
    connectivity=0.15, 
    seed=42, 
    variation_factor=0.2,
    alpha=2.5, 
    beta=4.0, 
    gamma=10.0,
    num_iterations=15,
    packets_per_iter=40,
    with_comparison_protocols=True,
    output_dir="visualization_results"
):
    print("Initializing simulation...")
    print(f"Parameters: {num_nodes} nodes, {connectivity} connectivity, α={alpha}, β={beta}, γ={gamma}")
    
    # Create enterprise network
    network = EnterpriseNetwork(
        num_nodes=num_nodes, 
        connectivity=connectivity, 
        seed=seed,
        variation_factor=variation_factor
    )
    print(f"Network created with {len(network.graph.nodes())} nodes and {len(network.graph.edges())} links")
    
    # Print department statistics
    print("\nDepartment Node Distribution:")
    for dept, nodes in network.departments.items():
        print(f"  {dept}: {len(nodes)} nodes")
    
    # Create visualization of network
    print("\nVisualizing network topology...")
    network_viz_path = visualize_enterprise_network(network, output_dir=output_dir)
    print(f"Network visualization saved to: {network_viz_path}")
    
    # Create identical network copies for different routing algorithms
    network_data = pickle.dumps(network)
    
    # Initialize PAMR router - USE OPTIMIZED VERSION
    print(f"\nInitializing Optimized PAMR router...")
    pamr_network = pickle.loads(network_data)
    pamr_router = OptimizedPAMRRouter(pamr_network.graph, alpha=alpha, beta=beta, gamma=gamma)
    pamr_simulator = AdvancedEnterpriseRoutingSimulator(pamr_network, pamr_router)
    
    # Initialize comparison routers if available and requested
    simulators = {'PAMR': pamr_simulator}
    
    if with_comparison_protocols:
        try:
            print("Initializing OSPF router...")
            ospf_network = pickle.loads(network_data)
            ospf_router = OSPFRouter(ospf_network.graph)
            ospf_simulator = EnterpriseRoutingSimulator(ospf_network, ospf_router)
            simulators['OSPF'] = ospf_simulator
            
            print("Initializing RIP router...")
            rip_network = pickle.loads(network_data)
            rip_router = RIPRouter(rip_network.graph)
            rip_simulator = EnterpriseRoutingSimulator(rip_network, rip_router)
            simulators['RIP'] = rip_simulator
        except Exception as e:
            print(f"Warning: Could not initialize comparison routers: {e}. Continuing with PAMR only.")
    
    # Run simulations
    print(f"\nRunning simulations with {num_iterations} iterations and {packets_per_iter} packets per iteration...")
    
    for name, simulator in simulators.items():
        print(f"\nRunning {name} simulation...")
        simulator.run_simulation(num_iterations, packets_per_iter)
        print(f"{name} simulation completed")
    
    # Generate comparison visualizations
    print("\nGenerating comparison visualizations...")
    comparison_viz_path = visualize_routing_comparison(simulators, output_dir=output_dir)
    print(f"Comparison saved to: {comparison_viz_path}")
    
    traffic_viz_path = visualize_traffic_class_performance(simulators, output_dir=output_dir)
    print(f"Traffic class performance saved to: {traffic_viz_path}")
    
    iter_viz_path = visualize_packet_iteration_analysis(simulators, output_dir=output_dir)  
    print(f"Packet iteration analysis saved to: {iter_viz_path}")
    
    path_viz_images, _ = visualize_protocol_path_comparison(network, simulators, output_dir=output_dir)
    if path_viz_images:
        print(f"Path comparisons saved to visualization_results directory")
    
    return comparison_viz_path, traffic_viz_path, iter_viz_path

class OptimizedPAMRRouter(PAMRRouter):
    def __init__(self, graph, alpha=2.5, beta=4.0, gamma=10.0, adapt_weights=True):
        super().__init__(graph, alpha, beta, gamma, adapt_weights)
        
        # Enhanced parameters for better routing
        self.path_update_interval = 10
        self.pheromone_evaporation = 0.97
        self.min_path_share = 0.15
        
        # Packet loss optimization parameters
        self.packet_loss_weight = 3.0
        self.congestion_threshold = 0.35
        self.voip_latency_factor = 2.0
        self.video_bandwidth_factor = 1.5
        
        # Traffic handlers
        self.traffic_handlers = {
            'voip': self._handle_voip_traffic,
            'video': self._handle_video_traffic,
            'data': self._handle_data_traffic,
            'backup': self._handle_backup_traffic
        }
    
    def find_path(self, source, destination, max_steps=100, traffic_class='standard', qos_required=False):
        # Check if we have a specialized handler for this traffic class
        if traffic_class in self.traffic_handlers:
            return self.traffic_handlers[traffic_class](source, destination, qos_required)
        
        # Fall back to standard path finding with enhanced packet loss avoidance
        return self._find_optimized_path(source, destination, qos_required)
    
    def _find_optimized_path(self, source, destination, qos_required=False):
        if source == destination:
            return [source], 1.0
        
        # Path key for caching
        path_key = (source, destination)
        
        # Check cache first for recently calculated paths
        if self.use_advanced_cache and path_key in self.path_cache:
            cache_entry = self.path_cache[path_key]
            cache_age = self.iteration - cache_entry['iteration']
            
            if cache_age < self.cache_ttl:
                self.cache_hits += 1
                return cache_entry['path'], cache_entry['quality']
        
        self.cache_misses += 1
        
        # Try to find a path with minimal packet loss
        min_loss_result = self._find_min_loss_path(source, destination)
        
        # If we got a valid result tuple with a path
        if min_loss_result and isinstance(min_loss_result, tuple) and len(min_loss_result) == 2:
            path, quality = min_loss_result
            return path, quality
        
        # If no specific path found, use the standard router
        try:
            # Use regular weighted path finding
            path = nx.shortest_path(self.graph, source, destination, weight='distance')
            quality = self._calculate_enhanced_path_quality(path)
            
            # Update pheromones and cache the path
            self._update_pheromones(path, quality)
            self.path_cache[path_key] = {
                'path': path,
                'quality': quality,
                'iteration': self.iteration
            }
            
            return path, quality
        except nx.NetworkXNoPath:
            # Return an empty path with 0 quality if no path found
            return [], 0.0
    
    def _calculate_enhanced_path_quality(self, path):
        """Calculate path quality with increased weight on packet loss."""
        if len(path) < 2:
            return 0
        
        total_distance = 0
        max_congestion = 0
        total_packet_loss = 0
        min_bandwidth = float('inf')
        
        # Calculate metrics along the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.graph[u][v]
            
            # Get edge metrics
            distance = edge_data.get('distance', 1.0)
            congestion = edge_data.get('congestion', 0.0)
            packet_loss = edge_data.get('packet_loss', 0.01)
            bandwidth = edge_data.get('bandwidth', 1.0)
            
            # Accumulate metrics
            total_distance += distance
            max_congestion = max(max_congestion, congestion)
            # Use compound probability for packet loss
            total_packet_loss = total_packet_loss + packet_loss - (total_packet_loss * packet_loss)
            min_bandwidth = min(min_bandwidth, bandwidth)
        
        # Calculate quality with higher weight for packet loss
        distance_factor = total_distance / max(1, len(path) - 1)
        congestion_factor = max_congestion ** 2  # Quadratic penalty for congestion
        packet_loss_factor = total_packet_loss * self.packet_loss_weight  # Enhanced packet loss weight
        bandwidth_factor = 1.0 / max(0.1, min_bandwidth)
        
        # Calculate inverse quality (lower is better)
        inverse_quality = (
            0.3 * distance_factor +  # Distance component
            0.3 * congestion_factor +  # Congestion component
            0.35 * packet_loss_factor +  # Packet loss component (higher weight)
            0.05 * bandwidth_factor +  # Bandwidth component
            0.1  # Prevent division by zero
        )
        
        # Return quality (higher is better)
        return 1.0 / inverse_quality
    
    def _handle_voip_traffic(self, source, destination, qos_required):
        """Specialized handler for VoIP traffic with strict latency and jitter requirements."""
        # For VoIP, prioritize low latency and minimal packet loss over bandwidth
        try:
            # Define a weight function specifically for VoIP
            def voip_weight(u, v, data):
                distance = data.get('distance', 1.0)
                packet_loss = data.get('packet_loss', 0.01)
                congestion = data.get('congestion', 0.0)
                
                # VoIP is extremely sensitive to latency and packet loss
                # but less concerned with bandwidth
                latency_factor = distance * self.voip_latency_factor
                loss_factor = packet_loss * 20.0  # Severely penalize any packet loss
                congestion_factor = congestion * 5.0  # Avoid congested links
                
                return latency_factor + loss_factor + congestion_factor
            
            # Find path using VoIP-specific weight function
            path = nx.shortest_path(self.graph, source, destination, weight=voip_weight)
            quality = self._calculate_enhanced_path_quality(path)
            
            # Update pheromones and traffic
            self._update_pheromones(path, quality)
            
            return path, quality
        except (nx.NetworkXNoPath, Exception):
            # Fall back to the optimized path if voice-specific path fails
            return self._find_optimized_path(source, destination, qos_required)
    
    def _handle_video_traffic(self, source, destination, qos_required):
        """Specialized handler for video traffic with bandwidth and loss requirements."""
        # For video, balance bandwidth needs with packet loss avoidance
        try:
            # Define a weight function specifically for video
            def video_weight(u, v, data):
                distance = data.get('distance', 1.0)
                packet_loss = data.get('packet_loss', 0.01)
                congestion = data.get('congestion', 0.0)
                bandwidth = data.get('bandwidth', 1.0)
                
                # Video needs good bandwidth and low packet loss
                bandwidth_factor = (1.0 / max(0.1, bandwidth)) * self.video_bandwidth_factor
                loss_factor = packet_loss * 15.0  # High penalty for packet loss
                congestion_factor = congestion * 3.0  # Moderate congestion avoidance
                
                return distance + bandwidth_factor + loss_factor + congestion_factor
            
            # Find path using video-specific weight function
            path = nx.shortest_path(self.graph, source, destination, weight=video_weight)
            quality = self._calculate_enhanced_path_quality(path)
            
            # Update pheromones and traffic
            self._update_pheromones(path, quality)
            
            return path, quality
        except (nx.NetworkXNoPath, Exception):
            # Fall back to the optimized path if video-specific path fails
            return self._find_optimized_path(source, destination, qos_required)
    
    def _handle_data_traffic(self, source, destination, qos_required):
        """Handler for general data traffic with balanced requirements."""
        # For regular data, use a balanced approach
        return self._find_optimized_path(source, destination, qos_required)
    
    def _handle_backup_traffic(self, source, destination, qos_required):
        """Handler for backup traffic prioritizing reliability over speed."""
        # For backup traffic, prioritize reliability over speed
        try:
            # Define a weight function specifically for backup traffic
            def backup_weight(u, v, data):
                distance = data.get('distance', 1.0)
                packet_loss = data.get('packet_loss', 0.01)
                congestion = data.get('congestion', 0.0)
                
                # Backup traffic is less concerned with latency but needs reliability
                loss_factor = packet_loss * 10.0  # High penalty for packet loss
                congestion_factor = congestion * 2.0  # Some congestion avoidance
                
                # Distance is less important for backup traffic
                return (distance * 0.5) + loss_factor + congestion_factor
            
            # Find path using backup-specific weight function
            path = nx.shortest_path(self.graph, source, destination, weight=backup_weight)
            quality = self._calculate_enhanced_path_quality(path)
            
            # Update pheromones and traffic
            self._update_pheromones(path, quality)
            
            return path, quality
        except (nx.NetworkXNoPath, Exception):
            # Fall back to the optimized path if backup-specific path fails
            return self._find_optimized_path(source, destination, qos_required)

    def _find_min_loss_path(self, source, destination):
        """Find path with minimum packet loss."""
        if source == destination:
            return [source], 1.0
        
        # Path key for caching
        path_key = (source, destination)
        
        # Check cache first for recently calculated paths
        if self.use_advanced_cache and path_key in self.path_cache:
            cache_entry = self.path_cache[path_key]
            cache_age = self.iteration - cache_entry['iteration']
            
            if cache_age < self.cache_ttl:
                self.cache_hits += 1
                return cache_entry['path'], cache_entry['quality']
        
        self.cache_misses += 1
        
        # Try to find a path with minimal packet loss
        try:
            # Define weight function that penalizes high packet loss
            def loss_weight(u, v, data):
                # Get packet loss probability, default to 1% if not defined
                loss = data.get('packet_loss', 0.01)
                
                # Apply exponential scaling to heavily penalize high loss links
                return 1.0 + (loss * 100) ** 2
            
            # Use shortest path with the loss-based weight function
            path = nx.shortest_path(self.graph, source, destination, weight=loss_weight)
            
            if path:
                # Calculate quality with higher weight on packet loss
                quality = self._calculate_enhanced_path_quality(path)
                
                # Update pheromones and cache the path
                self._update_pheromones(path, quality)
                self.path_cache[path_key] = {
                    'path': path,
                    'quality': quality,
                    'iteration': self.iteration
                }
                
                return path, quality
        except (nx.NetworkXNoPath, Exception):
            pass
        
        # If no specific path found, use the standard router
        return super().find_path(source, destination)

    def _calculate_enhanced_path_quality(self, path):
        if len(path) < 2:
            return 0
        
        total_distance = 0
        max_congestion = 0
        total_packet_loss = 0
        min_bandwidth = float('inf')
        
        # Calculate metrics along the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.graph[u][v]
            
            # Get edge metrics
            distance = edge_data.get('distance', 1.0)
            congestion = edge_data.get('congestion', 0.0)
            packet_loss = edge_data.get('packet_loss', 0.01)
            bandwidth = edge_data.get('bandwidth', 1.0)
            
            # Accumulate metrics
            total_distance += distance
            max_congestion = max(max_congestion, congestion)
            # Use compound probability for packet loss
            total_packet_loss = total_packet_loss + packet_loss - (total_packet_loss * packet_loss)
            min_bandwidth = min(min_bandwidth, bandwidth)
        
        # Calculate quality with higher weight for packet loss
        distance_factor = total_distance / max(1, len(path) - 1)
        congestion_factor = max_congestion ** 2  # Quadratic penalty for congestion
        packet_loss_factor = total_packet_loss * self.packet_loss_weight
        bandwidth_factor = 1.0 / max(0.1, min_bandwidth)
        
        # Calculate inverse quality (lower is better)
        inverse_quality = (
            0.3 * distance_factor +
            0.3 * congestion_factor +
            0.35 * packet_loss_factor +
            0.05 * bandwidth_factor +
            0.1  # Prevent division by zero
        )
        
        # Return quality (higher is better)
        return 1.0 / inverse_quality

    def _handle_voip_traffic(self, source, destination, qos_required):
        try:
            # Define a weight function specifically for VoIP
            def voip_weight(u, v, data):
                distance = data.get('distance', 1.0)
                packet_loss = data.get('packet_loss', 0.01)
                congestion = data.get('congestion', 0.0)
                
                # VoIP needs low latency and low packet loss
                latency_factor = distance * self.voip_latency_factor
                loss_factor = packet_loss * 20.0
                congestion_factor = congestion * 5.0
                
                return latency_factor + loss_factor + congestion_factor
            
            # Find path using VoIP-specific weight function
            path = nx.shortest_path(self.graph, source, destination, weight=voip_weight)
            quality = self._calculate_enhanced_path_quality(path)
            
            # Update pheromones and traffic
            self._update_pheromones(path, quality)
            
            return path, quality
        except (nx.NetworkXNoPath, Exception):
            # Fall back to the optimized path if voice-specific path fails
            return self._find_optimized_path(source, destination, qos_required)

    def _handle_video_traffic(self, source, destination, qos_required):
        try:
            # Define a weight function specifically for video
            def video_weight(u, v, data):
                distance = data.get('distance', 1.0)
                packet_loss = data.get('packet_loss', 0.01)
                congestion = data.get('congestion', 0.0)
                bandwidth = data.get('bandwidth', 1.0)
                
                # Video needs good bandwidth and low packet loss
                bandwidth_factor = (1.0 / max(0.1, bandwidth)) * self.video_bandwidth_factor
                loss_factor = packet_loss * 15.0
                congestion_factor = congestion * 3.0
                
                return distance + bandwidth_factor + loss_factor + congestion_factor
            
            # Find path using video-specific weight function
            path = nx.shortest_path(self.graph, source, destination, weight=video_weight)
            quality = self._calculate_enhanced_path_quality(path)
            
            # Update pheromones and traffic
            self._update_pheromones(path, quality)
            
            return path, quality
        except (nx.NetworkXNoPath, Exception):
            # Fall back to the optimized path if video-specific path fails
            return self._find_optimized_path(source, destination, qos_required)

    def _handle_data_traffic(self, source, destination, qos_required):
        # For regular data, use a balanced approach
        return self._find_optimized_path(source, destination, qos_required)

    def _handle_backup_traffic(self, source, destination, qos_required):
        try:
            # Define a weight function specifically for backup traffic
            def backup_weight(u, v, data):
                distance = data.get('distance', 1.0)
                packet_loss = data.get('packet_loss', 0.01)
                congestion = data.get('congestion', 0.0)
                
                # Backup traffic needs reliability over speed
                loss_factor = packet_loss * 10.0
                congestion_factor = congestion * 2.0
                
                return (distance * 0.5) + loss_factor + congestion_factor
            
            # Find path using backup-specific weight function
            path = nx.shortest_path(self.graph, source, destination, weight=backup_weight)
            quality = self._calculate_enhanced_path_quality(path)
            
            # Update pheromones and traffic
            self._update_pheromones(path, quality)
            
            return path, quality
        except (nx.NetworkXNoPath, Exception):
            # Fall back to the optimized path if backup-specific path fails
            return self._find_optimized_path(source, destination, qos_required)

if __name__ == "__main__":
    try:
        print("PAMR Enterprise IP Routing Simulation")
        print("======================================")
        
        # Simple user input with defaults
        use_defaults = input("Use default parameters? (y/n, default: y): ").strip().lower() != 'n'
        
        if use_defaults:
            print("Using default parameters")
            comparison_viz, traffic_viz, iter_viz = run_enterprise_simulation(
                num_nodes=60,
                connectivity=0.2,
                num_iterations=30,
                packets_per_iter=50
            )
        else:
            # Get basic parameters
            num_nodes = int(input("Number of nodes (20-100, default: 60): ") or 60)
            num_iterations = int(input("Number of iterations (10-50, default: 30): ") or 30)
            with_comparison = input("Include comparison protocols (OSPF, RIP)? (y/n, default: y): ").strip().lower() != 'n'
            
            # Run simulation
            comparison_viz, traffic_viz, iter_viz = run_enterprise_simulation(
                num_nodes=num_nodes,
                num_iterations=num_iterations,
                with_comparison_protocols=with_comparison
            )
        
        print(f"\nSimulation completed. Visualizations are in visualization_results directory.")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError in simulation: {e}")
        import traceback
        traceback.print_exc()
