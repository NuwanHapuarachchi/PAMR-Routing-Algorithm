import random
import numpy as np
import networkx as nx
import heapq
from collections import defaultdict

class PAMRRouter:
    """Path selection and routing logic for PAMR protocol with advanced features."""
    
    def __init__(self, graph, alpha=2.0, beta=3.0, gamma=8.0, adapt_weights=True):
        self.graph = graph
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Distance importance
        self.gamma = gamma  # Congestion importance
        self.adapt_weights = adapt_weights  # Dynamically adapt weights
        self.path_history = {}  # Track routing history
        self.iteration = 0  # Track iterations
        
        # Advanced routing features
        self.pheromone_table = defaultdict(dict)  # Enhanced pheromone table
        self.routing_table = defaultdict(dict)    # Global routing table
        self.link_state_db = {}                   # Link state database
        self.path_cache = {}                      # Path cache for quick lookups
        self.congestion_history = defaultdict(list)  # Track historical congestion
        self.traffic_matrix = defaultdict(dict)      # Traffic matrix for prediction
        
        # Algorithm parameters
        self.path_update_interval = 20   # Update core paths less frequently (was 5)
        self.pheromone_evaporation = 0.95 # Slower pheromone evaporation (was 0.9)
        self.local_search_depth = 2      # Reduced depth of local path improvement search (was 3)
        self.load_balancing_threshold = 0.25  # Higher threshold to reduce alternative path computations (was 0.15)
        self.quick_reroute_enabled = True     # Enable fast rerouting
        self.use_advanced_cache = True        # Enable advanced caching
        self.cache_ttl = 10                   # Time to live for cache entries
        
        # Cache hit rate tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize the routing system
        self._initialize_routing()
    
    def _initialize_routing(self):
        """Initialize the routing system with proactive discovery."""
        # Initialize pheromone for all edges
        for u, v, data in self.graph.edges(data=True):
            self.pheromone_table[u][v] = 1.0
            
            # Initialize link state database with edge properties
            if u not in self.link_state_db:
                self.link_state_db[u] = {}
            self.link_state_db[u][v] = {
                'distance': data.get('distance', 1.0),
                'bandwidth': data.get('bandwidth', 1.0),
                'congestion': data.get('congestion', 0.0),
                'last_updated': 0
            }
        
        # Pre-compute initial routing tables
        self._update_core_paths()
    
    def _update_core_paths(self):
        """Update the core routing paths for all nodes using a modified Dijkstra algorithm."""
        # Only process a subset of nodes - this significantly reduces computation time
        nodes = list(self.graph.nodes())
        
        # Process only major nodes or a random subset if we have too many nodes
        if len(nodes) > 50:
            # Identify "major" nodes based on degree
            node_degrees = {n: len(list(self.graph.neighbors(n))) for n in nodes}
            sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            major_nodes = [n for n, _ in sorted_nodes[:min(30, len(nodes)//2)]]
            
            # Add some random nodes for coverage
            remaining = set(nodes) - set(major_nodes)
            if remaining:
                major_nodes.extend(random.sample(list(remaining), 
                                              min(20, len(remaining))))
            nodes_to_process = major_nodes
        else:
            nodes_to_process = nodes
        
        # For each source node in our selected subset
        for source in nodes_to_process:
            # Add entry in routing table
            if source not in self.routing_table:
                self.routing_table[source] = {}
            
            # Use modified Dijkstra to compute paths to all destinations
            distance = {node: float('infinity') for node in nodes}
            predecessor = {node: None for node in nodes}
            distance[source] = 0
            
            # Priority queue for Dijkstra
            pq = [(0, source)]
            
            while pq:
                current_distance, current_node = heapq.heappop(pq)
                
                # If we've processed this node with a better distance, skip
                if current_distance > distance[current_node]:
                    continue
                
                # Process all neighbors
                for neighbor in self.graph.neighbors(current_node):
                    # Calculate composite path metric more efficiently
                    edge_distance = self.graph[current_node][neighbor].get('distance', 1.0)
                    congestion = self.graph[current_node][neighbor].get('congestion', 0.0)
                    pheromone = self.pheromone_table[current_node].get(neighbor, 0.1)
                    
                    # Simplified weight calculation - less computation
                    congestion_factor = 1 + congestion * 10
                    pheromone_factor = 1 / (pheromone + 0.1)
                    edge_weight = edge_distance * congestion_factor * pheromone_factor * 0.1
                    
                    # Compute new distance
                    new_distance = distance[current_node] + edge_weight
                    
                    # If we found a better path
                    if new_distance < distance[neighbor]:
                        distance[neighbor] = new_distance
                        predecessor[neighbor] = current_node
                        heapq.heappush(pq, (new_distance, neighbor))
            
            # Construct paths from predecessor map
            for destination in nodes:
                if destination == source:
                    continue
                    
                if predecessor[destination] is not None:
                    # Reconstruct path
                    path = []
                    current = destination
                    while current is not None:
                        path.append(current)
                        current = predecessor[current]
                    path.reverse()
                    
                    # Store in routing table
                    self.routing_table[source][destination] = path
    
    def _find_alternative_paths(self, source, destination, primary_path=None):
        """Find alternative paths for load balancing and fault tolerance."""
        if primary_path is None and source in self.routing_table and destination in self.routing_table[source]:
            primary_path = self.routing_table[source][destination]
        
        if primary_path is None or len(primary_path) < 3:
            return []
            
        alternative_paths = []
        
        # Check if congestion along primary path is severe enough to warrant alternatives
        avg_congestion = 0
        for i in range(len(primary_path) - 1):
            u, v = primary_path[i], primary_path[i+1]
            avg_congestion += self.graph[u][v].get('congestion', 0)
        
        avg_congestion /= (len(primary_path) - 1)
        
        # Skip alternative path computation if congestion is low - major performance improvement
        if avg_congestion < self.load_balancing_threshold:
            return []
        
        # Create a copy of the graph to remove edges from the primary path
        temp_graph = self.graph.copy()
        
        # Remove a critical edge from primary path to force an alternative
        if len(primary_path) > 2:
            # Find the edge with highest congestion
            max_congestion = 0
            max_idx = 1
            
            for i in range(1, len(primary_path) - 1):
                u, v = primary_path[i-1], primary_path[i]
                congestion = self.graph[u][v].get('congestion', 0)
                if congestion > max_congestion:
                    max_congestion = congestion
                    max_idx = i
            
            # Remove this edge
            u, v = primary_path[max_idx-1], primary_path[max_idx]
            if temp_graph.has_edge(u, v):
                temp_graph.remove_edge(u, v)
                
                # Try to find a path in the modified graph
                try:
                    def alternative_weight(src, dst, edge_data):
                        # Simplified weight function for better performance
                        congestion = edge_data.get('congestion', 0.0)
                        distance = edge_data.get('distance', 1.0)
                        return distance * (1 + congestion * 10)
                    
                    alt_path = nx.shortest_path(temp_graph, source, destination, weight=alternative_weight)
                    
                    # Only add if it's significantly different
                    if len(set(alt_path) - set(primary_path)) >= 2:
                        alternative_paths.append(alt_path)
                except nx.NetworkXNoPath:
                    pass
        
        return alternative_paths
    
    def find_path(self, source, destination, max_steps=100):
        """Find a path from source to destination using PAMR."""
        if source == destination:
            return [source], 0
        
        self.iteration += 1
        
        # Path key for caching
        path_key = (source, destination)
        
        # OPTIMIZATION: Check cache first - use cached path if still valid
        if self.use_advanced_cache and path_key in self.path_cache:
            cache_entry = self.path_cache[path_key]
            cache_age = self.iteration - cache_entry['iteration']
            
            # Use cache if it's recent enough and the congestion hasn't changed significantly
            if cache_age < self.cache_ttl:
                # Verify that no edge is extremely congested now
                path = cache_entry['path']
                max_new_congestion = 0
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    max_new_congestion = max(max_new_congestion, self.graph[u][v].get('congestion', 0))
                
                # If congestion is reasonable, use cached path
                if max_new_congestion < 0.8:
                    self.cache_hits += 1
                    
                    # Still update traffic on the path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        self.graph[u][v]['traffic'] = self.graph[u][v].get('traffic', 0) + 1
                        
                        # Update congestion based on capacity 
                        capacity = self.graph[u][v].get('capacity', 10)
                        new_congestion = min(0.95, self.graph[u][v]['traffic'] / capacity)
                        self.graph[u][v]['congestion'] = new_congestion
                    
                    return path, cache_entry['quality']
        
        self.cache_misses += 1
        
        # Check if it's time to update core paths - now less frequent
        if self.iteration % self.path_update_interval == 0:
            self._update_core_paths()
        
        # Check if path is in routing table
        if source in self.routing_table and destination in self.routing_table[source]:
            primary_path = self.routing_table[source][destination]
            
            # Only compute alternatives for severely congested paths
            all_paths = [primary_path]
            need_alternatives = False
            
            # Quick check - only examine a sample of edges if path is long
            edges_to_check = min(5, len(primary_path) - 1)
            sample_indices = random.sample(range(len(primary_path) - 1), edges_to_check) if len(primary_path) > edges_to_check else range(len(primary_path) - 1)
            
            for i in sample_indices:
                u, v = primary_path[i], primary_path[i+1]
                if self.graph[u][v].get('congestion', 0) > self.load_balancing_threshold * 1.5:
                    need_alternatives = True
                    break
            
            if need_alternatives:
                alternative_paths = self._find_alternative_paths(source, destination, primary_path)
                all_paths.extend(alternative_paths)
            
            # Find best path more efficiently
            best_path = primary_path
            best_quality = self._calculate_path_quality(primary_path)
            
            # Only evaluate alternatives if they exist
            for path in all_paths[1:]:
                path_quality = self._calculate_path_quality(path)
                if path_quality > best_quality:
                    best_path = path
                    best_quality = path_quality
            
            # Update path usage data
            self._update_pheromones(best_path, best_quality)
            
            # Update traffic on the path more efficiently
            self._update_path_traffic(best_path)
            
            # Cache this path with current iteration
            self.path_cache[path_key] = {
                'path': best_path, 
                'quality': best_quality,
                'iteration': self.iteration
            }
            
            return best_path, best_quality
        
        # OPTIMIZATION: For smaller networks, use direct shortest path
        if len(self.graph) < 50 and nx.has_path(self.graph, source, destination):
            # Use standard shortest path for small networks
            try:
                path = nx.shortest_path(self.graph, source, destination, weight='distance')
                path_quality = self._calculate_path_quality(path)
                
                # Update pheromones and traffic
                self._update_pheromones(path, path_quality)
                self._update_path_traffic(path)
                
                # Cache the path
                self.path_cache[path_key] = {
                    'path': path,
                    'quality': path_quality,
                    'iteration': self.iteration
                }
                
                return path, path_quality
            except nx.NetworkXNoPath:
                pass
        
        # If we don't have a path in the routing table, use simplified local path finding
        path = [source]
        visited = {source}
        current = source
        step_count = 0
        
        # Simplified local search with early termination
        while current != destination and step_count < min(30, max_steps):
            next_node = self._select_next_node(current, destination, visited)
            if next_node is None:
                break
                
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            step_count += 1
        
        if current == destination:
            # Calculate path quality
            path_quality = self._calculate_path_quality(path)
            
            # Update pheromones on this successful path
            self._update_pheromones(path, path_quality)
            
            # Update traffic on the path
            self._update_path_traffic(path)
            
            # Cache this path
            self.path_cache[path_key] = {
                'path': path, 
                'quality': path_quality,
                'iteration': self.iteration
            }
            
            return path, path_quality
        
        # No path found
        return path, -1
    
    def _update_path_traffic(self, path):
        """Update traffic and congestion along a path more efficiently."""
        if len(path) < 2:
            return
            
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # Update traffic counter
            self.graph[u][v]['traffic'] = self.graph[u][v].get('traffic', 0) + 1
            
            # Update congestion based on capacity
            capacity = self.graph[u][v].get('capacity', 10)
            new_congestion = min(0.95, self.graph[u][v]['traffic'] / capacity)
            self.graph[u][v]['congestion'] = new_congestion
            
            # More efficient congestion history update
            if len(self.congestion_history[(u, v)]) >= 20:
                self.congestion_history[(u, v)].pop(0)
            self.congestion_history[(u, v)].append(new_congestion)
    
    def _update_pheromones(self, path, quality):
        """Update pheromones along a path based on its quality."""
        if len(path) < 2:
            return
            
        # Scale the pheromone increase by path quality
        # Higher quality paths get more pheromone
        base_increase = quality * 0.5
        
        # Apply pheromone along the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Initialize if not exists
            if v not in self.pheromone_table[u]:
                self.pheromone_table[u][v] = 1.0
            
            # Apply pheromone increase
            self.pheromone_table[u][v] += base_increase
    
    def _evaporate_pheromones(self):
        """Evaporate pheromones across the network."""
        # OPTIMIZATION: Only evaporate pheromones on active paths
        for u in self.pheromone_table:
            for v in list(self.pheromone_table[u].keys()):
                # Slower evaporation rate
                self.pheromone_table[u][v] *= self.pheromone_evaporation
    
    def _select_next_node(self, current_node, destination, visited):
        """Select next node using PAMR algorithm for local routing."""
        try:
            neighbors = list(self.graph.neighbors(current_node))
        except (nx.NetworkXError, AttributeError):
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
            
            # Get edge attributes
            edge_data = self.graph[current_node][neighbor]
            pheromone = self.pheromone_table[current_node].get(neighbor, 0.1)
            distance = edge_data.get('distance', 1.0)
            congestion = edge_data.get('congestion', 0.0)
            bandwidth = edge_data.get('bandwidth', 1.0)
            
            # Use simplified congestion prediction
            congestion_history = self.congestion_history.get((current_node, neighbor), [congestion])
            if len(congestion_history) >= 2:
                # Simplified trend detection
                congestion_trend = congestion_history[-1] - congestion_history[0]
                predicted_congestion = max(0.01, min(0.99, congestion + congestion_trend))
            else:
                predicted_congestion = congestion
            
            # Simplified desirability calculation
            pheromone_factor = pheromone ** self.alpha
            distance_factor = (1.0 / distance) ** self.beta
            congestion_factor = (1.0 - predicted_congestion) ** self.gamma
            
            # OPTIMIZATION: For destination neighbor, give high priority
            if neighbor == destination:
                # If this neighbor is the destination, give it very high priority
                desirability = pheromone_factor * distance_factor * congestion_factor * 10.0
            else:
                bandwidth_factor = bandwidth  # Simplified
                
                # Simplified heuristic - use direct distance or hop count estimate
                try:
                    # Simple topological estimate
                    heuristic = 1.0 / (len(list(self.graph.neighbors(neighbor))) + 1)
                    desirability = pheromone_factor * distance_factor * congestion_factor * bandwidth_factor * heuristic
                except:
                    desirability = pheromone_factor * distance_factor * congestion_factor * bandwidth_factor
            
            probabilities.append(max(0.001, desirability))  # Ensure minimum probability
        
        # If no valid neighbors, return None
        if not valid_neighbors:
            return None
        
        # Select next node using weighted probability
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        # OPTIMIZATION: Sometimes pick the highest probability option deterministically
        if random.random() < 0.7:  # 70% of the time
            max_prob_idx = probabilities.index(max(probabilities))
            return valid_neighbors[max_prob_idx]
        else:
            # Otherwise use weighted random selection
            selected_idx = np.random.choice(range(len(valid_neighbors)), p=probabilities)
            return valid_neighbors[selected_idx]
    
    def _calculate_path_quality(self, path):
        """Calculate the quality of a path with improved balancing of factors."""
        if len(path) < 2:
            return 0
            
        # OPTIMIZATION: Faster path quality calculation
        total_distance = 0
        max_congestion = 0
        hop_count = len(path) - 1
        
        # Fixed parameters for computation
        min_bandwidth = float('inf')
        sum_congestion = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.graph[u][v]
            
            distance = edge_data.get('distance', 1.0)
            total_distance += distance
            
            congestion = edge_data.get('congestion', 0.0)
            sum_congestion += congestion
            max_congestion = max(max_congestion, congestion)
            
            bandwidth = edge_data.get('bandwidth', 1.0)
            min_bandwidth = min(min_bandwidth, bandwidth)
        
        # Simplified quality computation - much faster
        avg_congestion = sum_congestion / hop_count
        
        # Simplified quality calculation
        delay_factor = 1.0 / (total_distance * (1 + max_congestion * 2))
        congestion_factor = 1.0 / (1.0 + avg_congestion * 3)
        hop_factor = 1.0 / (1.0 + hop_count * 0.1)
        
        # Final quality combining all factors
        final_quality = (
            delay_factor * 0.6 +     # Higher weight to delay
            congestion_factor * 0.3 + # Medium weight to congestion
            hop_factor * 0.1         # Lower weight to hop count
        )
        
        # Store minimal path history
        path_key = (path[0], path[-1])
        if path_key not in self.path_history:
            self.path_history[path_key] = []
            
        # Store only essential metrics
        compact_path_data = {
            'iteration': self.iteration,
            'quality': final_quality,
            'congestion': max_congestion
        }
        
        self.path_history[path_key].append(compact_path_data)
        
        # Limit history size more aggressively
        if len(self.path_history[path_key]) > 5:
            self.path_history[path_key].pop(0)
            
        return final_quality
        
    def update_iteration(self):
        """Update the iteration counter and perform periodic maintenance."""
        self.iteration += 1
        
        # OPTIMIZATION: Perform maintenance less frequently
        # Evaporate pheromones less frequently
        if self.iteration % 10 == 0:
            self._evaporate_pheromones()
        
        # Major routing table update even less frequently
        if self.iteration % 40 == 0:
            self._update_core_paths()
        
        # Print cache statistics when needed
        if self.iteration % 100 == 0 and (self.cache_hits + self.cache_misses) > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            self.cache_hits = 0
            self.cache_misses = 0
        
        # Adaptive parameter adjustment based on network conditions
        if self.adapt_weights and self.iteration % 30 == 0:
            congestion_levels = []
            
            # Collect congestion data more efficiently
            edge_sample = random.sample(list(self.graph.edges()), min(30, len(self.graph.edges())))
            for u, v in edge_sample:
                congestion_levels.append(self.graph[u][v].get('congestion', 0))
            
            # Calculate network-wide congestion metrics
            if congestion_levels:
                avg_congestion = sum(congestion_levels) / len(congestion_levels)
                max_congestion = max(congestion_levels)
                
                # Adapt gamma (congestion sensitivity) based on congestion levels
                if max_congestion > 0.8:
                    # High congestion - increase congestion avoidance
                    self.gamma = min(12.0, self.gamma * 1.2)
                elif avg_congestion < 0.3:
                    # Low congestion - balance parameters more evenly
                    self.gamma = max(6.0, self.gamma * 0.9)
    
    def integrate_ml_prediction(self, traffic_predictor):
        """Integrate a machine learning model for traffic prediction."""
        self.traffic_predictor = traffic_predictor

    def predict_congestion(self, source, destination):
        """Predict congestion between source and destination using ML."""
        if hasattr(self, 'traffic_predictor') and self.traffic_predictor:
            try:
                features = self._extract_features(source, destination)
                return self.traffic_predictor.predict(features)
            except Exception as e:
                print(f"ML prediction failed: {e}")
        return None

    def _extract_features(self, source, destination):
        """Extract features for ML prediction."""
        # Example: Use distance, current congestion, and historical data
        features = []
        path = self.routing_table.get(source, {}).get(destination, [])
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.graph[u][v]
            features.append([
                edge_data.get('distance', 1.0),
                edge_data.get('congestion', 0.0),
                sum(self.congestion_history.get((u, v), [0])) / len(self.congestion_history.get((u, v), [1]))
            ])
        return np.array(features).flatten()

    def _update_core_paths_parallel(self):
        """Update core paths using parallel processing for scalability."""
        from concurrent.futures import ThreadPoolExecutor

        def process_node(source):
            self._update_paths_for_node(source)

        nodes = list(self.graph.nodes())
        with ThreadPoolExecutor() as executor:
            executor.map(process_node, nodes)

    def _update_paths_for_node(self, source):
        """Update paths for a single node."""
        distance = {node: float('infinity') for node in self.graph.nodes()}
        predecessor = {node: None for node in self.graph.nodes()}
        distance[source] = 0

        pq = [(0, source)]
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_distance > distance[current_node]:
                continue
            for neighbor in self.graph.neighbors(current_node):
                edge_data = self.graph[current_node][neighbor]
                edge_weight = edge_data.get('distance', 1.0) * (1 + edge_data.get('congestion', 0.0))
                new_distance = distance[current_node] + edge_weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    predecessor[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor))

        for destination in self.graph.nodes():
            if destination == source:
                continue
            path = []
            current = destination
            while current is not None:
                path.append(current)
                current = predecessor[current]
            path.reverse()
            self.routing_table[source][destination] = path

    def monitor_and_switch_paths(self):
        """Continuously monitor congestion and switch paths dynamically."""
        for source in self.routing_table:
            for destination in self.routing_table[source]:
                primary_path = self.routing_table[source][destination]
                if not primary_path:
                    continue

                # Calculate average congestion on the primary path
                avg_congestion = sum(
                    self.graph[primary_path[i]][primary_path[i + 1]].get('congestion', 0)
                    for i in range(len(primary_path) - 1)
                ) / (len(primary_path) - 1)

                # If congestion exceeds threshold, switch to an alternative path
                if avg_congestion > self.load_balancing_threshold:
                    alternative_paths = self._find_alternative_paths(source, destination, primary_path)
                    if alternative_paths:
                        best_alternative = max(
                            alternative_paths,
                            key=lambda path: self._calculate_path_quality(path)
                        )
                        self.routing_table[source][destination] = best_alternative

    def fast_reroute(self, source, destination):
        """Implement fast rerouting in case of congestion or failure."""
        primary_path = self.routing_table.get(source, {}).get(destination, [])
        if not primary_path:
            return None

        # Check for severe congestion or failure on the primary path
        for i in range(len(primary_path) - 1):
            u, v = primary_path[i], primary_path[i + 1]
            if self.graph[u][v].get('congestion', 0) > 0.9 or not self.graph.has_edge(u, v):
                # Trigger rerouting
                alternative_paths = self._find_alternative_paths(source, destination, primary_path)
                if alternative_paths:
                    best_alternative = max(
                        alternative_paths,
                        key=lambda path: self._calculate_path_quality(path)
                    )
                    self.routing_table[source][destination] = best_alternative
                    return best_alternative

        return primary_path

    def hierarchical_routing(self):
        """Divide the network into regions for hierarchical routing."""
        # Example: Divide nodes into clusters based on connectivity
        clusters = self._cluster_nodes()
        for cluster in clusters:
            self._update_core_paths_for_cluster(cluster)

    def _cluster_nodes(self):
        """Cluster nodes into regions based on connectivity."""
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(self.graph))
        return [list(community) for community in communities]

    def _update_core_paths_for_cluster(self, cluster):
        """Update core paths within a cluster."""
        for source in cluster:
            for destination in cluster:
                if source != destination:
                    self._update_paths_for_node(source)
