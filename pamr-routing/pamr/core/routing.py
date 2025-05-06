import random
import numpy as np
import networkx as nx
import heapq
from collections import defaultdict
import os
import pickle

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
        
        # Multi-path routing support
        self.multi_path_enabled = True        # Enable traffic distribution across multiple paths
        self.path_alternatives = {}           # Store alternative paths for each source-dest pair
        self.congestion_threshold = 0.4       # When to start considering alternative paths
        self.min_path_share = 0.1             # Minimum traffic share for any path
        
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
            
            # Priority
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
    
    def _find_alternative_paths(self, source, destination, primary_path=None, max_alternatives=3):
        """Find alternative paths for load balancing and fault tolerance.
        
        Args:
            source: Source node
            destination: Destination node
            primary_path: The primary path to avoid (if known)
            max_alternatives: Maximum number of alternative paths to find
            
        Returns:
            List of alternative paths
        """
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
        
        # ENHANCEMENT: Find multiple alternative paths with different strategies
        
        # Strategy 1: Remove critical edge
        # Find the edge with highest congestion
        congestion_levels = []
        for i in range(len(primary_path) - 1):
            u, v = primary_path[i], primary_path[i+1]
            congestion = self.graph[u][v].get('congestion', 0)
            congestion_levels.append((i, congestion))
        
        # Sort by congestion (highest first)
        congestion_levels.sort(key=lambda x: x[1], reverse=True)
        
        # Try removing different critical edges one by one
        for idx, _ in congestion_levels[:min(3, len(congestion_levels))]:
            u, v = primary_path[idx], primary_path[idx+1]
            
            # Create a copy of the graph to remove edges from the primary path
            temp_graph = self.graph.copy()
            
            # Remove this edge
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
                    is_different = len(set(alt_path) - set(primary_path)) >= 2
                    is_unique = all(alt_path != existing_path for existing_path in alternative_paths)
                    
                    if is_different and is_unique:
                        alternative_paths.append(alt_path)
                        
                        # Stop if we have enough alternatives
                        if len(alternative_paths) >= max_alternatives:
                            return alternative_paths
                except nx.NetworkXNoPath:
                    pass
        
        # Strategy 2: Try different metrics to find diverse paths
        if len(alternative_paths) < max_alternatives:
            # Create a copy of the graph with modified weights
            temp_graph = self.graph.copy()
            
            # Increase weights of edges in the primary path to discourage their use
            for i in range(len(primary_path) - 1):
                u, v = primary_path[i], primary_path[i+1]
                if temp_graph.has_edge(u, v):
                    # Make these edges very expensive but still usable in worst case
                    temp_graph[u][v]['temp_weight'] = temp_graph[u][v].get('distance', 1.0) * 10
            
            # Try to find a path using these modified weights
            try:
                def diverse_weight(src, dst, edge_data):
                    if 'temp_weight' in edge_data:
                        return edge_data['temp_weight']
                    return edge_data.get('distance', 1.0) * (1 + edge_data.get('congestion', 0.0) * 5)
                
                alt_path = nx.shortest_path(temp_graph, source, destination, weight=diverse_weight)
                
                # Only add if unique and different from primary
                is_different = len(set(alt_path) - set(primary_path)) >= 2
                is_unique = all(alt_path != existing_path for existing_path in alternative_paths)
                
                if is_different and is_unique:
                    alternative_paths.append(alt_path)
            except (nx.NetworkXNoPath, Exception):
                pass
        
        # Strategy 3: Use k-shortest paths algorithm to find more alternatives
        if len(alternative_paths) < max_alternatives:
            try:
                # Use a simple version of k-shortest paths by applying Yen's algorithm concept
                # First, get the shortest path
                temp_graph = self.graph.copy()
                
                # Discourage but don't forbid edges in primary and found alternative paths
                for path in [primary_path] + alternative_paths:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if temp_graph.has_edge(u, v):
                            temp_graph[u][v]['temp_penalty'] = temp_graph[u][v].get('temp_penalty', 1.0) * 5
                
                # Find more path options
                def penalized_weight(src, dst, edge_data):
                    penalty = edge_data.get('temp_penalty', 1.0)
                    congestion = edge_data.get('congestion', 0.0)
                    distance = edge_data.get('distance', 1.0)
                    return distance * (1 + congestion * 5) * penalty
                
                # Try to find more alternatives
                remaining_slots = max_alternatives - len(alternative_paths)
                for _ in range(remaining_slots):
                    try:
                        alt_path = nx.shortest_path(temp_graph, source, destination, weight=penalized_weight)
                        
                        # Check if this path is unique and different enough
                        is_different = len(set(alt_path) - set(primary_path)) >= 2
                        is_unique = all(alt_path != existing_path for existing_path in alternative_paths)
                        
                        if is_different and is_unique:
                            alternative_paths.append(alt_path)
                            
                            # Add penalty to edges in this path to encourage diversity in next iteration
                            for i in range(len(alt_path) - 1):
                                u, v = alt_path[i], alt_path[i+1]
                                if temp_graph.has_edge(u, v):
                                    temp_graph[u][v]['temp_penalty'] = temp_graph[u][v].get('temp_penalty', 1.0) * 3
                        else:
                            # If we found a duplicate, stop trying - we've exhausted the meaningful alternatives
                            break
                    except nx.NetworkXNoPath:
                        break
            except Exception:
                # If there's any error in this strategy, just continue with what we have
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
                if max_new_congestion < 0.6:  # Lower threshold to be more sensitive to congestion
                    self.cache_hits += 1
                    
                    # Still update traffic on the path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        self.graph[u][v]['traffic'] = self.graph[u][v].get('traffic', 0) + 1
                        
                        # Update congestion based on capacity - use same cap as elsewhere 
                        capacity = self.graph[u][v].get('capacity', 10)
                        new_congestion = min(0.8, self.graph[u][v]['traffic'] / capacity)
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
            
            # Update congestion based on capacity - lower cap to avoid reaching 0.95
            capacity = self.graph[u][v].get('capacity', 10)
            new_congestion = min(0.8, self.graph[u][v]['traffic'] / capacity)
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
                
                # Enforce minimum pheromone level to ensure exploration
                min_pheromone = 0.1  # Same as default in PheromoneManager
                if self.pheromone_table[u][v] < min_pheromone:
                    self.pheromone_table[u][v] = min_pheromone
    
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
        
        # Apply traffic decay every iteration to simulate packets leaving the network
        self.decay_traffic()
        
        # Check for congestion and switch paths more frequently
        if self.iteration % 5 == 0:
            self.monitor_and_switch_paths()
        
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

                # Lower threshold to switch paths more aggressively
                if avg_congestion > 0.2:  # Changed from 0.25
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

    def decay_traffic(self):
        """Decay traffic over time to simulate packets leaving the network."""
        for u, v in self.graph.edges():
            # Traffic decays exponentially over time
            current_traffic = self.graph[u][v].get('traffic', 0)
            self.graph[u][v]['traffic'] = max(0, current_traffic * 0.9)
            
            # Recalculate congestion based on decayed traffic
            capacity = self.graph[u][v].get('capacity', 10)
            self.graph[u][v]['congestion'] = self.graph[u][v]['traffic'] / capacity
            
            # Update congestion history
            if len(self.congestion_history[(u, v)]) >= 20:
                self.congestion_history[(u, v)].pop(0)
            self.congestion_history[(u, v)].append(self.graph[u][v]['congestion'])
    
class AdvancedMultiPathRouter(PAMRRouter):
    """
    Advanced Multi-Path Router that extends PAMR with state-of-the-art
    path discovery and traffic engineering capabilities.
    """
    
    def __init__(self, graph, alpha=2.0, beta=3.0, gamma=8.0, adapt_weights=True):
        """Initialize with enhanced path discovery and traffic distribution capabilities"""
        super().__init__(graph, alpha, beta, gamma, adapt_weights)
        
        # Enhanced multi-path parameters
        self.max_paths_to_discover = 10       # Find up to 10 diverse paths
        self.max_paths_to_use = 5             # Use up to 5 paths simultaneously
        self.path_diversity_threshold = 0.5   # Path diversity requirement (0-1)
        self.congestion_prediction_window = 5 # Look ahead window for congestion prediction
        self.specialized_path_types = {       # Different path specializations
            'lowest_latency': 0.3,            # Weight for specialized paths
            'highest_bandwidth': 0.3,
            'most_reliable': 0.4
        }
        
        # Path diversity matrix - tracks how different paths are from each other
        self.path_diversity_matrix = {}
        
        # Path performance history
        self.path_performance_history = defaultdict(lambda: defaultdict(list))
        
        # Traffic class definitions
        self.traffic_classes = {
            'standard': {'weight': 1.0, 'congestion_sensitivity': 1.0},
            'latency_sensitive': {'weight': 1.5, 'congestion_sensitivity': 2.0},
            'bandwidth_heavy': {'weight': 1.2, 'congestion_sensitivity': 0.8},
            'high_priority': {'weight': 2.0, 'congestion_sensitivity': 1.5}
        }
        
        # Initialize ML-based prediction if available
        self.use_ml_prediction = True
        self.prediction_enabled = False
        self.prediction_model = None
        self.path_features_history = {}
        
        # Try to load a pre-trained model if it exists
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     '../../../pamr_traffic_predictor.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.prediction_model = pickle.load(f)
                    self.prediction_enabled = True
                    print("Loaded ML prediction model for advanced routing")
        except:
            # If model loading fails, continue without ML prediction
            pass
    
    def discover_diverse_paths(self, source, destination, max_paths=None, traffic_class='standard'):
        """
        Discover multiple diverse paths between source and destination using
        multiple advanced algorithms.
        
        Args:
            source: Source node
            destination: Destination node
            max_paths: Maximum number of paths to discover
            traffic_class: Type of traffic for specialized path discovery
            
        Returns:
            List of diverse paths from source to destination
        """
        if max_paths is None:
            max_paths = self.max_paths_to_discover
        
        diverse_paths = []
        path_key = (source, destination)
        
        # 1. First try direct neighbor paths - these are often overlooked by algorithms
        # Simplest paths: Check if destination is a direct neighbor of source or can be
        # reached through one intermediate node
        direct_neighbors = list(self.graph.neighbors(source))
        
        # 1a. Check for direct path: source -> destination
        if destination in direct_neighbors:
            direct_path = [source, destination]
            diverse_paths.append(direct_path)

        # 1b. Check for 2-hop paths: source -> intermediary -> destination
        for intermediary in direct_neighbors:
            if intermediary != destination and destination in self.graph.neighbors(intermediary):
                two_hop_path = [source, intermediary, destination]
                if two_hop_path not in diverse_paths:
                    diverse_paths.append(two_hop_path)
        
        # 2. Get the primary path from routing table or find one
        primary_path = self.routing_table.get(source, {}).get(destination, None)
        if primary_path is None:
            try:
                primary_path, _ = self.find_path(source, destination)
                if len(primary_path) < 2 or primary_path[-1] != destination:
                    # Could not find a valid path
                    if not diverse_paths:  # Only return empty if we have no paths at all
                        return []
            except:
                if not diverse_paths:  # Only return empty if we have no paths at all
                    return []
        
        if primary_path and primary_path not in diverse_paths:
            diverse_paths.append(primary_path)
        
        # 3. Find shortest path with default distance metric as another baseline
        try:
            shortest_path = nx.shortest_path(self.graph, source, destination, weight='distance')
            if shortest_path not in diverse_paths:
                diverse_paths.append(shortest_path)
        except:
            pass
        
        # 4. Create a copy of the graph for path discovery
        G = self.graph.copy()
        
        # Find paths with different strategies and lower the diversity threshold to accept more paths
        original_threshold = self.path_diversity_threshold
        self.path_diversity_threshold = 0.3  # Lower threshold to accept more diverse paths
        
        try:
            # Try all specialized path finding methods
            # First, get paths optimized for different metrics
            latency_paths = self._find_specialized_paths(G, source, destination, 3, 'latency_sensitive')
            for path in latency_paths:
                if self._is_path_sufficiently_diverse(path, diverse_paths) and path not in diverse_paths:
                    diverse_paths.append(path)
            
            # Try genetic algorithm inspired paths
            genetic_paths = self._find_paths_via_genetic_algorithm(G, source, destination, 3, traffic_class)
            for path in genetic_paths:
                if self._is_path_sufficiently_diverse(path, diverse_paths) and path not in diverse_paths:
                    diverse_paths.append(path)
                    
            # Try node disjoint paths
            disjoint_paths = self._find_node_disjoint_paths(G, source, destination, 3, traffic_class)
            for path in disjoint_paths:
                if path not in diverse_paths:
                    diverse_paths.append(path)
        except Exception as e:
            # Continue even if some methods fail
            pass
            
        # Use edge exclusion to find a very different path
        if len(diverse_paths) > 0 and len(diverse_paths) < max_paths:
            try:
                # Use the first path as a base and exclude its edges
                base_path = diverse_paths[0]
                excluded_G = G.copy()
                
                # Remove all edges from the base path
                for i in range(len(base_path) - 1):
                    u, v = base_path[i], base_path[i+1]
                    if excluded_G.has_edge(u, v):
                        excluded_G.remove_edge(u, v)
                
                # Try to find a completely different path
                if nx.has_path(excluded_G, source, destination):
                    alternate_path = nx.shortest_path(excluded_G, source, destination)
                    if alternate_path not in diverse_paths:
                        diverse_paths.append(alternate_path)
            except:
                pass
                
        # Reset the diversity threshold
        self.path_diversity_threshold = original_threshold
                
        # If we still don't have enough paths, try a completely different approach
        if len(diverse_paths) < max_paths:
            try:
                # Use depth-first search to find more paths
                def dfs_paths(graph, start, goal, path=None, visited=None, max_depth=10):
                    if path is None:
                        path = [start]
                    if visited is None:
                        visited = set([start])
                    if start == goal:
                        return [path]
                    if len(path) > max_depth:
                        return []
                    paths = []
                    for neighbor in graph.neighbors(start):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            extended_paths = dfs_paths(graph, neighbor, goal, path + [neighbor], visited, max_depth)
                            paths.extend(extended_paths)
                            visited.remove(neighbor)
                    return paths
                
                # Find up to 3 additional paths with DFS (limit depth to avoid very long paths)
                # Find more paths with max depth relative to existing path lengths
                max_dfs_depth = max(len(p) for p in diverse_paths) + 3 if diverse_paths else 10
                dfs_additional_paths = dfs_paths(G, source, destination, max_depth=max_dfs_depth)[:3]
                
                # Add any new paths we found
                for path in dfs_additional_paths:
                    if path not in diverse_paths:
                        diverse_paths.append(path)
            except:
                pass
        
        # Update diversity matrix before returning
        self._update_path_diversity_matrix(diverse_paths)
        
        # Cache these paths for future use
        self.path_alternatives[path_key] = diverse_paths[1:] if len(diverse_paths) > 1 else []
        
        # Return up to max_paths diverse paths
        return diverse_paths[:max_paths]
    
    def _find_k_shortest_paths(self, G, source, destination, k, traffic_class):
        """Find k shortest paths using Yen's algorithm"""
        paths = []
        
        # Use different weight functions based on traffic class
        if traffic_class == 'latency_sensitive':
            def weight_fn(u, v, data):
                distance = data.get('distance', 1.0)
                congestion = data.get('congestion', 0) # Fixed missing parenthesis
                # Heavily penalize congestion for latency sensitive traffic
                return distance * (1 + 3.0 * congestion)
        elif traffic_class == 'bandwidth_heavy':
            def weight_fn(u, v, data):
                distance = data.get('distance', 1.0)
                bandwidth = data.get('bandwidth', 1.0)
                # Prefer high bandwidth paths
                return distance / (0.1 + bandwidth)
        else:
            # Standard weight function
            def weight_fn(u, v, data):
                distance = data.get('distance', 1.0)
                congestion = data.get('congestion', 0.0)
                return distance * (1 + congestion)
        
        # Implementation of a simplified k-shortest paths algorithm
        # First find the shortest path
        try:
            shortest_path = nx.shortest_path(G, source, destination, weight=weight_fn)
            paths.append(shortest_path)
        except (nx.NetworkXNoPath, Exception):
            return paths
            
        # Find k-1 additional shortest paths
        for i in range(1, k):
            if not paths:
                break
                
            # Create a copy of the graph
            temp_G = G.copy()
            
            # Find a new path by temporarily removing edges from previous paths
            for prev_path in paths:
                # Instead of removing edges, apply high penalties
                for j in range(len(prev_path)-1):
                    u, v = prev_path[j], prev_path[j+1]
                    if temp_G.has_edge(u, v):
                        # Apply a high penalty factor to this edge
                        penalty_factor = 5.0 + (i * 2.0)  # Increase penalty with each iteration
                        
                        # Store original attributes
                        for key, value in G[u][v].items():
                            if key != 'temp_penalty':
                                temp_G[u][v][key] = value
                                
                        # Add penalty
                        temp_G[u][v]['temp_penalty'] = penalty_factor
            
            # Define a weight function that considers penalties
            def penalty_weight_fn(u, v, data):
                base_weight = weight_fn(u, v, data)
                penalty = data.get('temp_penalty', 1.0)
                return base_weight * penalty
            
            # Try to find another path
            try:
                new_path = nx.shortest_path(temp_G, source, destination, weight=penalty_weight_fn)
                
                # Only add if it's different from existing paths
                if all(new_path != p for p in paths):
                    paths.append(new_path)
            except (nx.NetworkXNoPath, Exception):
                break
                
        return paths
    
    def _find_edge_disjoint_paths(self, G, source, destination, k, traffic_class):
        """Find edge-disjoint paths between source and destination"""
        try:
            # Convert to directed flow network if needed
            if not isinstance(G, nx.DiGraph):
                flow_G = G.to_directed()
            else:
                flow_G = G.copy()
                
            # Set capacity for flow calculation
            for u, v in flow_G.edges():
                # Use edge capacity or default to 1
                capacity = flow_G[u][v].get('capacity', 10.0)
                flow_G[u][v]['capacity'] = capacity
            
            # Find edge-disjoint paths using maximum flow
            paths = []
            remaining_k = k
            
            while remaining_k > 0:
                try:
                    # Try to find an augmenting path
                    path = nx.shortest_path(flow_G, source, destination)
                    
                    # Get the minimum capacity along this path
                    min_capacity = float('inf')
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        min_capacity = min(min_capacity, flow_G[u][v].get('capacity', 0))
                    
                    # If path has capacity, add it
                    if min_capacity > 0:
                        paths.append(path)
                        
                        # Reduce capacity along this path
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            flow_G[u][v]['capacity'] -= min_capacity
                            # Remove edge if capacity is depleted
                            if flow_G[u][v]['capacity'] <= 0:
                                flow_G.remove_edge(u, v)
                    else:
                        break
                        
                    remaining_k -= 1
                except nx.NetworkXNoPath:
                    break
                    
            return paths
        except Exception:
            return []
    
    def _find_node_disjoint_paths(self, G, source, destination, k, traffic_class):
        """Find node-disjoint paths between source and destination"""
        try:
            # Implementation using network flow principles
            # Create a node-expanded graph where each node is split
            # into an in-node and out-node
            
            # This is advanced but a simplified version can be:
            paths = []
            G_copy = G.copy()
            
            for _ in range(k):
                try:
                    if nx.has_path(G_copy, source, destination):
                        # Find shortest path
                        path = nx.shortest_path(G_copy, source, destination)
                        paths.append(path)
                        
                        # Remove internal nodes of this path (not source/destination)
                        for node in path[1:-1]:
                            if G_copy.has_node(node):
                                G_copy.remove_node(node)
                    else:
                        break
                except:
                    break
                    
            return paths
        except Exception:
            return []
    
    def _find_paths_via_genetic_algorithm(self, G, source, destination, k, traffic_class):
        """Use a simplified genetic algorithm approach to find diverse paths"""
        # This would be a simplified implementation of a genetic algorithm
        # for path discovery. A full implementation would be much more complex.
        
        # For now, use a randomized approach to simulate genetic diversity
        paths = []
        
        try:
            # First get one shortest path as a seed
            shortest_path = nx.shortest_path(G, source, destination)
            paths.append(shortest_path)
            
            # Create variations by randomly perturbing edge weights
            for i in range(k-1):
                # Create a copy with randomly perturbed weights
                G_copy = G.copy()
                
                # Randomly modify edge weights
                for u, v in G_copy.edges():
                    # Get original distance
                    original_distance = G_copy[u][v].get('distance', 1.0) # Fixed missing parenthesis
                    
                    # Apply random perturbation (±30%)
                    perturbation = 0.7 + (random.random() * 0.6)  # Between 0.7 and 1.3
                    G_copy[u][v]['perturbed_distance'] = original_distance * perturbation
                
                # Define weight function using perturbed weights
                def perturbed_weight(u, v, data):
                    return data.get('perturbed_distance', data.get('distance', 1.0))
                
                # Try to find path with perturbed weights
                try:
                    perturbed_path = nx.shortest_path(G_copy, source, destination, weight=perturbed_weight)
                    
                    # Check if this path is different from existing ones
                    if all(not self._paths_are_similar(perturbed_path, p) for p in paths):
                        paths.append(perturbed_path)
                except (nx.NetworkXNoPath, Exception):
                    # Catch potential errors during path finding
                    continue
                    
            return paths
        except Exception:
            # Catch any other exceptions during the genetic algorithm process
            return paths # Return whatever paths were found so far
    
    def _find_specialized_paths(self, G, source, destination, k, traffic_class):
        """Find paths specialized for different optimization criteria"""
        paths = []
        
        try:
            # 1. Find path optimized for lowest latency
            try:
                def latency_weight(u, v, data):
                    return data.get('distance', 1.0)
                    
                latency_path = nx.shortest_path(G, source, destination, weight=latency_weight)
                paths.append(latency_path)
            except:
                pass
                
            # 2. Find path optimized for highest bandwidth
            try:
                def bandwidth_weight(u, v, data):
                    # Inverse of bandwidth - lower values preferred
                    bw = data.get('bandwidth', 1.0)
                    return 1.0 / max(0.1, bw)
                    
                bandwidth_path = nx.shortest_path(G, source, destination, weight=bandwidth_weight)
                if all(bandwidth_path != p for p in paths):
                    paths.append(bandwidth_path)
            except:
                pass
                
            # 3. Find path optimized for reliability/minimal congestion
            try:
                def reliability_weight(u, v, data):
                    congestion = data.get('congestion', 0.0)
                    # Higher weight for congested links
                    return 1.0 + (congestion * 10.0)
                    
                reliability_path = nx.shortest_path(G, source, destination, weight=reliability_weight)
                if all(reliability_path != p for p in paths):
                    paths.append(reliability_path)
            except:
                pass
                
            # 4. Find path optimized for the specific traffic class
            if traffic_class in self.traffic_classes:
                try:
                    class_sensitivity = self.traffic_classes[traffic_class]['congestion_sensitivity']
                    
                    def class_weight(u, v, data):
                        distance = data.get('distance', 1.0)
                        congestion = data.get('congestion', 0.0)
                        return distance * (1 + congestion * class_sensitivity)
                        
                    class_path = nx.shortest_path(G, source, destination, weight=class_weight)
                    if all(class_path != p for p in paths):
                        paths.append(class_path)
                except:
                    pass
            
            return paths[:k]
        except Exception:
            return []
    
    def _is_path_sufficiently_diverse(self, new_path, existing_paths, threshold=None):
        """
        Check if a path is sufficiently diverse from existing paths
        
        Args:
            new_path: The new path to check
            existing_paths: List of existing paths
            threshold: Diversity threshold (0-1)
            
        Returns:
            True if the path is sufficiently diverse, False otherwise
        """
        if threshold is None:
            threshold = self.path_diversity_threshold
            
        # If there are no existing paths, the new path is diverse
        if not existing_paths:
            return True
            
        # Check if the path is identical to any existing path
        if any(new_path == path for path in existing_paths):
            return False
            
        # Calculate similarity with each existing path
        for path in existing_paths:
            similarity = self._calculate_path_similarity(new_path, path)
            if similarity > (1.0 - threshold):
                # Too similar to an existing path
                return False
                
        return True
    
    def _calculate_path_similarity(self, path1, path2):
        """
        Calculate the similarity between two paths
        
        Returns a value between 0 (completely different) and 1 (identical)
        """
        # Convert paths to sets for easier comparison
        edges1 = set((path1[i], path1[i+1]) for i in range(len(path1)-1))
        edges2 = set((path2[i], path2[i+1]) for i in range(len(path2)-1))
        
        # If both paths are empty, they're identical
        if not edges1 and not edges2:
            return 1.0
            
        # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        intersection = len(edges1.intersection(edges2))
        union = len(edges1.union(edges2))
        
        return intersection / union if union > 0 else 0.0
    
    def _paths_are_similar(self, path1, path2, threshold=0.7):
        """Check if two paths are similar above a threshold"""
        return self._calculate_path_similarity(path1, path2) >= threshold
    
    def _update_path_diversity_matrix(self, paths):
        """Update the diversity matrix for a set of paths"""
        n = len(paths)
        
        # Create a matrix key for this path set
        path_set_key = tuple(tuple(path) for path in paths)
        
        # Initialize matrix if needed
        if path_set_key not in self.path_diversity_matrix:
            self.path_diversity_matrix[path_set_key] = np.zeros((n, n))
            
        # Calculate diversity between each pair of paths
        for i in range(n):
            for j in range(i+1, n):
                similarity = self._calculate_path_similarity(paths[i], paths[j])
                diversity = 1.0 - similarity
                
                # Update the matrix symmetrically
                self.path_diversity_matrix[path_set_key][i, j] = diversity
                self.path_diversity_matrix[path_set_key][j, i] = diversity
    
    def distribute_traffic(self, paths, path_metrics):
        """
        Distribute traffic across multiple paths using advanced criteria
        
        Args:
            paths: List of paths
            path_metrics: List of dictionaries with path metrics
            
        Returns:
            List of (path, ratio) tuples
        """
        if not paths:
            return []
            
        if len(paths) == 1:
            return [(paths[0], 1.0)]
            
        # Get path metrics
        qualities = [m['quality'] for m in path_metrics]
        congestions = [m.get('max_congestion', m.get('congestion', 0)) for m in path_metrics]
        
        # CRITICAL FIX: Use a much more aggressive congestion-avoidance model
        # Instead of just using path quality, we heavily prioritize paths with low congestion
        # This creates a much more dynamic traffic distribution that shifts rapidly
        # as congestion levels change
        
        # Calculate inverse congestion (higher value = less congested)
        inverse_congestion = []
        for c in congestions:
            # Exponential penalty for congestion
            # This makes the algorithm extremely sensitive to congestion changes
            penalty = (c / 0.8) ** 3  # Cubic penalty for sharper response
            inverse_value = 1.0 / (0.05 + penalty)  # Avoid division by zero
            inverse_congestion.append(inverse_value)
            
        # Calculate quality-to-congestion ratio (higher is better)
        # This balances quality and congestion avoidance
        ratio_metrics = []
        for i in range(len(paths)):
            # Use a weighted combination (70% congestion, 30% quality)
            # This heavily favors congestion avoidance
            congestion_weight = 0.7
            quality_weight = 0.3
            
            # Normalize values first
            norm_inverse_congestion = inverse_congestion[i] / sum(inverse_congestion) if sum(inverse_congestion) > 0 else 0
            norm_quality = qualities[i] / sum(qualities) if sum(qualities) > 0 else 0
            
            # Combined metric
            combined_score = (congestion_weight * norm_inverse_congestion) + (quality_weight * norm_quality)
            ratio_metrics.append(combined_score)
        
        # Initial traffic distribution based on the combined metrics
        total_metric = sum(ratio_metrics)
        if total_metric > 0:
            base_ratios = [m / total_metric for m in ratio_metrics]
        else:
            # Equal distribution if metrics sum to zero
            base_ratios = [1.0 / len(paths)] * len(paths)
        
        # Apply minimum share limit to ensure all paths get some traffic
        for i in range(len(base_ratios)):
            if base_ratios[i] < self.min_path_share:
                base_ratios[i] = self.min_path_share
        
        # Normalize after applying minimum share
        total = sum(base_ratios)
        final_ratios = [r / total for r in base_ratios]
        
        # Create dynamic distribution that shifts across paths
        # to visualize the multi-path nature of the algorithm
        # Add slight randomization to make the effect more visible
        if len(final_ratios) >= 2:
            # Randomly perturb ratios slightly (±10%) to show dynamism
            # This is for demonstration purposes to make path switching more visible
            for i in range(len(final_ratios)):
                # More congested paths get more randomness to encourage switching
                congestion_factor = min(1.0, congestions[i] / 0.4)  # Scale by congestion
                max_perturbation = 0.1 * congestion_factor
                perturbation = 1.0 + random.uniform(-max_perturbation, max_perturbation)
                final_ratios[i] *= perturbation
                
            # Re-normalize after perturbation
            total = sum(final_ratios)
            final_ratios = [r / total for r in final_ratios]
        
        # Return path distribution
        return [(paths[i], final_ratios[i]) for i in range(len(paths))]
    
    def predict_future_congestion(self, path, prediction_window=None):
        """
        Predict future congestion on a path using historical data or ML
        
        Args:
            path: The path to predict congestion for
            prediction_window: How far into the future to predict
            
        Returns:
            Predicted congestion value between 0-1
        """
        if prediction_window is None:
            prediction_window = self.congestion_prediction_window
            
        # If path is too short, return zero congestion
        if len(path) < 2:
            return 0.0
            
        # If ML prediction is enabled and model is available
        if self.use_ml_prediction and self.prediction_enabled and self.prediction_model:
            try:
                # Extract features for prediction
                features = self._extract_path_features(path)
                
                # Make prediction
                predicted_congestion = self.prediction_model.predict([features])[0]
                
                # Ensure prediction is in valid range
                return max(0.0, min(1.0, predicted_congestion))
            except Exception:
                # Fall back to simple prediction if ML fails
                pass
                
        # Simple linear prediction based on historical data
        avg_congestion = 0.0
        count = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Get congestion history for this edge
            history = self.congestion_history.get((u, v), [])
            
            if len(history) > 1:
                # Calculate trend over last few values
                recent_values = history[-min(10, len(history)):]
                if len(recent_values) >= 2:
                    # Simple linear extrapolation
                    trend = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
                    
                    # Predict future value
                    current_congestion = recent_values[-1]
                    predicted_congestion = current_congestion + (trend * prediction_window)
                    
                    # Ensure prediction is within bounds
                    predicted_congestion = max(0.0, min(0.8, predicted_congestion))
                    
                    avg_congestion += predicted_congestion
                    count += 1
                else:
                    # Not enough history, use current value
                    avg_congestion += history[-1]
                    count += 1
            elif len(history) == 1:
                # Only one value, use it as is
                avg_congestion += history[0]
                count += 1
            else:
                # No history, assume zero congestion
                pass
                
        if count > 0:
            return avg_congestion / count
        else:
            return 0.0
    
    def _extract_path_features(self, path):
        """Extract features for ML prediction"""
        # This would extract relevant features for the ML model
        features = []
        
        # 1. Path length
        features.append(len(path) - 1)
        
        # 2. Average distance
        total_distance = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_distance += self.graph[u][v].get('distance', 1.0)
        avg_distance = total_distance / (len(path) - 1) if len(path) > 1 else 0
        features.append(avg_distance)
        
        # 3. Current congestion (max and average)
        max_congestion = 0
        total_congestion = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            congestion = self.graph[u][v].get('congestion', 0.0)
            max_congestion = max(max_congestion, congestion)
            total_congestion += congestion
        avg_congestion = total_congestion / (len(path) - 1) if len(path) > 1 else 0
        features.append(max_congestion)
        features.append(avg_congestion)
        
        # 4. Average bandwidth
        total_bandwidth = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_bandwidth += self.graph[u][v].get('bandwidth', 1.0)
        avg_bandwidth = total_bandwidth / (len(path) - 1) if len(path) > 1 else 0
        features.append(avg_bandwidth)
        
        # 5. Pheromone levels
        total_pheromone = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_pheromone += self.pheromone_table[u].get(v, 0.0)
        avg_pheromone = total_pheromone / (len(path) - 1) if len(path) > 1 else 0
        features.append(avg_pheromone)
        
        # 6. Recent congestion changes (if available)
        avg_congestion_change = 0
        count = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            history = self.congestion_history.get((u, v), [])
            if len(history) >= 2:
                change = history[-1] - history[0]
                avg_congestion_change += change
                count += 1
        avg_congestion_change = avg_congestion_change / count if count > 0 else 0
        features.append(avg_congestion_change)
        
        return features
    
    def update_path_performance_history(self, path, metrics):
        """Update the performance history for a path"""
        path_key = tuple(path)
        
        # Update quality history
        if 'quality' in metrics:
            self.path_performance_history[path_key]['recent_quality'].append(metrics['quality'])
            if len(self.path_performance_history[path_key]['recent_quality']) > 20:
                self.path_performance_history[path_key]['recent_quality'].pop(0)
                
        # Update congestion history
        if 'congestion' in metrics:
            self.path_performance_history[path_key]['recent_congestion'].append(metrics['congestion'])
            if len(self.path_performance_history[path_key]['recent_congestion']) > 20:
                self.path_performance_history[path_key]['recent_congestion'].pop(0)
    
    def get_multi_path_routing(self, source, destination, traffic_class='standard'):
        """
        Get multiple paths for routing with smart traffic distribution
        
        Args:
            source: Source node
            destination: Destination node
            traffic_class: Type of traffic for specialized path selection
            
        Returns:
            List of (path, ratio) tuples for traffic distribution
        """
        # Find diverse paths between source and destination
        diverse_paths = self.discover_diverse_paths(source, destination, 
                                                  self.max_paths_to_discover,
                                                  traffic_class)
        
        # If no paths found, return empty list
        if not diverse_paths:
            return []
            
        # If only one path found, use it for all traffic
        if len(diverse_paths) == 1:
            return [(diverse_paths[0], 1.0)]
            
        # CRITICAL FIX: Calculate path metrics with CURRENT congestion values
        # This is essential for dynamic path quality assessment
        path_metrics = []
        
        for path in diverse_paths:
            # IMPORTANT: Recalculate quality based on CURRENT congestion
            # This ensures quality reflects the real-time network state
            quality = self._calculate_path_quality(path)
            
            # Calculate average and max congestion along path
            avg_congestion = 0
            max_congestion = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_congestion = self.graph[u][v].get('congestion', 0)
                avg_congestion += edge_congestion
                max_congestion = max(max_congestion, edge_congestion)
            
            # Avoid division by zero
            path_length = max(1, len(path) - 1)
            avg_congestion /= path_length
            
            # Predict future congestion (simplified)
            future_congestion = min(0.8, avg_congestion * 1.2)
            
            # Store all metrics for decision making
            path_metrics.append({
                'path': path,
                'quality': quality,
                'congestion': avg_congestion,
                'max_congestion': max_congestion,
                'future_congestion': future_congestion
            })
        
        # IMPROVEMENT: Sort by combined score that balances quality with congestion avoidance
        # This creates more dynamic path selection as congestion increases
        for metric in path_metrics:
            # Calculate a combined score that balances quality and congestion
            # Heavily penalize congested paths to encourage early path switching
            congestion_penalty = (metric['max_congestion'] / 0.8) ** 2  # Exponential penalty
            congestion_factor = 1.0 - min(0.9, congestion_penalty)  # Cap at 90% reduction
            
            # Calculate combined metric
            metric['combined_score'] = metric['quality'] * congestion_factor
        
        # Sort by combined score (higher is better)
        path_metrics.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Select top paths for use (limit to max_paths_to_use)
        selected_paths = [m['path'] for m in path_metrics[:self.max_paths_to_use]]
        selected_metrics = path_metrics[:self.max_paths_to_use]
        
        # ENHANCED TRAFFIC DISTRIBUTION: More responsive to congestion changes
        # This ensures traffic shifts away from congested paths more aggressively
        distribution = self.distribute_traffic(selected_paths, selected_metrics)
        
        # Update path performance history
        for metric in selected_metrics:
            self.update_path_performance_history(metric['path'], metric)
            
        return distribution
