import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from collections import defaultdict

class GridSimulator:
    """
    Simulates a city grid with connected intersections, managing traffic flow
    and inter-intersection communication.
    """
    
    def __init__(self, rows=2, cols=2, distance_between=200):
        """
        Initialize a city grid.
        
        Args:
            rows: Number of intersections in each row
            cols: Number of intersections in each column
            distance_between: Distance between adjacent intersections (meters)
        """
        self.rows = rows
        self.cols = cols
        self.distance = distance_between
        
        # Create a graph representing the city grid
        self.graph = self._create_grid_graph()
        
        # Dictionary to track traffic at each intersection
        self.intersection_traffic = {}
        self.initialize_traffic()
        
        # Dictionary to track vehicles moving between intersections
        self.vehicles_in_transit = {}
        
        # Parameters for traffic flow simulation
        self.speed_kmh = 40  # Average vehicle speed in km/h
        self.speed_mps = self.speed_kmh * 1000 / 3600  # Convert to meters/second
        
        # Direction mapping for clarity
        # IMPORTANT: These directions represent FROM where traffic is coming
        # For example, 'north' means traffic coming FROM the north INTO the intersection
        self.directions = {
            'north': (0, 1),   # FROM north TO intersection
            'south': (0, -1),  # FROM south TO intersection
            'east': (1, 0),    # FROM east TO intersection
            'west': (-1, 0)    # FROM west TO intersection
        }
        
        # Map directions to opposite directions (for receiving traffic)
        self.opposite_direction = {
            'north': 'south',  # If traffic comes FROM north, it goes TO south
            'south': 'north',  # If traffic comes FROM south, it goes TO north
            'east': 'west',    # If traffic comes FROM east, it goes TO west
            'west': 'east'     # If traffic comes FROM west, it goes TO east
        }
        
        # Turn probabilities - how likely a vehicle is to go each direction
        self.turn_probabilities = {
            'straight': 0.7,  # 70% go straight
            'left': 0.15,     # 15% turn left
            'right': 0.15     # 15% turn right
        }
        
        # Direction changes for turns
        # When traffic comes FROM a direction and makes a turn, where does it go?
        self.turn_mapping = {
            # Traffic coming FROM north
            'north': {
                'left': 'west',      # Left turn goes TO west
                'straight': 'south', # Straight goes TO south
                'right': 'east'      # Right turn goes TO east
            },
            # Traffic coming FROM south
            'south': {
                'left': 'east',      # Left turn goes TO east
                'straight': 'north', # Straight goes TO north
                'right': 'west'      # Right turn goes TO west
            },
            # Traffic coming FROM east
            'east': {
                'left': 'north',     # Left turn goes TO north
                'straight': 'west',  # Straight goes TO west
                'right': 'south'     # Right turn goes TO south
            },
            # Traffic coming FROM west
            'west': {
                'left': 'south',     # Left turn goes TO south
                'straight': 'east',  # Straight goes TO east
                'right': 'north'     # Right turn goes TO north
            }
        }
        
        # Track historical traffic patterns
        self.historical_traffic = defaultdict(lambda: defaultdict(list))
        
        # NEW: Track inter-intersection influence
        self.neighbor_influence = {}
        self.congestion_spillover = {}
        
        # NEW: Initialize the neighbor influence matrices
        self._initialize_neighbor_influence()
    
    def _create_grid_graph(self):
        """
        Create a graph representing the city grid.
        
        Returns:
            NetworkX graph object with nodes as intersections and edges as roads
        """
        G = nx.grid_2d_graph(self.rows, self.cols)
        
        # Convert to a more usable format and add attributes
        H = nx.DiGraph()
        
        for r in range(self.rows):
            for c in range(self.cols):
                # Use intersection ID format: I{row+1}-{col+1}
                intersection_id = f"I{r+1}-{c+1}"
                
                # Add node with position attributes
                H.add_node(intersection_id, 
                          pos=(r * self.distance, c * self.distance),
                          coords=(r, c))
        
        # Add edges (directed roads between intersections)
        for r in range(self.rows):
            for c in range(self.cols):
                current = f"I{r+1}-{c+1}"
                
                # Connect to adjacent intersections
                adjacents = []
                
                # North neighbor
                if c < self.cols - 1:
                    north = f"I{r+1}-{c+2}"
                    H.add_edge(current, north, direction='north', 
                              distance=self.distance)
                    H.add_edge(north, current, direction='south', 
                              distance=self.distance)
                
                # East neighbor
                if r < self.rows - 1:
                    east = f"I{r+2}-{c+1}"
                    H.add_edge(current, east, direction='east', 
                              distance=self.distance)
                    H.add_edge(east, current, direction='west', 
                              distance=self.distance)
        
        return H
    
    def initialize_traffic(self):
        """Initialize traffic data for all intersections with random distribution"""
        for node in self.graph.nodes():
            # Generate random traffic for each direction
            north_traffic = random.randint(0, 20)
            south_traffic = random.randint(0, 20)
            east_traffic = random.randint(0, 15)
            west_traffic = random.randint(0, 15)
            
            # Add some variations between intersections
            variation_factor = random.uniform(0.7, 1.3)
            north_traffic = int(north_traffic * variation_factor)
            south_traffic = int(south_traffic * variation_factor)
            east_traffic = int(east_traffic * variation_factor)
            west_traffic = int(west_traffic * variation_factor)
            
            # Some intersections could have heavy traffic in one direction
            if random.random() < 0.3:  # 30% chance of heavy directional traffic
                heavy_direction = random.choice(['north', 'south', 'east', 'west'])
                if heavy_direction == 'north':
                    north_traffic = int(north_traffic * 2.5)
                elif heavy_direction == 'south':
                    south_traffic = int(south_traffic * 2.5)
                elif heavy_direction == 'east':
                    east_traffic = int(east_traffic * 2.5)
                else:
                    west_traffic = int(west_traffic * 2.5)
            
            self.intersection_traffic[node] = {
                'north': north_traffic,
                'south': south_traffic,
                'east': east_traffic,
                'west': west_traffic,
                'total': north_traffic + south_traffic + east_traffic + west_traffic
            }
    
    def get_intersection_ids(self):
        """Return a list of all intersection IDs"""
        return list(self.graph.nodes())
    
    def get_adjacent_intersections(self, intersection_id):
        """
        Get adjacent intersections to a given intersection.
        
        Args:
            intersection_id: The ID of the intersection
            
        Returns:
            Dictionary mapping direction to adjacent intersection ID
        """
        if intersection_id not in self.graph:
            return {}
        
        adjacents = {}
        for neighbor in self.graph.neighbors(intersection_id):
            edge_data = self.graph.get_edge_data(intersection_id, neighbor)
            direction = edge_data['direction']
            adjacents[direction] = neighbor
            
        return adjacents
    
    def update_intersection_traffic(self, intersection_id, traffic_data):
        """
        Update traffic data for a specific intersection.
        
        Args:
            intersection_id: ID of the intersection
            traffic_data: Dictionary with lane-wise vehicle counts
        """
        if intersection_id in self.intersection_traffic:
            self.intersection_traffic[intersection_id].update(traffic_data)
            # Update total count
            total = sum(v for k, v in traffic_data.items() if k in self.directions)
            self.intersection_traffic[intersection_id]['total'] = int(total)
    
    def get_intersection_traffic(self, intersection_id):
        """Get current traffic data for a specific intersection"""
        return self.intersection_traffic.get(intersection_id, {})
    
    def get_incoming_traffic(self, intersection_id):
        """
        Get incoming traffic from adjacent intersections.
        
        Args:
            intersection_id: The ID of the intersection
            
        Returns:
            Dictionary mapping direction to incoming traffic count
        """
        incoming = {}
        adjacent = self.get_adjacent_intersections(intersection_id)
        
        for direction, neighbor_id in adjacent.items():
            # Get traffic flowing in the opposite direction from the adjacent intersection
            # e.g., if looking north from current intersection, check south-flowing
            # traffic from the north neighbor
            opposite = self.opposite_direction[direction]
            if neighbor_id in self.intersection_traffic and opposite in self.intersection_traffic[neighbor_id]:
                incoming[direction] = self.intersection_traffic[neighbor_id][opposite]
            else:
                incoming[direction] = 0
                
        return incoming
    
    def add_random_traffic(self, min_vehicles=1, max_vehicles=10):
        """
        Add random traffic to all intersections for simulation purposes.
        
        Args:
            min_vehicles: Minimum number of vehicles to add per lane
            max_vehicles: Maximum number of vehicles to add per lane
        """
        # Define a maximum cap for vehicles at any intersection lane
        MAX_VEHICLES_PER_LANE = 30
        
        for intersection_id in self.graph.nodes():
            for direction in self.directions.keys():
                # Get current vehicle count
                current_count = self.intersection_traffic[intersection_id].get(direction, 0)
                
                # Only add vehicles if below maximum capacity
                if current_count < MAX_VEHICLES_PER_LANE:
                    # Add fewer vehicles when approaching capacity
                    remaining_capacity = MAX_VEHICLES_PER_LANE - current_count
                    actual_max = min(max_vehicles, remaining_capacity)
                    
                    # Adjust probability of new vehicles based on current count
                    # Higher counts mean lower probability of new vehicles
                    probability = max(0.1, 1.0 - (current_count / MAX_VEHICLES_PER_LANE))
                    
                    if random.random() < probability:
                        # Generate a smaller random number when closer to capacity
                        scale_factor = max(0.2, remaining_capacity / MAX_VEHICLES_PER_LANE)
                        new_vehicles = random.randint(min_vehicles, max(min_vehicles, int(actual_max * scale_factor)))
                        
                        if direction in self.intersection_traffic[intersection_id]:
                            self.intersection_traffic[intersection_id][direction] += new_vehicles
                            self.intersection_traffic[intersection_id]['total'] += new_vehicles
    
    def _initialize_neighbor_influence(self):
        """
        Initialize the neighbor influence tracking for intersections.
        This establishes how traffic conditions at one intersection affect its neighbors.
        """
        for node_id in self.graph.nodes():
            self.neighbor_influence[node_id] = {}
            self.congestion_spillover[node_id] = {}
            
            # Get all adjacent intersections
            adjacents = self.get_adjacent_intersections(node_id)
            
            # Initialize influence values for each neighbor
            for direction, neighbor_id in adjacents.items():
                # Initialize with default influence factor
                # This represents how strongly a change in traffic at the neighbor
                # affects traffic at this intersection
                self.neighbor_influence[node_id][neighbor_id] = 0.5  # Default influence
                
                # Initialize congestion spillover (how congestion spreads)
                self.congestion_spillover[node_id][neighbor_id] = {
                    'threshold': 20,  # Traffic level that begins to affect neighbors
                    'factor': 0.3     # How strongly congestion spreads
                }
    
    def _calculate_neighbor_effect(self, intersection_id, signal_states=None):
        """
        Calculate the effect of neighboring intersections' traffic on this intersection.
        
        Args:
            intersection_id: The intersection to calculate effects for
            signal_states: Current signal states
            
        Returns:
            Dictionary with traffic effects per direction
        """
        # Initialize effects
        direction_effects = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        # Get adjacent intersections
        adjacents = self.get_adjacent_intersections(intersection_id)
        
        # For each adjacent intersection, calculate its effect
        for direction, neighbor_id in adjacents.items():
            # Skip if neighbor doesn't exist in our traffic data
            if neighbor_id not in self.intersection_traffic:
                continue
                
            # Get opposite direction (traffic flowing from neighbor to here comes in opposite direction)
            opposite_dir = self.opposite_direction[direction]
            
            # Get neighbor's traffic in connecting direction
            # E.g., if neighbor is to the north, check its south-facing traffic
            connecting_dir = opposite_dir
            neighbor_traffic = self.intersection_traffic[neighbor_id].get(connecting_dir, 0)
            
            # Get signal state of neighbor's connecting direction
            neighbor_has_green = False
            if signal_states and neighbor_id in signal_states:
                neighbor_signals = signal_states[neighbor_id].get('signals', {}).get('states', {})
                neighbor_has_green = neighbor_signals.get(connecting_dir, 'red') == 'green'
            
            # Calculate base effect - traffic from neighbor flowing toward this intersection
            # Higher effect if neighbor has green light in our direction
            if neighbor_has_green:
                # Green light means more traffic flows to us
                influence_factor = 0.4
            else:
                # Red/yellow means less immediate effect
                influence_factor = 0.1
            
            # Calculate congestion spillover effect
            congestion_threshold = self.congestion_spillover[intersection_id][neighbor_id]['threshold']
            spillover_factor = self.congestion_spillover[intersection_id][neighbor_id]['factor']
            
            # If neighbor's traffic exceeds threshold, it spills over
            if neighbor_traffic > congestion_threshold:
                spillover_effect = (neighbor_traffic - congestion_threshold) * spillover_factor
            else:
                spillover_effect = 0
                
            # Calculate combined effect for this direction
            direction_effect = (neighbor_traffic * influence_factor) + spillover_effect
            
            # Apply the effect to the corresponding direction at this intersection
            # Traffic coming FROM north affects our north-facing traffic, etc.
            direction_effects[direction] += int(direction_effect)
            
        return direction_effects
    
    def simulate_traffic_flow(self, time_step_seconds=60, signal_states=None):
        """
        Simulate traffic flowing between intersections with turning movements.
        
        Args:
            time_step_seconds: Simulation time step in seconds
            signal_states: Dictionary mapping intersection IDs to their signal states
            
        Returns:
            Updated traffic state after flow simulation
        """
        # Calculate how far vehicles can travel in this time step
        distance_traveled = self.speed_mps * time_step_seconds
        
        # Process vehicles already in transit
        new_vehicles_in_transit = {}
        
        # Track vehicles arriving at destinations for historical data
        arriving_vehicles = defaultdict(lambda: defaultdict(int))
        
        # ENHANCEMENT: First, identify all green lights to model traffic flow better
        green_directions = {}  # Map of intersection_id -> direction with green light
        if signal_states:
            for intersection_id, state in signal_states.items():
                if 'signals' in state and 'states' in state['signals']:
                    signal_states_dict = state['signals']['states']
                    for direction, color in signal_states_dict.items():
                        if color == 'green':
                            if intersection_id not in green_directions:
                                green_directions[intersection_id] = []
                            green_directions[intersection_id].append(direction)
        
        # NEW: Calculate neighbor effects for all intersections
        neighbor_effects = {}
        for intersection_id in self.graph.nodes():
            neighbor_effects[intersection_id] = self._calculate_neighbor_effect(
                intersection_id, signal_states
            )
        
        # NEW: Apply neighbor effects to traffic counts
        for intersection_id, effects in neighbor_effects.items():
            for direction, effect in effects.items():
                if effect != 0 and direction in self.intersection_traffic[intersection_id]:
                    # Check signal state to determine how influence is applied
                    signal_is_red = True  # Default to red
                    if signal_states and intersection_id in signal_states:
                        intersection_signals = signal_states[intersection_id].get('signals', {}).get('states', {})
                        signal_is_red = intersection_signals.get(direction, 'red') != 'green'
                    
                    # Add more effect to directions with red signals (traffic builds up)
                    # Reduce effect for green signals (traffic flows through)
                    if signal_is_red:
                        # Amplify effect for red signals (traffic accumulates)
                        adjusted_effect = effect * 1.5
                    else:
                        # Reduce effect for green signals (traffic flows through)
                        adjusted_effect = effect * 0.5
                    
                    # Apply effect with limits to prevent unrealistic changes
                    MAX_SINGLE_STEP_CHANGE = 5
                    actual_effect = max(-MAX_SINGLE_STEP_CHANGE, min(MAX_SINGLE_STEP_CHANGE, adjusted_effect))
                    
                    # Apply change if non-zero
                    if actual_effect != 0:
                        # Don't let traffic go below zero
                        current_value = self.intersection_traffic[intersection_id][direction]
                        new_value = max(0, int(current_value + actual_effect))
                        
                        # Apply the change
                        self.intersection_traffic[intersection_id][direction] = new_value
                        
                        # Record for history if it's an increase
                        if actual_effect > 0:
                            arriving_vehicles[intersection_id][direction] += int(actual_effect)
        
        # Process vehicles in transit (moving between intersections)
        for transit_key, transit_data in self.vehicles_in_transit.items():
            parts = transit_key.split('|')
            source = parts[0]
            destination = parts[1]
            direction = parts[2]  # Direction vehicle is coming FROM at source
            turn_type = parts[3] if len(parts) > 3 else 'straight'
            
            # Update remaining distance
            remaining_distance = transit_data['remaining_distance'] - distance_traveled
            vehicles = transit_data['vehicles']
            
            if remaining_distance <= 0:
                # Vehicles have reached the destination intersection
                # Calculate which direction the vehicles are coming FROM at the destination
                # This is the opposite of the direction they left from the source
                dest_direction = self.opposite_direction[direction]
                
                if destination in self.intersection_traffic and dest_direction in self.intersection_traffic[destination]:
                    # Check if destination direction has a red light
                    # CORRECTED: If dest_direction is green, traffic can flow from that direction through the intersection
                    has_red_light = True  # Default to red if no info
                    if signal_states and destination in signal_states:
                        intersection_signals = signal_states[destination].get('signals', {}).get('states', {})
                        has_red_light = intersection_signals.get(dest_direction, 'red') != 'green'
                    
                    # Cap the maximum number of vehicles that can be added
                    MAX_VEHICLES_PER_LANE = 30
                    current_count = self.intersection_traffic[destination][dest_direction]
                    
                    # ENHANCEMENT: If red light, allow more buildup beyond the normal capacity
                    if has_red_light:
                        # Allow more vehicles to accumulate at red lights (up to 150% of normal capacity)
                        effective_max = int(MAX_VEHICLES_PER_LANE * 1.5)
                    else:
                        effective_max = MAX_VEHICLES_PER_LANE
                    
                    # Only add vehicles if there's capacity
                    if current_count < effective_max:
                        # Calculate how many vehicles can be added
                        space_available = effective_max - current_count
                        vehicles_to_add = min(vehicles, space_available)
                        
                        # ENHANCEMENT: Add a higher percentage of vehicles if red light
                        if has_red_light:
                            # Apply a higher percentage of incoming traffic (80-100%)
                            # Traffic accumulates at red lights
                            vehicles_to_add = int(vehicles_to_add * random.uniform(0.8, 1.0))
                        else:
                            # Apply a lower percentage for green lights
                            # Traffic flows through green lights, so less accumulation
                            vehicles_to_add = int(vehicles_to_add * random.uniform(0.3, 0.5))
                        
                        # Make sure we add at least one vehicle if there are any
                        vehicles_to_add = max(1, vehicles_to_add) if vehicles > 0 else 0
                        
                        self.intersection_traffic[destination][dest_direction] += vehicles_to_add
                        
                        # Record for historical data
                        arriving_vehicles[destination][dest_direction] += vehicles_to_add
                        
                        # If not all vehicles could be added, keep the rest in transit
                        if vehicles_to_add < vehicles:
                            # Keep remaining vehicles in transit with a small distance
                            # to simulate queuing outside the intersection
                            new_vehicles_in_transit[transit_key] = {
                                'vehicles': vehicles - vehicles_to_add,
                                'remaining_distance': 10  # Small distance to represent queue
                            }
                    else:
                        # Intersection is at capacity, all vehicles remain in transit
                        new_vehicles_in_transit[transit_key] = {
                            'vehicles': vehicles,
                            'remaining_distance': 10  # Small distance to represent queue
                        }
            else:
                # Vehicles still in transit
                new_vehicles_in_transit[transit_key] = {
                    'vehicles': vehicles,
                    'remaining_distance': remaining_distance
                }
        
        # Replace with updated transit data
        self.vehicles_in_transit = new_vehicles_in_transit
        
        # Generate traffic at each intersection from external sources and neighbors
        for intersection_id in self.graph.nodes():
            adjacents = self.get_adjacent_intersections(intersection_id)
            
            # ENHANCEMENT: Get the traffic from nearby intersections
            nearby_traffic = {}
            for direction, neighbor_id in adjacents.items():
                # CORRECTED: If the neighbor has a green light in the direction facing this intersection,
                # traffic should flow FROM that direction at this intersection
                if neighbor_id in green_directions:
                    # Find which direction at the neighbor points to this intersection
                    connecting_direction = None
                    for d, n_id in self.get_adjacent_intersections(neighbor_id).items():
                        if n_id == intersection_id:
                            connecting_direction = d
                            break
                    
                    if connecting_direction and connecting_direction in green_directions[neighbor_id]:
                        # Neighbor has green light in our direction - traffic will flow to us
                        # Get the volume of traffic in that direction
                        if neighbor_id in self.intersection_traffic:
                            # The traffic will arrive from the opposite direction
                            arriving_from = self.opposite_direction[connecting_direction]
                            nearby_traffic[arriving_from] = self.intersection_traffic[neighbor_id].get(connecting_direction, 0)
            
            # For each direction at the intersection
            for direction in self.directions.keys():
                # Get signal state for this direction
                signal_is_red = True  # Default to red if no signal info
                if signal_states and intersection_id in signal_states:
                    intersection_signals = signal_states[intersection_id].get('signals', {}).get('states', {})
                    signal_is_red = intersection_signals.get(direction, 'red') != 'green'
                
                # Get current traffic count
                current_count = self.intersection_traffic[intersection_id].get(direction, 0)
                
                # ENHANCEMENT: Determine traffic accumulation based on red light + neighboring green
                if signal_is_red and direction in nearby_traffic:
                    neighboring_traffic = nearby_traffic[direction]
                    # Calculate how much traffic to add based on neighbor's volume
                    # Higher flow when traffic is coming from a neighboring green light
                    traffic_influx = int(neighboring_traffic * random.uniform(0.3, 0.6))
                    
                    # Apply to the current direction with a red light
                    MAX_VEHICLES_PER_LANE = 30
                    # Allow more vehicles to accumulate at red lights
                    effective_max = int(MAX_VEHICLES_PER_LANE * 1.5)
                    
                    if current_count < effective_max:
                        # Calculate how many vehicles to add
                        space_available = effective_max - current_count
                        vehicles_to_add = min(traffic_influx, space_available)
                        
                        if vehicles_to_add > 0:
                            self.intersection_traffic[intersection_id][direction] += vehicles_to_add
                            arriving_vehicles[intersection_id][direction] += vehicles_to_add
                
                # Skip edges with adjacent intersections
                # Only handle external traffic for directions without neighboring intersections
                if direction in adjacents:
                    continue
                    
                # This is an "edge" direction (no modeled neighbor)
                # Generate some new traffic from outside the system
                
                # Add more vehicles if signal is red (traffic builds up)
                # Add fewer or none if signal is green (traffic flows through)
                MAX_VEHICLES_PER_LANE = 30
                
                # ENHANCEMENT: Allow more vehicles at red lights
                if signal_is_red:
                    effective_max = int(MAX_VEHICLES_PER_LANE * 1.5)
                else:
                    effective_max = MAX_VEHICLES_PER_LANE
                    
                if current_count < effective_max:
                    remaining_capacity = effective_max - current_count
                    
                    # Traffic addition probability and volume depends on signal state
                    if signal_is_red:
                        # Higher probability and volume for red light (traffic builds up)
                        probability = 0.9  # Increased from 0.8
                        max_new = min(5, remaining_capacity)  # Increased from 3
                    else:
                        # Lower probability and volume for green light (traffic flows through)
                        probability = 0.3
                        max_new = min(1, remaining_capacity)
                    
                    if random.random() < probability:
                        # Add new vehicles from outside the system
                        new_vehicles = random.randint(0, max_new)
                        if new_vehicles > 0:
                            self.intersection_traffic[intersection_id][direction] += new_vehicles
                            
                            # Record for historical data
                            arriving_vehicles[intersection_id][direction] += new_vehicles
                            
        # Move vehicles THROUGH each intersection based on signal states
        for intersection_id in self.graph.nodes():
            adjacents = self.get_adjacent_intersections(intersection_id)
            
            # Get signal states for this intersection if available
            intersection_signals = None
            if signal_states and intersection_id in signal_states:
                intersection_signals = signal_states[intersection_id].get('signals', {}).get('states', {})
            
            # Track which neighboring intersections will receive increased traffic
            neighbor_traffic_increase = defaultdict(int)
            
            # For each incoming direction at the intersection
            for direction in self.directions.keys():
                # Get number of vehicles IN this direction (coming FROM this direction)
                if direction in self.intersection_traffic[intersection_id]:
                    incoming_vehicles = self.intersection_traffic[intersection_id][direction]
                    
                    if incoming_vehicles > 0:
                        # CORRECTED: Check if the signal is green for this direction
                        # Green means traffic FROM this direction can flow THROUGH the intersection
                        is_green = False  # Default to false if no signal info
                        if intersection_signals:
                            is_green = intersection_signals.get(direction, 'red') == 'green'
                        
                        # Get time remaining for current signal to estimate flow rate
                        time_remaining = 30  # Default value
                        if signal_states and intersection_id in signal_states:
                            time_remaining = signal_states[intersection_id].get('signals', {}).get('time_remaining', 30)
                        
                        # ENHANCED: Outflow depends on signal state and traffic volume
                        if is_green:
                            # Base flow rate depends on volume - with higher volumes, flow is slower
                            base_outflow = 0.0
                            
                            # Calculate congestion factor (0-1, where 1 is max congestion)
                            MAX_EXPECTED_VOLUME = 30
                            congestion = min(1.0, incoming_vehicles / MAX_EXPECTED_VOLUME)
                            
                            # Get the previous signal state
                            prev_signal_state = 'unknown'
                            if intersection_id in self.historical_traffic and direction in self.historical_traffic[intersection_id]:
                                historical_data = self.historical_traffic[intersection_id][direction]
                                if len(historical_data) > 0 and 'signal_state' in historical_data[-1]:
                                    prev_signal_state = historical_data[-1]['signal_state']
                            
                            # Check if the light just turned green in this step
                            just_turned_green = prev_signal_state != 'green'
                            
                            # Calculate how long it might take for traffic to start moving
                            # Real traffic doesn't instantly flow when a light turns green
                            green_inertia_factor = 1.0
                            
                            if just_turned_green:
                                # Greatly reduce flow immediately after light turns green (start-up lag)
                                green_inertia_factor = 0.15  # Only 15% of normal flow in first step
                            else:
                                # Check time remaining to estimate how long the light has been green
                                GREEN_CYCLE_TIME = 60  # Approximate typical green cycle
                                time_factor = min(1.0, time_remaining / GREEN_CYCLE_TIME)
                                
                                if time_factor > 0.8:
                                    # Light just turned green recently
                                    green_inertia_factor = 0.3  # 30% of normal flow
                                elif time_factor > 0.6:
                                    # Light has been green for a short while
                                    green_inertia_factor = 0.5  # 50% of normal flow
                                elif time_factor > 0.4:
                                    # Light has been green for some time
                                    green_inertia_factor = 0.7  # 70% of normal flow
                                else:
                                    # Light has been green for a while, normal flow
                                    green_inertia_factor = 1.0
                            
                            # Calculate baseline flow rate based on congestion
                            if congestion > 0.8:
                                # Heavy traffic - slower flow (20-30% with high inertia)
                                base_outflow = random.uniform(0.2, 0.3)
                            elif congestion > 0.5:
                                # Medium traffic - moderate flow (25-35%)
                                base_outflow = random.uniform(0.25, 0.35)
                            else:
                                # Light traffic - faster flow (30-40%)
                                base_outflow = random.uniform(0.3, 0.4)
                            
                            # Apply inertia factor to model realistic acceleration after light change
                            outflow_percentage = base_outflow * green_inertia_factor
                            
                            # Initialize time_factor if it wasn't set yet (for just_turned_green case)
                            if just_turned_green:
                                # For newly turned green lights, use a high time factor
                                time_factor = 0.9
                            
                            # If light is about to change, flow might increase slightly
                            if time_factor < 0.3 and not just_turned_green:
                                # Light is about to change, some cars rush through
                                outflow_percentage *= 1.1
                                
                            # Add some randomness to make flow more natural
                            outflow_percentage *= random.uniform(0.9, 1.1)
                            
                            # Cap at reasonable values
                            outflow_percentage = min(0.4, max(0.05, outflow_percentage))
                        else:
                            # Much lower outflow for red/yellow (0-2%)
                            outflow_percentage = random.uniform(0, 0.02)
                        
                        # Don't let traffic drop too quickly in a single step
                        # For most realistic flow, limit even more for light just turned green
                        if is_green and just_turned_green:
                            # Very slow initial outflow (max 2-3 vehicles)
                            max_outflow_volume = min(incoming_vehicles, random.randint(2, 3))
                        elif is_green:
                            # More moderate outflow for established green (max 3-6 vehicles)
                            max_outflow_volume = min(incoming_vehicles, random.randint(3, 6))
                        else:
                            # Minimal outflow for red/yellow (max 1-2 vehicle)
                            max_outflow_volume = min(incoming_vehicles, random.randint(1, 2))
                        
                        # Calculate vehicles leaving
                        total_leaving = min(int(incoming_vehicles * outflow_percentage), max_outflow_volume)
                        
                        # For small volumes, ensure some reasonable minimum flow if green
                        if is_green and incoming_vehicles > 0:
                            # Ensure at least 1 vehicle leaves if volume is non-zero
                            # But also don't let all vehicles leave at once for small volumes
                            min_vehicles_to_leave = min(1, incoming_vehicles)
                            if incoming_vehicles <= 3:
                                # For very small volumes (1-3 vehicles), keep flow very gradual
                                max_to_leave = 1
                            else:
                                # For larger volumes, ensure some minimum flow
                                max_to_leave = max(min_vehicles_to_leave, total_leaving)
                            total_leaving = max_to_leave
                        
                        # Update source intersection - vehicles are leaving
                        self.intersection_traffic[intersection_id][direction] = int(self.intersection_traffic[intersection_id][direction] - total_leaving)
                        
                        # Distribute departing vehicles according to turn probabilities
                        for turn_type, probability in self.turn_probabilities.items():
                            # Calculate vehicles making this turn
                            vehicles_for_turn = int(total_leaving * probability)
                            if vehicles_for_turn == 0 and total_leaving > 0 and random.random() < probability:
                                # Ensure at least some vehicles go each way if there's flow
                                vehicles_for_turn = 1
                            
                            if vehicles_for_turn == 0:
                                continue
                                
                            # Determine new direction after turn
                            # Where will the vehicles go after turning?
                            new_direction = self.turn_mapping[direction][turn_type]
                            
                            # Find destination based on the new direction
                            destination = None
                            for dest_dir, dest_id in adjacents.items():
                                if dest_dir == new_direction:
                                    destination = dest_id
                                    break
                            
                            # If we found a destination for this direction
                            if destination:
                                # Record this in transit with turn information
                                transit_key = f"{intersection_id}|{destination}|{direction}|{turn_type}"
                                
                                # ENHANCEMENT: For green signals, increase immediate effect on neighbors
                                if is_green:
                                    # Add to neighbor_traffic_increase for immediate effect
                                    # This identifies which neighboring intersections will get more traffic
                                    neighbor_traffic_increase[destination] += vehicles_for_turn
                                    
                                    # Distance depends on congestion - heavier traffic moves slower
                                    transit_distance = self.distance * (0.8 + (congestion * 0.3))
                                else:
                                    # Longer distance for red/yellow to simulate slower movement
                                    transit_distance = self.distance
                                
                                # Record this in transit with turn information
                                self.vehicles_in_transit[transit_key] = {
                                    'vehicles': vehicles_for_turn,
                                    'remaining_distance': transit_distance
                                }
                            else:
                                # No destination in this direction (edge of grid)
                                # These vehicles exit the system
                                pass
        
        # ENHANCEMENT: Apply immediate traffic increase to neighbors from green signal flow
        # This creates the effect of traffic building up on neighbors as soon as a signal turns green
        for neighbor_id, vehicle_increase in neighbor_traffic_increase.items():
            # Find which direction at the neighbor this traffic is coming from
            incoming_direction = None
            for direction, dest_id in self.get_adjacent_intersections(neighbor_id).items():
                if dest_id == intersection_id:
                    incoming_direction = self.opposite_direction[direction]
                    break
            
            if incoming_direction and vehicle_increase > 0:
                # Get the signal state at the neighbor for this direction
                neighbor_signal_is_red = True  # Default to red
                if signal_states and neighbor_id in signal_states:
                    neighbor_signals = signal_states[neighbor_id].get('signals', {}).get('states', {})
                    neighbor_signal_is_red = neighbor_signals.get(incoming_direction, 'red') != 'green'
                
                # If the signal is red at the neighbor, traffic builds up more
                if neighbor_signal_is_red:
                    # Add more vehicles to the neighbor's incoming direction
                    # This immediately shows traffic building up at a red light when a neighboring
                    # intersection has a green light pointing at it
                    
                    # Calculate how many vehicles to add immediately
                    # More immediate effect when signal is red (30-50% of vehicles in transit)
                    immediate_effect = int(vehicle_increase * random.uniform(0.3, 0.5))
                    
                    if incoming_direction in self.intersection_traffic[neighbor_id]:
                        # Apply a cap to avoid excessive buildup
                        MAX_VEHICLES_PER_LANE = 30
                        current_count = self.intersection_traffic[neighbor_id][incoming_direction]
                        
                        # Allow higher capacity for red lights
                        effective_max = int(MAX_VEHICLES_PER_LANE * 1.5)
                        
                        if current_count < effective_max:
                            # Calculate how many vehicles to add
                            space_available = effective_max - current_count
                            vehicles_to_add = min(immediate_effect, space_available)
                            
                            if vehicles_to_add > 0:
                                self.intersection_traffic[neighbor_id][incoming_direction] += vehicles_to_add
                                arriving_vehicles[neighbor_id][incoming_direction] += vehicles_to_add
        
        # Update historical traffic data
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        time_key = f"{current_hour:02d}:{current_minute:02d}"
        
        for intersection_id in self.graph.nodes():
            for direction, count in self.intersection_traffic[intersection_id].items():
                if direction != 'total':  # Skip the total count
                    # Get current signal state for this direction
                    signal_state = 'red'  # Default
                    if signal_states and intersection_id in signal_states:
                        intersection_signals = signal_states[intersection_id].get('signals', {}).get('states', {})
                        signal_state = intersection_signals.get(direction, 'red')
                    
                    # Store current volume and signal state
                    self.historical_traffic[intersection_id][direction].append({
                        'time': time_key,
                        'volume': count,
                        'arrivals': arriving_vehicles[intersection_id][direction],
                        'signal_state': signal_state
                    })
                    
                    # Keep only the last 60 data points (1 hour with minute-by-minute data)
                    if len(self.historical_traffic[intersection_id][direction]) > 60:
                        self.historical_traffic[intersection_id][direction].pop(0)
        
        # Calculate total vehicles per intersection
        for intersection_id in self.graph.nodes():
            total = 0
            for direction in self.directions:
                if direction in self.intersection_traffic[intersection_id]:
                    total += self.intersection_traffic[intersection_id][direction]
            self.intersection_traffic[intersection_id]['total'] = int(total)
        
        # Return current state
        return {
            'intersection_traffic': self.intersection_traffic,
            'vehicles_in_transit': self.vehicles_in_transit,
            'historical_traffic': dict(self.historical_traffic)
        }
    
    def get_traffic_state_dataframe(self):
        """
        Get the current traffic state as a pandas DataFrame.
        
        Returns:
            DataFrame with intersection traffic data
        """
        data = []
        
        for intersection_id, traffic in self.intersection_traffic.items():
            row = {'intersection_id': intersection_id}
            row.update(traffic)
            data.append(row)
            
        return pd.DataFrame(data)
    
    def get_historical_traffic_data(self, intersection_id=None, direction=None):
        """
        Get historical traffic data for analysis.
        
        Args:
            intersection_id: Optional filter for specific intersection
            direction: Optional filter for specific direction
            
        Returns:
            Dictionary with historical traffic data
        """
        if intersection_id and direction:
            return self.historical_traffic[intersection_id][direction]
        elif intersection_id:
            return dict(self.historical_traffic[intersection_id])
        else:
            return dict(self.historical_traffic)


# Example usage
if __name__ == "__main__":
    # Create a 2x2 grid (4 intersections)
    grid = GridSimulator(rows=2, cols=2)
    
    print("Intersection IDs:")
    print(grid.get_intersection_ids())
    
    # Add some random traffic
    grid.add_random_traffic(5, 15)
    
    # Print current traffic state
    print("\nCurrent Traffic State:")
    print(grid.get_traffic_state_dataframe())
    
    # Get adjacent intersections for I1-1
    print("\nAdjacent Intersections for I1-1:")
    print(grid.get_adjacent_intersections("I1-1"))
    
    # Simulate traffic flow for 1 minute
    print("\nSimulating traffic flow...")
    grid.simulate_traffic_flow(60)
    
    # Print updated traffic state
    print("\nUpdated Traffic State:")
    print(grid.get_traffic_state_dataframe()) 