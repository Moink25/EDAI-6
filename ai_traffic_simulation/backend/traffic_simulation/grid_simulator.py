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
        self.directions = {
            'north': (0, 1),
            'south': (0, -1),
            'east': (1, 0),
            'west': (-1, 0)
        }
        
        # Map directions to opposite directions (for receiving traffic)
        self.opposite_direction = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        
        # Turn probabilities - how likely a vehicle is to go each direction
        self.turn_probabilities = {
            'straight': 0.7,  # 70% go straight
            'left': 0.15,     # 15% turn left
            'right': 0.15     # 15% turn right
        }
        
        # Direction changes for turns
        self.turn_mapping = {
            'north': {'left': 'west', 'straight': 'north', 'right': 'east'},
            'south': {'left': 'east', 'straight': 'south', 'right': 'west'},
            'east': {'left': 'north', 'straight': 'east', 'right': 'south'},
            'west': {'left': 'south', 'straight': 'west', 'right': 'north'}
        }
        
        # Track historical traffic patterns
        self.historical_traffic = defaultdict(lambda: defaultdict(list))
    
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
        """Initialize traffic data for all intersections"""
        for node in self.graph.nodes():
            self.intersection_traffic[node] = {
                'north': 0,
                'south': 0,
                'east': 0,
                'west': 0,
                'total': 0
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
            self.intersection_traffic[intersection_id]['total'] = total
    
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
        
        for transit_key, transit_data in self.vehicles_in_transit.items():
            parts = transit_key.split('|')
            source = parts[0]
            destination = parts[1]
            direction = parts[2]
            # If turn direction was specified
            turn_type = parts[3] if len(parts) > 3 else 'straight'
            
            # Update remaining distance
            remaining_distance = transit_data['remaining_distance'] - distance_traveled
            vehicles = transit_data['vehicles']
            
            if remaining_distance <= 0:
                # Vehicles have reached the destination
                # Add to the destination's incoming lane in the appropriate direction
                dest_direction = self.opposite_direction[direction]
                
                if destination in self.intersection_traffic and dest_direction in self.intersection_traffic[destination]:
                    # Cap the maximum number of vehicles that can be added
                    MAX_VEHICLES_PER_LANE = 30
                    current_count = self.intersection_traffic[destination][dest_direction]
                    
                    # Only add vehicles if there's capacity
                    if current_count < MAX_VEHICLES_PER_LANE:
                        # Calculate how many vehicles can be added
                        space_available = MAX_VEHICLES_PER_LANE - current_count
                        vehicles_to_add = min(vehicles, space_available)
                        
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
        
        # First, generate traffic at each intersection to simulate arrivals from non-modeled roads
        # This represents traffic coming from outside the system or from minor roads
        for intersection_id in self.graph.nodes():
            adjacents = self.get_adjacent_intersections(intersection_id)
            
            # For each direction at the intersection
            for direction in self.directions.keys():
                # Skip if this direction has a modeled neighboring intersection
                if direction in adjacents:
                    continue
                    
                # This is an "edge" direction (no modeled neighbor)
                # Generate some new traffic from outside the system
                current_count = self.intersection_traffic[intersection_id].get(direction, 0)
                
                # If we have signal states, use them to determine if traffic builds up
                signal_is_red = True  # Default to red (traffic builds up) if no signal info
                if signal_states and intersection_id in signal_states:
                    intersection_signals = signal_states[intersection_id].get('signals', {}).get('states', {})
                    signal_is_red = intersection_signals.get(direction, 'red') != 'green'
                
                # Add more vehicles if signal is red (traffic builds up)
                # Add fewer or none if signal is green (traffic flows through)
                MAX_VEHICLES_PER_LANE = 30
                
                if current_count < MAX_VEHICLES_PER_LANE:
                    remaining_capacity = MAX_VEHICLES_PER_LANE - current_count
                    
                    # Traffic addition probability and volume depends on signal state
                    if signal_is_red:
                        # Higher probability and volume for red light (traffic builds up)
                        probability = 0.8
                        max_new = min(3, remaining_capacity)
                    else:
                        # Lower probability and volume for green light
                        probability = 0.3
                        max_new = min(1, remaining_capacity)
                    
                    if random.random() < probability:
                        # Add new vehicles from outside the system
                        new_vehicles = random.randint(0, max_new)
                        if new_vehicles > 0:
                            self.intersection_traffic[intersection_id][direction] += new_vehicles
                            self.intersection_traffic[intersection_id]['total'] += new_vehicles
                            
                            # Record for historical data
                            arriving_vehicles[intersection_id][direction] += new_vehicles
        
        # Move vehicles from each intersection based on signal states and outgoing lanes
        for intersection_id in self.graph.nodes():
            adjacents = self.get_adjacent_intersections(intersection_id)
            
            # Get signal states for this intersection if available
            intersection_signals = None
            if signal_states and intersection_id in signal_states:
                intersection_signals = signal_states[intersection_id].get('signals', {}).get('states', {})
            
            # For each outgoing direction
            for direction in self.directions.keys():
                # Get number of vehicles heading this direction
                if direction in self.intersection_traffic[intersection_id]:
                    outgoing_vehicles = self.intersection_traffic[intersection_id][direction]
                    
                    if outgoing_vehicles > 0:
                        # Check if the signal is green for this direction
                        is_green = True  # Default to true if no signal info
                        if intersection_signals:
                            is_green = intersection_signals.get(direction, 'red') == 'green'
                        
                        # Outflow depends on signal state
                        if is_green:
                            # Higher outflow for green signal (40-70%)
                            outflow_percentage = random.uniform(0.4, 0.7)
                        else:
                            # Much lower outflow for red/yellow (0-5% - represents illegal movements or right turns on red)
                            outflow_percentage = random.uniform(0, 0.05)
                            
                        # Calculate vehicles leaving
                        total_leaving = min(int(outgoing_vehicles * outflow_percentage), outgoing_vehicles)
                        
                        # Update source intersection - vehicles are leaving
                        self.intersection_traffic[intersection_id][direction] -= total_leaving
                        
                        # Distribute departing vehicles according to turn probabilities
                        for turn_type, probability in self.turn_probabilities.items():
                            # Calculate vehicles making this turn
                            vehicles_for_turn = int(total_leaving * probability)
                            if vehicles_for_turn == 0:
                                continue
                                
                            # Determine new direction after turn
                            new_direction = self.turn_mapping[direction][turn_type]
                            
                            # Find destination based on the new direction
                            destination = None
                            for dest_dir, dest_id in adjacents.items():
                                if dest_dir == new_direction:
                                    destination = dest_id
                                    break
                            
                            # If we found a destination for this direction
                            if destination:
                                # Send vehicles to the destination with turn info
                                transit_key = f"{intersection_id}|{destination}|{direction}|{turn_type}"
                                
                                # Record this in transit with turn information
                                self.vehicles_in_transit[transit_key] = {
                                    'vehicles': vehicles_for_turn,
                                    'remaining_distance': self.distance
                                }
                            else:
                                # No destination in this direction (edge of grid)
                                # These vehicles exit the system
                                pass
        
        # Update historical traffic data
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        time_key = f"{current_hour:02d}:{current_minute:02d}"
        
        for intersection_id in self.graph.nodes():
            for direction, count in self.intersection_traffic[intersection_id].items():
                if direction != 'total':  # Skip the total count
                    # Store current volume
                    self.historical_traffic[intersection_id][direction].append({
                        'time': time_key,
                        'volume': count,
                        'arrivals': arriving_vehicles[intersection_id][direction]
                    })
                    
                    # Keep only the last 60 data points (1 hour with minute-by-minute data)
                    if len(self.historical_traffic[intersection_id][direction]) > 60:
                        self.historical_traffic[intersection_id][direction].pop(0)
        
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