import os
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# Import our simulation modules
from backend.traffic_simulation.generate_data import TrafficDataGenerator
from backend.traffic_simulation.signal_logic import SignalController
from backend.traffic_simulation.grid_simulator import GridSimulator

class TrafficSimulationApp:
    """
    Main application class that coordinates the traffic simulation components.
    """
    
    def __init__(self, grid_rows=2, grid_cols=2, data_dir='../data'):
        """
        Initialize the traffic simulation application.
        
        Args:
            grid_rows: Number of rows in the city grid
            grid_cols: Number of columns in the city grid
            data_dir: Directory to store simulation data
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Initialize simulation components
        self.grid = GridSimulator(rows=grid_rows, cols=grid_cols)
        self.traffic_generator = TrafficDataGenerator(
            num_intersections=grid_rows * grid_cols
        )
        
        # Create a signal controller for each intersection
        self.controllers = {}
        for intersection_id in self.grid.get_intersection_ids():
            self.controllers[intersection_id] = SignalController()
        
        # Simulation state
        self.current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        self.simulation_step = 0
        self.time_step_seconds = 60  # 1 minute per step
        
        # Store for historical data
        self.historical_data = pd.DataFrame()
        
        # Initialize simulation state
        self.initialize_simulation()
    
    def initialize_simulation(self):
        """Initialize the simulation with synthetic traffic data"""
        # Generate initial traffic for 60 minutes
        initial_data = self.traffic_generator.generate_traffic_dataset(
            self.current_time, duration_minutes=60
        )
        
        # Store as historical data
        self.historical_data = initial_data
        self.save_historical_data()
        
        # Set initial traffic state for grid
        self._update_grid_from_generator_data(initial_data, step=0)
    
    def _update_grid_from_generator_data(self, data, step=0):
        """
        Update grid simulator with traffic data from generator.
        
        Args:
            data: DataFrame with traffic data
            step: Simulation step to use
        """
        # Get data for the current time step
        time_data = data[data['timestamp'] == data['timestamp'].unique()[step]]
        
        # Update each intersection in the grid
        intersection_map = self._map_generator_to_grid_ids(len(self.grid.get_intersection_ids()))
        
        for _, row in time_data.iterrows():
            gen_id = row['intersection_id']
            
            if gen_id in intersection_map:
                grid_id = intersection_map[gen_id]
                
                # Extract lane traffic
                traffic_data = {
                    'north': row['north_vehicles'],
                    'south': row['south_vehicles'],
                    'east': row['east_vehicles'],
                    'west': row['west_vehicles']
                }
                
                # Update intersection traffic in grid
                self.grid.update_intersection_traffic(grid_id, traffic_data)
    
    def _map_generator_to_grid_ids(self, num_intersections):
        """
        Map generator intersection IDs to grid intersection IDs.
        
        Args:
            num_intersections: Number of intersections
            
        Returns:
            Dictionary mapping generator IDs to grid IDs
        """
        gen_ids = [f"I{i+1}" for i in range(num_intersections)]
        grid_ids = self.grid.get_intersection_ids()
        
        # Simple mapping assuming same order
        return {gen_id: grid_id for gen_id, grid_id in zip(gen_ids, grid_ids)}
    
    def step_simulation(self):
        """
        Advance the simulation by one time step.
        
        Returns:
            Dictionary with updated simulation state
        """
        # Advance time
        self.current_time += timedelta(seconds=self.time_step_seconds)
        self.simulation_step += 1
        
        # Generate new traffic if needed (every 10 steps)
        if self.simulation_step % 10 == 0:
            new_data = self.traffic_generator.generate_traffic_dataset(
                self.current_time, duration_minutes=10
            )
            self.historical_data = pd.concat([self.historical_data, new_data])
            self.save_historical_data()
        
        # Add some random traffic to existing intersections for variety is now handled
        # directly in the grid simulator based on signal states for more realism
        
        # First, get the current traffic state for all intersections
        all_intersection_traffic = {}
        for intersection_id in self.grid.get_intersection_ids():
            traffic_data = self.grid.get_intersection_traffic(intersection_id)
            all_intersection_traffic[intersection_id] = {
                'north': traffic_data.get('north', 0),
                'south': traffic_data.get('south', 0),
                'east': traffic_data.get('east', 0),
                'west': traffic_data.get('west', 0)
            }
        
        # Update signal controllers
        intersection_states = {}
        all_signal_states = {}
        
        for intersection_id, controller in self.controllers.items():
            # Get traffic data and incoming traffic
            traffic_counts = all_intersection_traffic[intersection_id]
            incoming_traffic = self.grid.get_incoming_traffic(intersection_id)
            
            # Update signal controller
            signal_state = controller.update_signal_state(self.current_time, traffic_counts)
            
            intersection_states[intersection_id] = {
                'traffic': traffic_counts,
                'incoming': incoming_traffic,
                'signals': signal_state
            }
            
            # Collect signal states for traffic flow simulation
            all_signal_states[intersection_id] = signal_state
        
        # Cap all intersection traffic to ensure it stays within reasonable limits
        MAX_VEHICLES_PER_LANE = 30
        for intersection_id, traffic in all_intersection_traffic.items():
            for direction in ['north', 'south', 'east', 'west']:
                if direction in self.grid.intersection_traffic[intersection_id]:
                    current = self.grid.intersection_traffic[intersection_id][direction]
                    if current > MAX_VEHICLES_PER_LANE:
                        self.grid.intersection_traffic[intersection_id][direction] = MAX_VEHICLES_PER_LANE
        
        # Simulate traffic flow between intersections, passing signal states
        flow_results = self.grid.simulate_traffic_flow(
            self.time_step_seconds, 
            signal_states=intersection_states  # Pass the full intersection states to include signals
        )
        
        # Update the traffic state after flow simulation
        # This allows UI to display the latest traffic after movement
        for intersection_id in self.grid.get_intersection_ids():
            traffic_data = self.grid.get_intersection_traffic(intersection_id)
            updated_traffic = {
                'north': traffic_data.get('north', 0),
                'south': traffic_data.get('south', 0),
                'east': traffic_data.get('east', 0),
                'west': traffic_data.get('west', 0)
            }
            intersection_states[intersection_id]['traffic'] = updated_traffic
        
        # Return current state
        return {
            'time': self.current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'step': self.simulation_step,
            'intersections': intersection_states,
            'grid_state': {
                'traffic': flow_results['intersection_traffic'],
                'vehicles_in_transit': flow_results['vehicles_in_transit']
            }
        }
    
    def get_current_state(self):
        """Get the current state of all intersections"""
        intersection_states = {}
        for intersection_id, controller in self.controllers.items():
            traffic_data = self.grid.get_intersection_traffic(intersection_id)
            
            # Get traffic from neighboring intersections
            incoming_traffic = self.grid.get_incoming_traffic(intersection_id)
            
            # Extract the traffic counts only
            traffic_counts = {
                'north': traffic_data.get('north', 0),
                'south': traffic_data.get('south', 0),
                'east': traffic_data.get('east', 0),
                'west': traffic_data.get('west', 0)
            }
            
            # Get current signal state without updating
            signal_state = controller.get_historical_patterns()
            intersection_states[intersection_id] = {
                'traffic': traffic_counts,
                'incoming': incoming_traffic
            }
        
        return {
            'time': self.current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'step': self.simulation_step,
            'intersections': intersection_states
        }
    
    def get_grid_layout(self):
        """Get the layout of the city grid"""
        nodes = []
        edges = []
        
        # Get node positions
        for node_id in self.grid.graph.nodes():
            pos = self.grid.graph.nodes[node_id].get('pos', (0, 0))
            nodes.append({
                'id': node_id,
                'x': pos[0],
                'y': pos[1]
            })
        
        # Get edges (roads)
        for source, target, data in self.grid.graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'direction': data.get('direction', ''),
                'distance': data.get('distance', 0)
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def save_historical_data(self):
        """Save historical traffic data to CSV"""
        if not self.historical_data.empty:
            file_path = os.path.join(self.data_dir, 'historic_traffic.csv')
            self.historical_data.to_csv(file_path, index=False)
    
    def load_historical_data(self):
        """Load historical traffic data from CSV"""
        file_path = os.path.join(self.data_dir, 'historic_traffic.csv')
        if os.path.exists(file_path):
            self.historical_data = pd.read_csv(file_path)
            # Convert timestamp strings back to datetime
            self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
            return True
        return False


# Example usage
if __name__ == "__main__":
    # Create simulation app
    app = TrafficSimulationApp()
    
    # Run for 10 steps
    for i in range(10):
        state = app.step_simulation()
        print(f"Step {i+1} - Time: {state['time']}")
        
        # Print traffic for first intersection
        first_intersection = list(state['intersections'].keys())[0]
        print(f"Intersection {first_intersection} traffic:")
        print(f"North: {state['intersections'][first_intersection]['traffic']['north']}")
        print(f"South: {state['intersections'][first_intersection]['traffic']['south']}")
        print(f"East: {state['intersections'][first_intersection]['traffic']['east']}")
        print(f"West: {state['intersections'][first_intersection]['traffic']['west']}")
        print(f"Signal state: {state['intersections'][first_intersection]['signals']['states']}")
        print("-" * 40)
        
        # Pause for a moment
        time.sleep(0.5) 