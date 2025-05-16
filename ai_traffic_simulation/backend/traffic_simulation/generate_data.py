import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class TrafficDataGenerator:
    """
    Generates synthetic traffic data for multiple intersections
    with realistic patterns and variability.
    """
    
    def __init__(self, num_intersections=4, lanes_per_intersection=4):
        """
        Initialize the traffic generator.
        
        Args:
            num_intersections: Number of intersections to simulate
            lanes_per_intersection: Number of lanes per intersection (N/S/E/W)
        """
        self.num_intersections = num_intersections
        self.lanes_per_intersection = lanes_per_intersection
        self.lane_names = ['north', 'south', 'east', 'west']
        
        # Traffic pattern parameters
        self.base_traffic = {
            'morning_rush': {
                'peak_hour': 8,  # 8 AM
                'peak_factor': 2.5,
                'duration_hours': 3
            },
            'evening_rush': {
                'peak_hour': 17,  # 5 PM
                'peak_factor': 2.2,
                'duration_hours': 4
            },
            'lunch_time': {
                'peak_hour': 12,  # 12 PM
                'peak_factor': 1.5,
                'duration_hours': 2
            }
        }
        
        # Base traffic volumes by intersection 
        # (some intersections naturally have more traffic)
        self.intersection_base_volumes = {}
        for i in range(num_intersections):
            # Random base traffic for each intersection (5-20 vehicles per minute)
            self.intersection_base_volumes[f'I{i+1}'] = random.randint(5, 20)
            
        # Lane weights (some lanes get more traffic than others)
        self.lane_weights = {}
        for i in range(num_intersections):
            self.lane_weights[f'I{i+1}'] = {}
            # Distribute weights that sum to approximately 1.0
            weights = np.random.dirichlet(np.ones(lanes_per_intersection), size=1)[0]
            for j, lane in enumerate(self.lane_names):
                # Add some variability but keep sum close to 1.0
                self.lane_weights[f'I{i+1}'][lane] = max(0.1, weights[j])
    
    def _calculate_time_factor(self, hour):
        """Calculate traffic volume multiplier based on time of day"""
        time_factor = 1.0  # Default factor
        
        # Apply rush hour effects
        for rush_period, params in self.base_traffic.items():
            peak_hour = params['peak_hour']
            peak_factor = params['peak_factor']
            duration = params['duration_hours']
            
            # Calculate hours from peak
            hours_from_peak = min(abs(hour - peak_hour), 
                                  abs(hour - peak_hour + 24), 
                                  abs(hour - peak_hour - 24))
            
            # If within duration of peak
            if hours_from_peak <= duration / 2:
                # Linear decrease from peak
                period_factor = peak_factor * (1 - hours_from_peak / (duration / 2))
                time_factor = max(time_factor, period_factor)
        
        # Night time factor (midnight to 5 AM)
        if 0 <= hour < 5:
            night_factor = 0.3 + (hour / 5) * 0.7  # Gradually increase from 0.3 to 1.0
            time_factor = min(time_factor, night_factor)
            
        return time_factor
    
    def _add_noise(self, value, noise_level=0.2):
        """Add random noise to a value for more realism"""
        noise = np.random.normal(0, noise_level * value)
        return max(0, int(value + noise))
    
    def generate_minute_data(self, timestamp):
        """
        Generate traffic data for a specific minute across all intersections.
        
        Args:
            timestamp: The datetime to generate data for
            
        Returns:
            List of dictionaries with traffic data for each intersection
        """
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()  # 0-6 (Monday to Sunday)
        
        # Weekend factor (less traffic on weekends)
        weekend_factor = 0.7 if day_of_week >= 5 else 1.0
        
        # Time of day factor
        time_factor = self._calculate_time_factor(hour)
        
        # Combined factor
        combined_factor = time_factor * weekend_factor
        
        # Generate data for all intersections
        data_points = []
        
        for i in range(self.num_intersections):
            intersection_id = f'I{i+1}'
            base_volume = self.intersection_base_volumes[intersection_id]
            
            # Calculate total vehicles for this minute at this intersection
            total_volume = int(base_volume * combined_factor)
            
            # Distribute across lanes based on weights
            lane_volumes = {}
            remaining_volume = total_volume
            
            for lane, weight in self.lane_weights[intersection_id].items():
                # Calculate volume for this lane and add noise
                lane_volume = self._add_noise(total_volume * weight)
                lane_volumes[lane] = lane_volume
                remaining_volume -= lane_volume
            
            # Create data point
            data_point = {
                'timestamp': timestamp,
                'intersection_id': intersection_id,
                'total_vehicles': sum(lane_volumes.values())
            }
            
            # Add lane-specific counts
            for lane, volume in lane_volumes.items():
                data_point[f'{lane}_vehicles'] = volume
                
            data_points.append(data_point)
            
        return data_points
    
    def generate_traffic_dataset(self, start_time, duration_minutes):
        """
        Generate a dataset of traffic data for a specified duration.
        
        Args:
            start_time: Datetime object for the start time
            duration_minutes: How many minutes to generate data for
            
        Returns:
            Pandas DataFrame with the generated data
        """
        all_data = []
        
        current_time = start_time
        for _ in range(duration_minutes):
            minute_data = self.generate_minute_data(current_time)
            all_data.extend(minute_data)
            current_time += timedelta(minutes=1)
            
        # Convert to dataframe
        df = pd.DataFrame(all_data)
        return df
    
    def save_to_csv(self, df, file_path):
        """Save the generated dataset to a CSV file"""
        df.to_csv(file_path, index=False)
        print(f"Saved traffic data to {file_path}")


# Example usage
if __name__ == "__main__":
    # Generate 60 minutes (1 hour) of traffic data
    generator = TrafficDataGenerator(num_intersections=4)
    
    # Start at 8 AM today
    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    
    # Generate 60 minutes of data
    traffic_df = generator.generate_traffic_dataset(start_time, duration_minutes=60)
    
    # Display sample
    print(traffic_df.head())
    
    # Save to CSV
    generator.save_to_csv(traffic_df, "../../../data/historic_traffic.csv") 