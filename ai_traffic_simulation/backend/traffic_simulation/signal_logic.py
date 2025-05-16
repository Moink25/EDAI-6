import numpy as np
import pandas as pd
from collections import defaultdict
import math

class SignalController:
    """
    Smart signal controller that computes optimal traffic light timing
    based on current traffic conditions and optional historical data.
    """
    
    def __init__(self, min_green_time=10, max_green_time=90, yellow_time=5):
        """
        Initialize signal controller with timing constraints.
        
        Args:
            min_green_time: Minimum green light duration in seconds
            max_green_time: Maximum green light duration in seconds
            yellow_time: Yellow light duration in seconds
        """
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        
        # History of traffic patterns for learning
        self.history = defaultdict(list)
        
        # Default sequence: north, east, south, west
        self.lane_sequence = ['north', 'east', 'south', 'west']
        
        # Variables to track current state
        self.current_lane = None
        self.time_remaining = 0
        self.is_yellow = False
        
        # Parameters for sophisticated calculations
        self.saturation_flow_rate = 1800  # vehicles per hour of green per lane
        self.lost_time_per_phase = 2  # seconds
        self.critical_density = 25  # vehicles per km
        self.jam_density = 150  # vehicles per km
        self.average_vehicle_length = 5  # meters
        self.reaction_time = 1.5  # seconds
        
        # Adaptive parameters
        self.alpha = 0.3  # Weight for historical data
        self.beta = 0.7   # Weight for current data
        self.gamma = 0.4  # Weight for neighbor data
    
    def _normalize_volumes(self, volumes):
        """
        Normalize traffic volumes to determine proportional green times.
        
        Args:
            volumes: Dictionary of lane -> vehicle count
        
        Returns:
            Dictionary of normalized values (sum = 1.0)
        """
        total = sum(volumes.values())
        if total == 0:
            # If no vehicles, distribute evenly
            return {lane: 1.0/len(volumes) for lane in volumes}
        
        return {lane: count/total for lane, count in volumes.items()}
    
    def compute_green_times(self, traffic_data, neighbors_data=None, 
                           time_of_day=None, historical_data=None):
        """
        Compute optimal green light durations for each lane based on current traffic.
        
        Args:
            traffic_data: Dictionary with lane -> vehicle count
            neighbors_data: Optional data from neighboring intersections
            time_of_day: Optional time (hour) for historical pattern matching
            historical_data: Optional historical traffic patterns
            
        Returns:
            Dictionary of lane -> green time in seconds
        """
        # Step 1: Calculate traffic flow rates (vehicles per hour)
        flow_rates = {lane: count * 60 for lane, count in traffic_data.items()}
        
        # Step 2: Calculate degree of saturation (v/c ratio)
        saturation_degrees = {}
        for lane, flow in flow_rates.items():
            # v/c ratio = flow rate / saturation flow rate
            # Higher ratio means lane needs more green time
            saturation_degrees[lane] = min(1.0, flow / self.saturation_flow_rate)
        
        # Step 3: Calculate optimal cycle length using Webster's formula
        # C = (1.5L + 5) / (1 - Y)
        # where L is total lost time and Y is sum of critical flow ratios
        total_lost_time = len(traffic_data) * self.lost_time_per_phase
        sum_critical_ratios = sum(saturation_degrees.values())
        
        # Ensure the sum doesn't exceed or equal 1 to avoid division by zero
        if sum_critical_ratios >= 0.95:
            sum_critical_ratios = 0.95
            
        # Webster's formula for optimal cycle length
        optimal_cycle = (1.5 * total_lost_time + 5) / (1 - sum_critical_ratios)
        
        # Apply practical constraints
        cycle_length = max(60, min(120, optimal_cycle))
        
        # Step 4: Distribute green time proportionally to v/c ratios
        allocatable_time = cycle_length - (len(traffic_data) * self.yellow_time)
        
        # Green time is proportional to degree of saturation
        green_times = {}
        for lane, degree in saturation_degrees.items():
            # Base green time calculation
            if sum_critical_ratios > 0:
                green_proportion = degree / sum_critical_ratios
                green_times[lane] = int(allocatable_time * green_proportion)
            else:
                # Equal distribution if all volumes are zero
                green_times[lane] = int(allocatable_time / len(traffic_data))
        
        # Step 5: Apply queue-length-based adjustments
        self._adjust_for_queue_length(green_times, traffic_data)
        
        # Step 6: Adjust for neighboring intersections if available
        if neighbors_data:
            self._adjust_for_neighbors(green_times, traffic_data, neighbors_data)
        
        # Step 7: Adjust based on historical patterns if available
        if historical_data and time_of_day is not None:
            self._adjust_for_historical(green_times, traffic_data, 
                                       time_of_day, historical_data)
        
        # Ensure constraints
        for lane in green_times:
            green_times[lane] = min(green_times[lane], self.max_green_time)
            if traffic_data[lane] > 0:
                green_times[lane] = max(green_times[lane], self.min_green_time)
        
        return green_times
    
    def _adjust_for_queue_length(self, green_times, traffic_data):
        """
        Adjust green times based on estimated queue lengths and clearance times.
        
        Args:
            green_times: Dictionary of current green time allocations (modified in-place)
            traffic_data: Current traffic volumes by lane
        """
        for lane, volume in traffic_data.items():
            if volume > 0:
                # Estimate queue length in meters
                queue_length = volume * self.average_vehicle_length
                
                # Estimate time to clear queue (s = v0*t + 0.5*a*t^2)
                # Using simple acceleration model with 2.5 m/s^2 average acceleration
                acceleration = 2.5  # m/s^2
                avg_speed = 8.33    # 30 km/h = 8.33 m/s
                
                # Time needed to clear the queue
                # t = sqrt(2*s/a) assuming starting from stop
                clearance_time = math.sqrt(2 * queue_length / acceleration)
                
                # Add startup delay for each vehicle (reaction time)
                clearance_time += min(5, volume * 0.5)  # Cap the total reaction time
                
                # Adjust green time based on clearance time
                min_needed_time = max(self.min_green_time, int(clearance_time))
                green_times[lane] = max(green_times[lane], min_needed_time)
    
    def _adjust_for_neighbors(self, green_times, traffic_data, neighbors_data):
        """
        Adjust green times based on traffic at neighboring intersections using 
        a coordination algorithm for "green wave" effect.
        
        Args:
            green_times: Dictionary of current green time allocations (modified in-place)
            traffic_data: Current traffic data for this intersection
            neighbors_data: Dictionary mapping direction to upstream traffic
        """
        # Map of which direction's traffic affects which lane
        direction_to_lane = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        
        for direction, incoming_volume in neighbors_data.items():
            if direction in direction_to_lane:
                affected_lane = direction_to_lane[direction]
                
                if affected_lane in green_times and incoming_volume > 0:
                    # Calculate platoon factor based on incoming volume
                    platoon_factor = 1.0 + (0.5 * math.log(1 + incoming_volume / 10))
                    
                    # Calculate offset needed for coordination
                    # Based on distance and average speed (simplified)
                    # In a real system, this would use actual distances and speeds
                    coordination_bonus = int(5 * platoon_factor)
                    
                    # Apply adjustment
                    green_times[affected_lane] += coordination_bonus
    
    def _adjust_for_historical(self, green_times, traffic_data, 
                              time_of_day, historical_data):
        """
        Adjust green times based on historical traffic patterns using
        exponential smoothing and time series prediction.
        
        Args:
            green_times: Dictionary of current green time allocations (modified in-place)
            traffic_data: Current traffic data
            time_of_day: Current hour (0-23)
            historical_data: DataFrame with historical traffic patterns
        """
        # Filter historical data for the current hour
        hour_data = historical_data[historical_data['hour'] == time_of_day]
        
        if len(hour_data) > 0:
            # Calculate average and trend from historical data
            historical_volumes = {}
            historical_trends = {}
            
            for lane in traffic_data.keys():
                col_name = f'{lane}_vehicles'
                if col_name in hour_data.columns:
                    # Get average volume
                    historical_volumes[lane] = hour_data[col_name].mean()
                    
                    # Calculate trend (simplified)
                    if len(hour_data) > 1:
                        values = hour_data[col_name].values
                        trend = np.mean(np.diff(values)) if len(values) > 1 else 0
                        historical_trends[lane] = trend
                    else:
                        historical_trends[lane] = 0
            
            # Predict next period volume using simple forecasting
            predicted_volumes = {}
            for lane in traffic_data.keys():
                if lane in historical_volumes:
                    # Current + (alpha * historical + beta * trend)
                    current_volume = traffic_data[lane]
                    hist_volume = historical_volumes[lane]
                    trend = historical_trends.get(lane, 0)
                    
                    # Blend current with historical and apply trend
                    predicted = (
                        self.beta * current_volume + 
                        self.alpha * hist_volume + 
                        trend
                    )
                    predicted_volumes[lane] = max(0, predicted)
            
            # Calculate new green times based on predicted volumes
            if predicted_volumes:
                # Calculate total predicted volume
                total_predicted = sum(predicted_volumes.values())
                
                if total_predicted > 0:
                    # Allocate green time proportionally to predicted volumes
                    total_green_time = sum(green_times.values())
                    for lane, predicted in predicted_volumes.items():
                        proportion = predicted / total_predicted
                        adjusted_time = int(total_green_time * proportion)
                        
                        # Apply weighted adjustment
                        green_times[lane] = int(
                            self.beta * green_times[lane] + 
                            self.alpha * adjusted_time
                        )
    
    def calculate_traffic_statistics(self, traffic_data, green_times):
        """
        Calculate advanced traffic statistics based on current volumes and green times.
        
        Args:
            traffic_data: Dictionary with lane -> vehicle count
            green_times: Dictionary with lane -> green time in seconds
            
        Returns:
            Dictionary with traffic performance metrics
        """
        results = {}
        total_delay = 0
        total_vehicles = sum(traffic_data.values())
        cycle_length = sum(green_times.values()) + len(green_times) * self.yellow_time
        
        for lane, volume in traffic_data.items():
            if volume > 0:
                # Calculate lane capacity
                green_ratio = green_times[lane] / cycle_length
                capacity = self.saturation_flow_rate * green_ratio * 60  # per hour
                
                # Calculate v/c ratio
                vc_ratio = min(1.0, (volume * 60) / capacity)
                
                # Calculate average delay using Webster's formula
                # d = 0.5*C*(1-g/C)²/(1-min(1,X)*g/C) + 0.65*(C/X²)*(X)^(2+5*g/C)
                # where C is cycle length, g is green time, X is v/c ratio
                g_over_c = green_times[lane] / cycle_length
                term1 = 0.5 * cycle_length * ((1 - g_over_c)**2) / (1 - min(1, vc_ratio) * g_over_c)
                term2 = 0.65 * (cycle_length / (vc_ratio**2)) * (vc_ratio**(2 + 5*g_over_c))
                delay = min(120, term1 + term2)  # Cap at 120 seconds for realism
                
                # Store lane results
                results[lane] = {
                    'volume': volume,
                    'capacity': capacity,
                    'v_c_ratio': vc_ratio,
                    'delay': delay,
                    'level_of_service': self._get_level_of_service(delay)
                }
                
                total_delay += delay * volume
        
        # Calculate intersection performance
        if total_vehicles > 0:
            results['intersection'] = {
                'total_vehicles': total_vehicles,
                'average_delay': total_delay / total_vehicles,
                'level_of_service': self._get_level_of_service(total_delay / total_vehicles)
            }
        else:
            results['intersection'] = {
                'total_vehicles': 0,
                'average_delay': 0,
                'level_of_service': 'A'
            }
        
        return results
    
    def _get_level_of_service(self, delay):
        """
        Determine Level of Service (LOS) based on average delay.
        
        Args:
            delay: Average delay in seconds per vehicle
            
        Returns:
            Level of Service letter grade (A-F)
        """
        if delay <= 10:
            return 'A'
        elif delay <= 20:
            return 'B'
        elif delay <= 35:
            return 'C'
        elif delay <= 55:
            return 'D'
        elif delay <= 80:
            return 'E'
        else:
            return 'F'
    
    def update_signal_state(self, current_time, traffic_data):
        """
        Update the internal state of the signal controller.
        
        Args:
            current_time: Current simulation time
            traffic_data: Current traffic volumes by lane
            
        Returns:
            Dictionary with current signal states for all lanes
        """
        # Compute green times based on current traffic
        green_times = self.compute_green_times(traffic_data)
        
        # Ensure green times are properly capped at min and max values
        for lane in green_times:
            green_times[lane] = max(self.min_green_time, min(green_times[lane], self.max_green_time))
        
        # Initialize if first call
        if self.current_lane is None:
            self.current_lane = self.lane_sequence[0]
            # Use the calculated green time for the current lane
            self.time_remaining = green_times[self.current_lane]
            self.is_yellow = False
        
        # Decrement time
        self.time_remaining -= 1
        
        # Check if current phase is complete
        if self.time_remaining <= 0:
            if not self.is_yellow:
                # Switch to yellow
                self.is_yellow = True
                self.time_remaining = self.yellow_time
            else:
                # Switch to next lane
                current_idx = self.lane_sequence.index(self.current_lane)
                next_idx = (current_idx + 1) % len(self.lane_sequence)
                self.current_lane = self.lane_sequence[next_idx]
                
                # IMPORTANT: Use the computed green time for the new lane
                # Don't recompute here to ensure consistency
                self.time_remaining = green_times[self.current_lane]
                self.is_yellow = False
        # Reset the time if the calculated green time for the current lane changes significantly
        # This prevents incorrect long green phases
        elif not self.is_yellow and self.time_remaining > green_times[self.current_lane] * 1.1:
            # Only reset if the time is more than 10% higher than calculated
            self.time_remaining = green_times[self.current_lane]
        
        # Prepare signal states
        signal_states = {}
        for lane in self.lane_sequence:
            if self.is_yellow and lane == self.current_lane:
                signal_states[lane] = 'yellow'
            elif not self.is_yellow and lane == self.current_lane:
                signal_states[lane] = 'green'
            else:
                signal_states[lane] = 'red'
        
        # Calculate traffic statistics
        traffic_stats = self.calculate_traffic_statistics(traffic_data, green_times)
        
        # Add remaining time info
        result = {
            'states': signal_states,
            'current_phase': 'yellow' if self.is_yellow else 'green',
            'current_lane': self.current_lane,
            'time_remaining': self.time_remaining,
            'statistics': traffic_stats,
            'green_times': green_times
        }
        
        # Store data in history
        time_key = current_time.hour
        for lane, volume in traffic_data.items():
            self.history[time_key].append({
                'lane': lane,
                'volume': volume,
                'green_time': green_times.get(lane, 0) if not self.is_yellow else 0
            })
        
        return result
    
    def get_historical_patterns(self):
        """
        Return historical traffic and signal timing data.
        
        Returns:
            Dictionary with historical patterns
        """
        return dict(self.history)


# Example usage
if __name__ == "__main__":
    controller = SignalController()
    
    # Sample traffic data
    traffic = {
        'north': 15,
        'south': 25,
        'east': 10,
        'west': 5
    }
    
    # Sample traffic from neighboring intersections
    neighbors = {
        'north': 30,  # Heavy traffic coming from north
        'south': 15,
        'east': 5,
        'west': 10
    }
    
    # Compute optimal green times
    green_times = controller.compute_green_times(traffic, neighbors)
    
    print("Computed green times:")
    for lane, time in green_times.items():
        print(f"{lane}: {time} seconds") 