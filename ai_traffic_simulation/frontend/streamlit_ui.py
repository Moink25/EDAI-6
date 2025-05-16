import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import time as time_module  # Rename to avoid conflicts
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app import TrafficSimulationApp

# Set page config
st.set_page_config(
    page_title="AI Traffic Signal Simulator",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "simulation_app" not in st.session_state:
    # Default 2x2 grid (4 intersections)
    rows, cols = 2, 2
    st.session_state.simulation_app = TrafficSimulationApp(
        grid_rows=rows, grid_cols=cols
    )
    st.session_state.current_state = st.session_state.simulation_app.get_current_state()
    st.session_state.grid_layout = st.session_state.simulation_app.get_grid_layout()
    st.session_state.auto_run = False
    st.session_state.run_speed = 1.0
    st.session_state.sim_history = []

# Function to update simulation state
def update_simulation():
    new_state = st.session_state.simulation_app.step_simulation()
    st.session_state.current_state = new_state
    
    # Add to history (keep last 60 steps)
    st.session_state.sim_history.append({
        'step': new_state['step'],
        'time': new_state['time'],
        'intersections': new_state['intersections']
    })
    
    if len(st.session_state.sim_history) > 60:
        st.session_state.sim_history.pop(0)

# Function to render traffic signals
def render_traffic_light(state, turn_info=None):
    """
    Renders a traffic light emoji with optional turn direction indicators.
    
    Args:
        state: Signal state (green, yellow, red)
        turn_info: Optional dictionary with turn information
        
    Returns:
        Formatted HTML string for the traffic light
    """
    if state == 'green':
        base_light = "🟢"
    elif state == 'yellow':
        base_light = "🟡"
    else:
        base_light = "🔴"
        
    if not turn_info:
        return base_light
        
    # Add turn indicators if provided
    turn_symbols = {
        'left': '↰',
        'straight': '↑',
        'right': '↱'
    }
    
    result = f"{base_light} "
    for turn, active in turn_info.items():
        if turn in turn_symbols and active:
            result += turn_symbols[turn]
            
    return result

# Function to get color for traffic volume
def get_traffic_color(volume):
    if volume < 5:
        return "green"
    elif volume < 15:
        return "orange"
    else:
        return "red"

# Function to create a grid visualization using Plotly
def create_grid_visualization(grid_layout, current_state):
    # Create figure
    fig = go.Figure()
    
    # Extract node positions
    nodes = grid_layout['nodes']
    edges = grid_layout['edges']
    
    # Set up a clean background
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title={
            'text': 'Traffic Grid Simulation',
            'font': {'size': 18, 'color': '#1f77b4'}
        }
    )
    
    # Plot edges first (roads)
    for edge in edges:
        source = edge['source']
        target = edge['target']
        direction = edge['direction']
        
        # Find source and target nodes
        source_node = next(node for node in nodes if node['id'] == source)
        target_node = next(node for node in nodes if node['id'] == target)
        
        # Draw edge with better styling
        fig.add_trace(go.Scatter(
            x=[source_node['x'], target_node['x']],
            y=[source_node['y'], target_node['y']],
            mode='lines',
            line=dict(color='#777777', width=4),
            showlegend=False,
            hovertext=f"Road from {source} to {target} ({direction})"
        ))
    
    # Plot vehicles in transit if available
    if 'grid_state' in current_state and 'vehicles_in_transit' in current_state['grid_state']:
        vehicles_in_transit = current_state['grid_state']['vehicles_in_transit']
        
        for transit_key, transit_data in vehicles_in_transit.items():
            # Parse transit key to get source, destination, direction and turn type
            parts = transit_key.split('|')
            if len(parts) < 3:
                continue
                
            source = parts[0]
            destination = parts[1]
            direction = parts[2]
            turn_type = parts[3] if len(parts) > 3 else 'straight'
            
            # Find source and destination nodes
            source_node = next((node for node in nodes if node['id'] == source), None)
            dest_node = next((node for node in nodes if node['id'] == destination), None)
            
            if not source_node or not dest_node:
                continue
                
            # Calculate position based on remaining distance
            remaining_distance = transit_data['remaining_distance']
            total_distance = 200  # Default distance between intersections
            
            # Calculate progress percentage (0 to 1)
            progress = 1.0 - (remaining_distance / total_distance)
            progress = max(0.05, min(0.95, progress))  # Keep within road
            
            # Calculate intermediate position
            x_pos = source_node['x'] + (dest_node['x'] - source_node['x']) * progress
            y_pos = source_node['y'] + (dest_node['y'] - source_node['y']) * progress
            
            # Set vehicle marker properties based on turn type
            vehicle_color = '#00BFFF'  # Straight
            vehicle_symbol = 'circle'
            
            if turn_type == 'left':
                vehicle_color = '#FFA500'  # Left turn
                vehicle_symbol = 'triangle-left'
            elif turn_type == 'right':
                vehicle_color = '#32CD32'  # Right turn
                vehicle_symbol = 'triangle-right'
                
            # Calculate size based on number of vehicles (capped)
            size = min(25, 10 + (transit_data['vehicles'] / 2))
            
            # Add marker for vehicles in transit
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode='markers',
                marker=dict(
                    size=size,
                    color=vehicle_color,
                    symbol=vehicle_symbol,
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hoverinfo='text',
                hovertext=f"{transit_data['vehicles']} vehicles moving from {source} to {destination}<br>{turn_type.capitalize()} turn ({direction} to {destination})<br>Remaining distance: {remaining_distance:.0f}m"
            ))
    
    # Plot nodes (intersections)
    for node in nodes:
        node_id = node['id']
        x, y = node['x'], node['y']
        
        # Get intersection state if available
        node_state = current_state['intersections'].get(node_id, {})
        traffic = node_state.get('traffic', {})
        signals = node_state.get('signals', {}).get('states', {})
        
        # Create hover info
        hover_text = [f"<b>Intersection: {node_id}</b>"]
        
        for direction in ['north', 'south', 'east', 'west']:
            volume = traffic.get(direction, 0)
            signal = signals.get(direction, 'red')
            hover_text.append(f"{direction.capitalize()}: {volume} vehicles, {signal} light")
        
        # Draw intersection node
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=24,  # Increased from 20
                color='#1f77b4',
                symbol='square',
                line=dict(color='white', width=1)
            ),
            name=node_id,
            text=node_id,
            textfont=dict(color='white', size=10),  # Increased from 9
            textposition="middle center",
            hoverinfo='text',
            hovertext="<br>".join(hover_text)
        ))
        
        # Add traffic signals and volumes with improved visuals
        if traffic:
            # Position offset for directional traffic with better spacing
            dir_offsets = {
                'north': (0, 30),  # Increased from 25
                'south': (0, -30),  # Increased from -25
                'east': (30, 0),    # Increased from 25
                'west': (-30, 0)    # Increased from -25
            }
            
            # Offset markers for turn types within each direction
            turn_offsets = {
                'left': (-8, -8),   # Increased offsets
                'straight': (0, 0),
                'right': (8, 8)     # Increased offsets
            }
            
            for direction, offset in dir_offsets.items():
                volume = traffic.get(direction, 0)
                signal_state = signals.get(direction, 'red')
                
                # Set color based on signal state
                if signal_state == 'green':
                    color = '#2ca02c'  # Green
                elif signal_state == 'yellow':
                    color = '#ff7f0e'  # Yellow/Orange
                else:
                    color = '#d62728'  # Red
                
                # Add main volume circle - MAKE IT BIGGER
                fig.add_trace(go.Scatter(
                    x=[x + offset[0]],
                    y=[y + offset[1]],
                    mode='markers+text',
                    text=str(volume),
                    textposition="middle center",
                    textfont=dict(size=10, color='black'),  # Increased from 9
                    marker=dict(
                        size=22,  # Increased from 15
                        color=color,
                        symbol='circle',
                        line=dict(color='white', width=1)
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"{direction.capitalize()}: {volume} vehicles, {signal_state} light"
                ))
                
                # Add turn indicators ONLY FOR GREEN LIGHTS
                if signal_state == 'green' and 'grid_state' in current_state and 'vehicles_in_transit' in current_state['grid_state']:
                    # Turn type symbols with better visibility
                    turn_symbols = {
                        'left': '↰',
                        'straight': '↑',
                        'right': '↱'
                    }
                    
                    # Turn type colors
                    turn_colors = {
                        'left': '#FFA500',    # Orange for left
                        'straight': '#00BFFF', # Blue for straight
                        'right': '#32CD32'     # Green for right
                    }
                    
                    # Calculate direction-based offset adjustments with more space
                    if direction == 'north':
                        turn_adjusted_offsets = {
                            'left': (-15, 0),
                            'straight': (0, 15),
                            'right': (15, 0)
                        }
                    elif direction == 'south':
                        turn_adjusted_offsets = {
                            'left': (15, 0),
                            'straight': (0, -15),
                            'right': (-15, 0)
                        }
                    elif direction == 'east':
                        turn_adjusted_offsets = {
                            'left': (0, -15),
                            'straight': (15, 0),
                            'right': (0, 15)
                        }
                    else:  # west
                        turn_adjusted_offsets = {
                            'left': (0, 15),
                            'straight': (-15, 0),
                            'right': (0, -15)
                        }
                    
                    # Turn distribution based on configured probabilities
                    turn_probs = {
                        'straight': 0.7,  # 70%
                        'left': 0.15,     # 15%
                        'right': 0.15     # 15%
                    }
                    
                    # Add indicators for turn directions with better labeling
                    for turn_type, turn_symbol in turn_symbols.items():
                        turn_offset = turn_adjusted_offsets[turn_type]
                        # Calculate vehicles for this turn type
                        vehicles = int(volume * turn_probs[turn_type])
                        
                        # Only show if there are vehicles for this turn
                        if vehicles > 0:
                            # Add a clear background with the appropriate color
                            fig.add_trace(go.Scatter(
                                x=[x + offset[0] + turn_offset[0]],
                                y=[y + offset[1] + turn_offset[1]],
                                mode='markers',
                                marker=dict(
                                    size=24,
                                    color=turn_colors[turn_type],
                                    opacity=0.7,
                                    symbol='circle',
                                    line=dict(color='white', width=1)
                                ),
                                showlegend=False,
                                hoverinfo='text',
                                hovertext=f"{direction.capitalize()} {turn_type} movement: {vehicles} vehicles"
                            ))
                            
                            # Add text over the background
                            fig.add_trace(go.Scatter(
                                x=[x + offset[0] + turn_offset[0]],
                                y=[y + offset[1] + turn_offset[1]],
                                mode='text',
                                text=f"{turn_symbol}{vehicles}",
                                textposition="middle center",
                                textfont=dict(size=11, color='white', family="Arial Black"),
                                showlegend=False,
                                hoverinfo='text',
                                hovertext=f"{direction.capitalize()} {turn_type} movement: {vehicles} vehicles"
                            ))
    
    # Set layout with better dimensions and styling
    fig.update_layout(
        hovermode="closest",
        showlegend=False,
        xaxis=dict(
            title="",
            showgrid=False,
            zeroline=False,
            visible=False
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500  # Increased from 450
    )
    
    return fig

# Function to create traffic history chart
def create_traffic_history_chart():
    if not st.session_state.sim_history:
        return None
    
    # Extract data from history
    data = []
    for entry in st.session_state.sim_history:
        step = entry['step']
        
        for intersection_id, state in entry['intersections'].items():
            for direction, volume in state['traffic'].items():
                data.append({
                    'Step': step,
                    'Intersection': intersection_id,
                    'Direction': direction.capitalize(),
                    'Volume': volume
                })
    
    # Create a dataframe
    df = pd.DataFrame(data)
    
    # Create the chart
    fig = px.line(
        df, 
        x='Step', 
        y='Volume', 
        color='Direction',
        facet_col='Intersection',
        title='Traffic Volume History by Intersection and Direction',
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Simulation Step",
        yaxis_title="Vehicle Count",
        legend_title="Direction",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    
    return fig

# App header
st.title("🚦 AI-Powered Smart Traffic Signal Simulator")
st.markdown("""
This simulation demonstrates a dynamic traffic signal system that adapts to changing traffic conditions.
The system uses synthetic data generation to model realistic traffic patterns across multiple intersections.
""")

# Sidebar controls
with st.sidebar:
    st.header("Simulation Controls")
    
    # Grid configuration
    st.subheader("Grid Configuration")
    col1, col2 = st.columns(2)
    rows = col1.number_input("Rows", min_value=1, max_value=4, value=2)
    cols = col2.number_input("Columns", min_value=1, max_value=4, value=2)
    
    if st.button("Reset Simulation"):
        st.session_state.simulation_app = TrafficSimulationApp(
            grid_rows=int(rows), grid_cols=int(cols)
        )
        st.session_state.current_state = st.session_state.simulation_app.get_current_state()
        st.session_state.grid_layout = st.session_state.simulation_app.get_grid_layout()
        st.session_state.sim_history = []
    
    # Auto-run controls
    st.subheader("Auto-Run")
    auto_run = st.checkbox("Auto-run simulation", value=st.session_state.auto_run)
    run_speed = st.slider("Simulation Speed", min_value=0.1, max_value=2.0, value=st.session_state.run_speed, step=0.1)
    
    st.session_state.auto_run = auto_run
    st.session_state.run_speed = run_speed
    
    # Manual step
    if st.button("Step Simulation"):
        update_simulation()
    
    # Show simulation info
    st.subheader("Simulation Info")
    st.write(f"Current Time: {st.session_state.current_state['time']}")
    st.write(f"Current Step: {st.session_state.current_state['step']}")

# Main visualization - Using a single container for the grid
st.header("City Traffic Grid")
grid_container = st.empty()

# Initial grid visualization
grid_fig = create_grid_visualization(
    st.session_state.grid_layout,
    st.session_state.current_state
)
grid_container.plotly_chart(grid_fig, use_container_width=True)

# Intersection details
st.header("Intersection Details")
if st.session_state.current_state:
    tabs = []
    for intersection_id in st.session_state.current_state['intersections']:
        tabs.append(intersection_id)
    
    selected_tab = st.radio("Select Intersection", tabs, horizontal=True)
    
    # Display detailed info for selected intersection
    if selected_tab:
        intersection_state = st.session_state.current_state['intersections'][selected_tab]
        traffic = intersection_state.get('traffic', {})
        signals = intersection_state.get('signals', {}).get('states', {})
        statistics = intersection_state.get('signals', {}).get('statistics', {})
        
        # Display traffic info in a better format
        st.subheader(f"Traffic at {selected_tab}")
        
        # Create columns for each direction
        col1, col2, col3, col4 = st.columns(4)
        
        # Helper function to show traffic details - only show turn distribution for green signals
        def show_traffic_details(col, direction, volume, signal_state):
            with col:
                # Main traffic count
                st.metric(
                    f"{direction.capitalize()} Traffic", 
                    volume,
                    delta=None,
                    delta_color="normal"
                )
                
                # Signal state with traffic light
                if signal_state:
                    st.markdown(f"Signal: {render_traffic_light(signal_state)}")
                
                # Show turn distribution based on configured probabilities
                # ONLY FOR GREEN SIGNALS
                if signal_state == 'green':
                    st.markdown("#### Turn Distribution:")
                    turn_probs = {
                        'Straight': 0.7,  # 70%
                        'Left': 0.15,     # 15%
                        'Right': 0.15     # 15%
                    }
                    
                    # Use simple markdown formatting instead of dataframe
                    for turn_type, prob in turn_probs.items():
                        # Calculate vehicles for this turn type
                        vehicles = int(volume * prob)
                        percentage = int(prob * 100)
                        st.markdown(f"**{turn_type}**: {vehicles} vehicles ({percentage}%)")
                else:
                    # For non-green signals, show just a note
                    st.markdown("#### Turn Distribution:")
                    st.markdown("*Not active - signal is not green*")
        
        # Display traffic details for each direction
        show_traffic_details(col1, "north", traffic.get('north', 0), signals.get('north', 'red'))
        show_traffic_details(col2, "south", traffic.get('south', 0), signals.get('south', 'red'))
        show_traffic_details(col3, "east", traffic.get('east', 0), signals.get('east', 'red'))
        show_traffic_details(col4, "west", traffic.get('west', 0), signals.get('west', 'red'))
        
        # Show current signal phase information
        st.subheader("Signal Timing")
        
        if 'signals' in intersection_state:
            signal_info = intersection_state['signals']
            
            # Create columns for current phase info
            phase_col1, phase_col2, phase_col3 = st.columns(3)
            
            with phase_col1:
                st.metric("Current Phase", signal_info.get('current_phase', 'unknown').capitalize())
            
            with phase_col2:
                st.metric("Active Direction", signal_info.get('current_lane', 'unknown').capitalize())
            
            with phase_col3:
                st.metric("Time Remaining", f"{signal_info.get('time_remaining', 0)} seconds")
            
            # Display calculated green times with clearer formatting
            if 'green_times' in signal_info:
                st.subheader("Calculated Green Times")
                
                # Create a more informative display for green times
                green_time_cols = st.columns(len(signal_info['green_times']))
                
                for i, (lane, time) in enumerate(signal_info['green_times'].items()):
                    with green_time_cols[i]:
                        # Highlight the active lane
                        is_active = lane == signal_info.get('current_lane', '')
                        time_label = f"{time} s"
                        
                        # Show calculated time with badge for current active direction
                        if is_active and not signal_info.get('current_phase') == 'yellow':
                            time_remaining = signal_info.get('time_remaining', 0)
                            delta = f"{time_remaining}/{time}"
                            st.metric(
                                f"{lane.capitalize()} Green",
                                time_label,
                                delta=f"Active: {delta} s",
                                delta_color="normal"
                            )
                        else:
                            st.metric(
                                f"{lane.capitalize()} Green", 
                                time_label,
                                delta=None,
                                delta_color="normal"
                            )
            
            # Show traffic statistics if available
            if 'statistics' in signal_info:
                st.subheader("Traffic Performance")
                stats_container = st.container()
                
                with stats_container:
                    stats_cols = st.columns(len(signal_info['statistics']))
                    
                    for i, (lane, stats) in enumerate(signal_info['statistics'].items()):
                        with stats_cols[i]:
                            st.markdown(f"**{lane.capitalize()}**:")
                            st.markdown(f"V/C Ratio: {stats.get('v_c_ratio', 0):.2f}")
                            st.markdown(f"Delay: {stats.get('delay', 0):.1f} sec")
                            st.markdown(f"Level of Service: {stats.get('level_of_service', 'unknown')}")
        else:
            st.info("No signal information available for this intersection.")
        
        # Show incoming traffic from neighbors
        if 'incoming' in intersection_state:
            st.subheader("Incoming Traffic from Neighbors")
            incoming = intersection_state['incoming']
            
            if incoming:
                incoming_cols = st.columns(len(incoming))
                
                for i, (direction, volume) in enumerate(incoming.items()):
                    with incoming_cols[i]:
                        st.metric(
                            f"From {direction.capitalize()}", 
                            volume,
                            delta=None,
                            delta_color="normal"
                        )
            else:
                st.write("No incoming traffic data available")
        
        # Add explanation of signal timing formulas
        with st.expander("Signal Timing Calculation Details (Simplified)"):
            st.markdown("""
            ### How the Traffic Signals Make Decisions
            
            The traffic signals in this simulation use several smart calculations to decide how long each green light should last:
            
            1. **Optimal Cycle Length Formula**:
               
               We calculate the ideal total time for one complete cycle (all directions getting a turn) based on how busy the intersection is. When traffic is heavier, we use longer cycles.
               
            2. **Green Time Distribution**:
               
               Each direction gets green time based on its traffic volume. If the north lane has twice as many cars as the east lane, it gets approximately twice as much green time.
               
            3. **Queue Clearing**:
               
               We estimate how long it would take for all waiting vehicles to clear the intersection, considering that:
               - Cars need time to start moving (reaction time)
               - Vehicles accelerate gradually from a stop
               - Longer queues need more time to clear
               
            4. **Delay Estimation**:
               
               We calculate how long the average vehicle waits at the light, which helps us grade the intersection performance from A (excellent) to F (failing).
               
            5. **Coordination Between Intersections**:
               
               When a group of cars (platoon) is moving from one intersection to another, we coordinate the lights to create a "green wave" - allowing cars to pass through multiple intersections without stopping.
               
            6. **Learning from Patterns**:
               
               The system uses past traffic data to predict future volumes, combining current data with historical patterns to anticipate traffic changes throughout the day.
            
            The goal is to minimize the overall waiting time for all vehicles while ensuring fair access from all directions.
            """)

# Traffic history chart
st.header("Traffic History")
history_chart = create_traffic_history_chart()
if history_chart:
    st.plotly_chart(history_chart, use_container_width=True)
else:
    st.info("Run the simulation to see traffic history.")

# Auto-run the simulation if enabled
if st.session_state.auto_run:
    # Update simulation
    update_simulation()
    
    # Update the grid visualization in-place
    grid_fig = create_grid_visualization(
        st.session_state.grid_layout,
        st.session_state.current_state
    )
    grid_container.plotly_chart(grid_fig, use_container_width=True)
    
    # Adjust wait time based on speed - fix sleep function
    # Use the properly imported time module
    time_module.sleep(1.0 / st.session_state.run_speed)
    
    # Force a rerun to refresh the UI
    st.experimental_rerun()


if __name__ == "__main__":
    # This section will be ignored by Streamlit
    print("Running Streamlit UI...") 