# backend/modules/routing.py
import logging
import numpy as np
import requests
import itertools
import random
import math
from datetime import datetime
from .utils import haversine_distance_km
from .route_interpolation import interpolate_route_points

# Constants for road network simulation
ROAD_NETWORK_DENSITY = 0.8  # Higher values create more road-like paths
MAX_CURVE_FACTOR = 2.0  # Maximum curve intensity
MIN_POINTS_PER_KM = 5  # Minimum number of points per kilometer
MAX_POINTS_PER_KM = 15  # Maximum number of points per kilometer

def transport_cost_adjustment(distance_km, transport_mode, config):
    """
    Adjust distance based on transport mode and configuration settings.
    """
    try:
        penalty_config = config['DISTANCE_PENALTIES'].get(transport_mode,
                                                         config['DISTANCE_PENALTIES']['driving'])
        threshold = penalty_config['threshold']
        penalty = penalty_config['penalty']
        
        if distance_km <= threshold:
            return distance_km
        else:
            return threshold + (distance_km - threshold) * penalty
    except Exception as e:
        return distance_km

def calculate_distance_matrix(locations, transport_mode, config):
    """
    Calculate a distance matrix between all locations using either the OpenRouteService API
    or fallback to enhanced haversine distance with transport mode adjustments.
    """
    try:
        # Remove duplicate locations first
        unique_locations = []
        seen_locations = set()
        for loc in locations:
            loc_key = f"{loc['lat']:.6f},{loc['lng']:.6f}"
            if loc_key not in seen_locations:
                unique_locations.append(loc)
                seen_locations.add(loc_key)
        
        n = len(unique_locations)
        distances = np.zeros((n, n))
        
        # Try to use OpenRouteService API for more accurate distances
        use_ors = True
        if config.get('ORS_API_KEY', '') == '':
            use_ors = False
            logging.info("No ORS API key provided, using enhanced distance calculation")
        
        for i in range(n):
            for j in range(i + 1, n):
                lat1, lon1 = unique_locations[i]['lat'], unique_locations[i]['lng']
                lat2, lon2 = unique_locations[j]['lat'], unique_locations[j]['lng']
                
                if use_ors:
                    # Try to get route from ORS
                    transport_profile = config['ORS_PROFILES'].get(transport_mode, 'driving-car')
                    route_data = get_route_from_ors(
                        {'lat': lat1, 'lng': lon1},
                        {'lat': lat2, 'lng': lon2},
                        transport_profile,
                        config,
                        logging.getLogger()
                    )
                    
                    if route_data:
                        distances[i, j] = route_data['distance_km']
                        distances[j, i] = route_data['distance_km']
                        continue
                
                # Generate synthetic route when ORS is not available
                synthetic_route = generate_synthetic_route(
                    {'lat': lat1, 'lng': lon1},
                    {'lat': lat2, 'lng': lon2},
                    transport_mode
                )
                base_dist = synthetic_route['distance_km']
                adjusted_dist = transport_cost_adjustment(base_dist, transport_mode, config)
                
                distances[i, j] = adjusted_dist
                distances[j, i] = adjusted_dist
        
        return distances
    except Exception as e:
        logging.error(f"Error calculating distance matrix: {e}")
        raise

def solve_linear_route_classical(distances, start_idx=0, end_idx=None):
    """
    Solve a linear route problem using a classical nearest neighbor algorithm.
    """
    try:
        n = len(distances)
        if n < 2:
            return list(range(n))

        if end_idx is None:
            end_idx = n - 1

        if n == 2:
            return [start_idx, end_idx]

        waypoint_indices = [i for i in range(n) if i not in [start_idx, end_idx]]
        
        if not waypoint_indices:
            return [start_idx, end_idx]

        route = [start_idx]
        remaining_waypoints = waypoint_indices.copy()
        current = start_idx

        while remaining_waypoints:
            nearest = min(remaining_waypoints, key=lambda x: distances[current][x])
            route.append(nearest)
            remaining_waypoints.remove(nearest)
            current = nearest

        if end_idx not in route:
            route.append(end_idx)

        return route

    except Exception as e:
        logging.error(f"Error in classical route solver: {e}")
        raise

def solve_tsp_classical(distances):
    """
    Solve a Traveling Salesperson Problem using classical algorithms.
    """
    try:
        n = len(distances)
        if n < 2:
            return list(range(n)) + [0] if n > 0 else []

        return solve_linear_route_classical(distances, 0, n-1)

    except Exception as e:
        logging.error(f"Error in classical TSP solver: {e}")
        raise

def generate_synthetic_route(start_coords, end_coords, transport_mode):
    """
    Generate a synthetic route between two points with some realistic variation
    """
    # Calculate number of points based on distance
    base_dist = haversine_distance_km(
        start_coords['lat'], start_coords['lng'],
        end_coords['lat'], end_coords['lng']
    )
    
    # More points for longer distances
    num_points = max(10, int(base_dist * 3))
    
    # Generate interpolated points
    route_points = interpolate_route_points(start_coords, end_coords, num_points)
    
    # Add some random variation to make it look more natural
    # but only for longer distances
    if base_dist > 1.0:  # Only add variation for routes longer than 1km
        for i in range(1, len(route_points) - 1):  # Don't modify start/end points
            variation = 0.0002 * base_dist * (np.random.random() - 0.5)
            route_points[i]['lat'] += variation
            route_points[i]['lng'] += variation
    
    return {
        'route_points': route_points,
        'distance_km': base_dist,
        'duration_min': base_dist * 3  # Rough estimate of duration
    }

def get_route_from_ors(start_coords, end_coords, transport_profile, config, logger):
    """
    Get a route between two points using the OpenRouteService API.
    """
    headers = {
        'Authorization': config.get('ORS_API_KEY', ''),
        'Content-Type': 'application/json; charset=utf-8',
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8'
    }
    
    payload = {
        'coordinates': [
            [start_coords['lng'], start_coords['lat']],
            [end_coords['lng'], end_coords['lat']]
        ],
        'instructions': 'true',
        'preference': 'recommended',
        'units': 'km',
        'geometry': 'true'
    }
    
    try:
        url = f"https://api.openrouteservice.org/v2/directions/{transport_profile}"
        logger.info(f"Calling ORS API with profile: {transport_profile}")
        logger.info(f"Request payload: {payload}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"ORS API Error: {response.status_code} - {response.text}")
            logger.error(f"Request headers: {headers}")
            logger.error(f"Request payload: {payload}")
            return None
            
        try:
            response_json = response.json()
            logger.info("Successfully parsed JSON response")
        except ValueError:
            logger.error(f"Invalid JSON in response: {response.text[:500]}")
            return None
            
        response.raise_for_status()
        
        data = response.json()
        
        geometry = data['features'][0]['geometry']['coordinates']
        summary = data['features'][0]['properties']['summary']
        distance_meters = summary['distance']
        duration_seconds = summary['duration']
        
        # Remove duplicate points from route
        unique_points = []
        seen_points = set()
        for lng, lat in geometry:
            point_key = f"{lat:.6f},{lng:.6f}"
            if point_key not in seen_points:
                unique_points.append({'lat': lat, 'lng': lng})
                seen_points.add(point_key)
        
        # Extract navigation instructions
        instructions = []
        segments = data['features'][0]['properties']['segments']
        for segment in segments:
            for step in segment['steps']:
                instructions.append({
                    'instruction': step['instruction'],
                    'distance': step['distance'],
                    'duration': step['duration'],
                    'type': step['type'],
                    'way_points': step['way_points'] # Indices into the geometry array for this segment
                })
        
        return {
            'route_points': unique_points,
            'distance_km': distance_meters / 1000.0,
            'duration_min': duration_seconds / 60.0,
            'instructions': instructions
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"ORS API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"ORS API response content: {e.response.text}")
        return None
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing ORS API response: {e}")
        return None

def generate_realistic_path(start_loc, end_loc, config, logger, transport_mode='driving'):
    """
    Generate a realistic path between two points using advanced road network simulation.
    This creates curved paths that follow realistic road patterns and better simulates actual roads.
    """
    logger.info(f"Generating realistic path between points using quantum-inspired algorithm")
    
    # Extract coordinates
    start_lat, start_lng = start_loc['lat'], start_loc['lng']
    end_lat, end_lng = end_loc['lat'], end_loc['lng']
    
    # Calculate direct distance
    direct_dist = haversine_distance_km(start_lat, start_lng, end_lat, end_lng)
    
    # Calculate bearing between points
    def calculate_bearing(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360
    
    bearing = calculate_bearing(start_lat, start_lng, end_lat, end_lng)
    
    # Determine if this is likely an urban or rural route based on distance
    # Shorter distances are more likely to be urban
    is_urban = random.random() < config['ROAD_NETWORK_SETTINGS']['urban_probability'] * (1 - min(1, direct_dist / 50))
    
    # Determine number of points based on distance and urban/rural setting
    # Urban routes need more points to simulate more complex road networks
    min_points = config['ROAD_NETWORK_SETTINGS']['min_points_per_km'] * 2  # Increased density for better road following
    max_points = config['ROAD_NETWORK_SETTINGS']['max_points_per_km'] * 2.5  # Increased max density
    
    if is_urban:
        min_points *= 2  # Even more points for urban areas
        max_points *= 2
    
    # Ensure we have enough points for a realistic path
    base_points = max(10, min(200, int(direct_dist * random.uniform(min_points, max_points))))
    
    # Create path with realistic road patterns
    path_points = [{'lat': start_lat, 'lng': start_lng}]
    
    # Generate a "main direction" with slight randomness
    main_direction = bearing + random.uniform(-5, 5)  # Reduced randomness for straighter initial segments
    
    # Create a more realistic path with road-like segments
    prev_lat, prev_lng = start_lat, start_lng
    prev_bearing = main_direction
    
    # Parameters for road network simulation - smaller segments for more detailed roads
    segment_length = direct_dist / base_points
    
    # For urban areas, use a grid system with two primary directions
    # For highways/rural, use more direct routes with gentle curves
    if is_urban:
        # Urban grid pattern simulation
        grid_angle = 90  # Perpendicular streets
        use_vertical = random.random() < 0.5
        
        for i in range(base_points - 1):
            if random.random() < 0.3:  # 30% chance to switch direction
                use_vertical = not use_vertical
            
            angle = grid_angle if use_vertical else 0
            angle += random.uniform(-5, 5)  # Slight variation
            
            # Calculate next point
            angle_rad = math.radians(angle)
            dx = segment_length * math.cos(angle_rad)
            dy = segment_length * math.sin(angle_rad)
            
            # Convert to lat/lng
            next_lat = prev_lat + (dy / 111.32)  # 1 degree lat = 111.32 km
            next_lng = prev_lng + (dx / (111.32 * math.cos(math.radians(prev_lat))))
            
            path_points.append({'lat': next_lat, 'lng': next_lng})
            prev_lat, prev_lng = next_lat, next_lng
    else:
        # Highway/rural route simulation with gentle curves
        for i in range(base_points - 1):
            # Gradually adjust bearing for smooth curves
            bearing_change = random.uniform(-15, 15) * (1 - i/(base_points-1))  # Less variation near end
            current_bearing = prev_bearing + bearing_change
            
            # Calculate next point
            angle_rad = math.radians(current_bearing)
            dx = segment_length * math.sin(angle_rad)
            dy = segment_length * math.cos(angle_rad)
            
            # Convert to lat/lng
            next_lat = prev_lat + (dy / 111.32)
            next_lng = prev_lng + (dx / (111.32 * math.cos(math.radians(prev_lat))))
            
            path_points.append({'lat': next_lat, 'lng': next_lng})
            prev_lat, prev_lng = next_lat, next_lng
            prev_bearing = current_bearing
    
    # Ensure the path ends at the destination
    path_points.append({'lat': end_lat, 'lng': end_lng})
    
    return path_points

def simulate_traffic_conditions(route_points, config, time_of_day=None):
    """
    Simulate traffic conditions along a route based on time of day and historical patterns.
    Returns traffic multipliers for each route segment.
    """
    if not time_of_day:
        time_of_day = datetime.now().hour

    # Time-based traffic patterns
    rush_hours = [7, 8, 9, 16, 17, 18]  # Morning and evening rush hours
    off_peak_hours = [22, 23, 0, 1, 2, 3, 4]  # Late night/early morning

    base_multiplier = 1.0
    if time_of_day in rush_hours:
        base_multiplier = random.uniform(1.3, 2.0)  # Higher traffic during rush hours
    elif time_of_day in off_peak_hours:
        base_multiplier = random.uniform(0.8, 1.0)  # Lower traffic during off-peak
    else:
        base_multiplier = random.uniform(1.0, 1.3)  # Normal traffic

    traffic_multipliers = []
    for i in range(len(route_points) - 1):
        # Add randomness to simulate local traffic variations
        local_multiplier = base_multiplier * random.uniform(0.9, 1.1)
        
        # Simulate traffic incidents with low probability
        if random.random() < 0.05:  # 5% chance of traffic incident
            local_multiplier *= random.uniform(1.5, 2.5)
        
        traffic_multipliers.append(local_multiplier)

    return traffic_multipliers

def apply_traffic_simulation(route_data, config, time_of_day=None):
    """
    Apply traffic simulation to route data and adjust duration estimates.
    """
    if not route_data or 'route_points' not in route_data:
        return route_data

    traffic_multipliers = simulate_traffic_conditions(route_data['route_points'], config, time_of_day)
    
    # Adjust duration based on traffic
    if 'duration_min' in route_data:
        total_multiplier = sum(traffic_multipliers) / len(traffic_multipliers)
        route_data['duration_min'] *= total_multiplier
        route_data['traffic_delay_min'] = route_data['duration_min'] - (route_data['duration_min'] / total_multiplier)

    # Add traffic info to navigation instructions
    if 'instructions' in route_data:
        for i, instruction in enumerate(route_data['instructions']):
            if i < len(traffic_multipliers):
                traffic_level = 'heavy' if traffic_multipliers[i] > 1.3 else 'moderate' if traffic_multipliers[i] > 1.1 else 'light'
                instruction['traffic_level'] = traffic_level
                if traffic_multipliers[i] > 1.3:
                    instruction['traffic_warning'] = f"Expect delays due to {traffic_level} traffic"

    return route_data

def calculate_route_statistics(route_indices, locations, transport_mode, traffic_enabled, config, logger):
    """
    Calculate route statistics including points, distance, duration, and navigation instructions.
    """
    try:
        route_points = []
        total_distance = 0
        total_duration = 0
        instructions = []
        
        # Build route points from indices
        for idx in route_indices:
            if idx < len(locations):
                route_points.append(locations[idx])
        
        # Calculate route statistics
        if len(route_points) >= 2:
            # Use OpenRouteService for accurate routing if available
            if config.get('ORS_API_KEY', ''):
                for i in range(len(route_points) - 1):
                    start = route_points[i]
                    end = route_points[i + 1]
                    transport_profile = config['ORS_PROFILES'].get(transport_mode, 'driving-car')
                    
                    route_data = get_route_from_ors(
                        start, end, transport_profile, config, logger
                    )
                    
                    if route_data:
                        total_distance += route_data['distance_km']
                        total_duration += route_data['duration_min']
                        instructions.extend(route_data['instructions'])
                    else:
                        # Fallback to haversine distance
                        dist = haversine_distance_km(start['lat'], start['lng'], end['lat'], end['lng'])
                        total_distance += dist
                        speed = config['SPEED_KMH'].get(transport_mode, 48.0)
                        total_duration += dist / speed * 60
            else:
                # Fallback to haversine distance calculation
                for i in range(len(route_points) - 1):
                    start = route_points[i]
                    end = route_points[i + 1]
                    dist = haversine_distance_km(start['lat'], start['lng'], end['lat'], end['lng'])
                    total_distance += dist
                    speed = config['SPEED_KMH'].get(transport_mode, 48.0)
                    total_duration += dist / speed * 60
        
        # Apply traffic simulation if enabled
        if traffic_enabled and config.get('ROAD_NETWORK_SETTINGS', {}).get('traffic_simulation', False):
            route_data = {
                'route_points': route_points,
                'distance_km': total_distance,
                'duration_min': total_duration,
                'instructions': instructions
            }
            route_data = apply_traffic_simulation(route_data, config)
            
            total_distance = route_data['distance_km']
            total_duration = route_data['duration_min']
            instructions = route_data.get('instructions', [])
        
        # Add traffic factor for transport mode
        traffic_factor = config['TRAFFIC_FACTORS'].get(transport_mode, 1.0)
        total_duration *= traffic_factor
        
        return route_points, total_distance, total_duration, instructions
        
    except Exception as e:
        logger.error(f"Error calculating route statistics: {e}")
        return route_points, total_distance, total_duration, []

def generate_alternative_routes(distances, optimal_route, config, vehicle_constraints=None):
    """
    Generate alternative routes by making small modifications to the optimal route.
    """
    try:
        alternative_routes = []
        n = len(distances)
        
        if n < 3:
            return alternative_routes
        
        # Generate 2-3 alternative routes
        num_alternatives = min(3, max(1, n - 2))
        
        for alt_num in range(num_alternatives):
            # Create a modified route
            modified_route = optimal_route.copy()
            
            # Apply different modification strategies
            if alt_num == 0 and len(modified_route) > 4:
                # Swap two non-adjacent waypoints
                swap_idx1 = random.randint(1, len(modified_route) - 3)
                swap_idx2 = random.randint(swap_idx1 + 1, len(modified_route) - 2)
                modified_route[swap_idx1], modified_route[swap_idx2] = modified_route[swap_idx2], modified_route[swap_idx1]
            
            elif alt_num == 1 and len(modified_route) > 3:
                # Reverse a segment
                start_idx = random.randint(1, len(modified_route) - 3)
                end_idx = random.randint(start_idx + 1, len(modified_route) - 2)
                modified_route[start_idx:end_idx+1] = reversed(modified_route[start_idx:end_idx+1])
            
            elif len(modified_route) > 2:
                # Insert a waypoint at a different position
                if len(modified_route) > 3:
                    waypoint_idx = random.randint(1, len(modified_route) - 2)
                    new_pos = random.randint(1, len(modified_route) - 1)
                    waypoint = modified_route.pop(waypoint_idx)
                    modified_route.insert(new_pos, waypoint)
            
            # Calculate route distance
            route_distance = 0
            for i in range(len(modified_route) - 1):
                if modified_route[i] < n and modified_route[i+1] < n:
                    route_distance += distances[modified_route[i]][modified_route[i+1]]
            
            alternative_routes.append({
                'route': modified_route,
                'distance': route_distance,
                'variation_type': ['waypoint_swap', 'segment_reverse', 'waypoint_reorder'][alt_num % 3]
            })
        
        return alternative_routes
        
    except Exception as e:
        logging.error(f"Error generating alternative routes: {e}")
        return []

def analyze_vehicle_fit(vehicle_dimensions, route_points, config):
    """
    Analyze if a vehicle can fit along a given route based on dimensions and road constraints.
    """
    try:
        if not vehicle_dimensions or not route_points:
            return {'fits': True, 'warnings': [], 'constraints': {}}
        
        # Extract vehicle dimensions
        vehicle_width = vehicle_dimensions.get('width', 1.8)
        vehicle_height = vehicle_dimensions.get('height', 1.5)
        vehicle_length = vehicle_dimensions.get('length', 4.8)
        
        # Get configuration constraints
        max_width = config['VEHICLE_CONSTRAINTS']['max_vehicle_width']
        max_height = config['VEHICLE_CONSTRAINTS']['max_vehicle_height']
        max_length = config['VEHICLE_CONSTRAINTS']['max_vehicle_length']
        min_road_width = config['VEHICLE_CONSTRAINTS']['min_road_width']
        tunnel_min_height = config['VEHICLE_CONSTRAINTS']['tunnel_min_height']
        
        warnings = []
        fits = True
        
        # Check against maximum vehicle dimensions
        if vehicle_width > max_width:
            warnings.append(f"Vehicle width ({vehicle_width}m) exceeds maximum allowed ({max_width}m)")
            fits = False
        
        if vehicle_height > max_height:
            warnings.append(f"Vehicle height ({vehicle_height}m) exceeds maximum allowed ({max_height}m)")
            fits = False
        
        if vehicle_length > max_length:
            warnings.append(f"Vehicle length ({vehicle_length}m) exceeds maximum allowed ({max_length}m)")
            fits = False
        
        # Check road width requirements
        required_road_width = vehicle_width + config['VEHICLE_CONSTRAINTS']['safety_buffer']
        if required_road_width > min_road_width:
            warnings.append(f"Vehicle requires wider roads ({required_road_width}m vs {min_road_width}m)")
        
        # Check tunnel clearance
        if vehicle_height > tunnel_min_height:
            warnings.append(f"Vehicle may not fit in tunnels (height {vehicle_height}m vs {tunnel_min_height}m)")
        
        return {
            'fits': fits,
            'warnings': warnings,
            'constraints': {
                'max_width': max_width,
                'max_height': max_height,
                'max_length': max_length,
                'min_road_width': min_road_width,
                'tunnel_min_height': tunnel_min_height
            },
            'vehicle_dimensions': vehicle_dimensions
        }
        
    except Exception as e:
        logging.error(f"Error analyzing vehicle fit: {e}")
        return {'fits': True, 'warnings': [str(e)], 'constraints': {}}