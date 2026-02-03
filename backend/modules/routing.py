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

# ============================================================================
# PHASE 1: SEGMENT METADATA EXTRACTION
# ============================================================================

# ORS waytype codes to road type names
WAYTYPE_CODES = {
    0: 'unknown',
    1: 'state_road',
    2: 'road',
    3: 'street',
    4: 'path',
    5: 'track',
    6: 'cycleway',
    7: 'footway',
    8: 'steps',
    9: 'ferry',
    10: 'construction'
}

# ORS surface codes to surface names
SURFACE_CODES = {
    0: 'unknown',
    1: 'paved',
    2: 'unpaved',
    3: 'asphalt',
    4: 'concrete',
    5: 'cobblestone',
    6: 'metal',
    7: 'wood',
    8: 'compacted_gravel',
    9: 'fine_gravel',
    10: 'gravel',
    11: 'dirt',
    12: 'ground',
    13: 'ice',
    14: 'paving_stones',
    15: 'sand',
    16: 'woodchips',
    17: 'grass',
    18: 'grass_paver'
}

# Road type to estimated width (meters)
ROAD_WIDTH_ESTIMATES = {
    'state_road': 3.5,
    'road': 3.2,
    'street': 2.8,
    'path': 1.8,
    'track': 2.2,
    'cycleway': 1.5,
    'footway': 1.2,
    'steps': 1.0,
    'ferry': 10.0,
    'construction': 2.5,
    'unknown': 2.5
}

def extract_segment_metadata(ors_response):
    """
    Extract road segment metadata from ORS response extra_info.
    Returns list of segments with road type, surface, and estimated width.
    Handles both GeoJSON format (features) and standard JSON format (routes).
    """
    try:
        # Handle both ORS response formats
        if 'features' in ors_response:
            # GeoJSON format
            properties = ors_response['features'][0]['properties']
        elif 'routes' in ors_response:
            # Standard JSON format
            properties = ors_response['routes'][0]
        else:
            logging.warning(f"Unknown ORS response format. Keys: {list(ors_response.keys())}")
            return {'segments': [], 'total_segments': 0, 'has_waytypes': False, 'has_surface': False}
        
        extras = properties.get('extras', {})
        
        # Extract waytype info (note: ORS uses 'waytype' singular)
        waytypes_info = extras.get('waytype', {}).get('values', [])
        surface_info = extras.get('surface', {}).get('values', [])
        
        segments = []
        
        # Process waytypes
        for waytype_entry in waytypes_info:
            start_idx, end_idx, waytype_code = waytype_entry
            road_type = WAYTYPE_CODES.get(waytype_code, 'unknown')
            
            segments.append({
                'start_index': start_idx,
                'end_index': end_idx,
                'road_type': road_type,
                'waytype_code': waytype_code,
                'estimated_width': ROAD_WIDTH_ESTIMATES.get(road_type, 2.5),
                'surface': 'unknown'  # Will be updated below
            })
        
        # Merge surface info into segments
        for surface_entry in surface_info:
            start_idx, end_idx, surface_code = surface_entry
            surface_name = SURFACE_CODES.get(surface_code, 'unknown')
            
            # Find matching segment and update surface
            for segment in segments:
                if segment['start_index'] <= start_idx and segment['end_index'] >= end_idx:
                    segment['surface'] = surface_name
                    break
            else:
                # Surface covers different range - add as separate entry
                segments.append({
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'road_type': 'unknown',
                    'surface': surface_name,
                    'estimated_width': 2.5
                })
        
        # Sort by start index
        segments.sort(key=lambda x: x['start_index'])
        
        return {
            'segments': segments,
            'total_segments': len(segments),
            'has_waytypes': len(waytypes_info) > 0,
            'has_surface': len(surface_info) > 0
        }
        
    except (KeyError, IndexError) as e:
        logging.warning(f"Could not extract segment metadata: {e}")
        return {
            'segments': [],
            'total_segments': 0,
            'has_waytypes': False,
            'has_surface': False
        }

def decode_polyline(polyline_str):
    """Decodes a Polyline string into a list of lat/lng dicts."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    length = len(polyline_str)

    while index < length:
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0

            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20:
                    break

            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)

        lat += changes['latitude']
        lng += changes['longitude']

        coordinates.append([lng / 100000.0, lat / 100000.0])

    return coordinates

def _parse_single_route(geometry, summary, segments, data):
    """Helper to parse a single route object into consistent format."""
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
    for segment in segments:
        for step in segment['steps']:
            instructions.append({
                'instruction': step['instruction'],
                'distance': step['distance'],
                'duration': step['duration'],
                'type': step['type'],
                'way_points': step['way_points'], # Indices into the geometry array for this segment
                'name': step.get('name', 'Unnamed Road')
            })
    
    # Extract segment metadata from extra_info (if available)
    # Note: 'data' passed here should be the root response for global metadata, 
    # but per-route metadata might be in 'extras' inside the route?
    # ORS usually returns extras at segment level.
    # We'll use the existing global extractor for now, but apply it carefully.
    # Actually, existing extract_segment_metadata parses 'features' or 'routes' from root.
    # If we have multiple routes, we need to extract metadata SPECIFIC to this route.
    # For now, let's keep it simple and re-use global metadata extractor if possible, 
    # or just skip enhanced metadata for alternatives if complex.
    # BUT, vehicle fit needs metadata!
    # Let's inspect extract_segment_metadata later.
    segment_metadata = extract_segment_metadata(data) 
    
    return {
        'route_points': unique_points,
        'distance_km': distance_meters / 1000.0,
        'duration_min': duration_seconds / 60.0,
        'instructions': instructions,
        'segment_metadata': segment_metadata
    }

def get_route_from_ors(start_coords, end_coords, transport_profile, config, logger, alternatives=False):
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
        'geometry': 'true',
        # Request extra road info for vehicle fit analysis (ORS uses singular 'waytype')
        'extra_info': ['waytype', 'surface', 'roadaccessrestrictions']
    }
    
    if alternatives:
        payload['alternative_routes'] = {'target_count': 3}
    
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
        
        # Handle both GeoJSON and standard JSON formats from ORS
        parsed_routes = []
        
        raw_routes = []
        
        if 'features' in data:
            # GeoJSON format (List of features)
            raw_routes = data['features']
            for feature in raw_routes:
                geometry = feature['geometry']['coordinates']
                route_props = feature['properties']
                summary = route_props['summary']
                segments = route_props.get('segments', [])
                
                # Mock a data object for metadata extraction (hacky but reuses existing logic)
                mock_data = {'features': [feature]}
                
                parsed_routes.append(_parse_single_route(geometry, summary, segments, mock_data))
                
        elif 'routes' in data:
            # Standard JSON format (List of routes)
            raw_routes = data['routes']
            for route in raw_routes:
                # Handle geometry: standard JSON often returns encoded polyline string
                raw_geometry = route.get('geometry')
                if isinstance(raw_geometry, str):
                    geometry = decode_polyline(raw_geometry)
                else:
                    geometry = raw_geometry if raw_geometry else []
                    
                summary = route['summary']
                segments = route.get('segments', [])
                 
                # Mock data for metadata
                mock_data = {'routes': [route]}
                
                parsed_routes.append(_parse_single_route(geometry, summary, segments, mock_data))
        else:
            logger.error(f"Unknown ORS response format. Keys: {list(data.keys())}")
            return None
            
        if alternatives:
            return parsed_routes
        else:
            return parsed_routes[0] if parsed_routes else None
        
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


# ============================================================================
# PHASE 3: SEGMENT-LEVEL CONSTRAINT CHECKING
# ============================================================================

def analyze_vehicle_fit_v2(vehicle_dimensions, route_points, segment_metadata, osm_constraints=None):
    """
    Enhanced vehicle fit analysis with segment-level checking.
    Uses ORS segment metadata (waytypes) and Overpass OSM constraints.
    
    Args:
        vehicle_dimensions: dict with 'width', 'height', 'weight' (optional)
        route_points: list of {'lat': float, 'lng': float}
        segment_metadata: output from extract_segment_metadata()
        osm_constraints: output from overpass_client.find_constraints_on_route()
    
    Returns:
        dict with fits, violations, summary
    """
    if not vehicle_dimensions:
        return {'fits': True, 'violations': [], 'summary': {}}
    
    # Extract vehicle dimensions with defaults
    vehicle_width = vehicle_dimensions.get('width', 1.8)
    vehicle_height = vehicle_dimensions.get('height', 1.5)
    vehicle_weight = vehicle_dimensions.get('weight', 0)  # tons
    
    # Safety buffer (30cm on each side for maneuvering)
    SAFETY_BUFFER = 0.3
    required_width = vehicle_width + SAFETY_BUFFER
    
    violations = []
    high_severity_count = 0
    
    # --- Check ORS segment metadata (estimated road widths) ---
    segments = segment_metadata.get('segments', []) if segment_metadata else []
    
    for segment in segments:
        road_type = segment.get('road_type', 'unknown')
        estimated_width = segment.get('estimated_width', 2.5)
        
        # Width violation
        if required_width > estimated_width:
            severity = 'high' if required_width > estimated_width + 0.5 else 'medium'
            violations.append({
                'type': 'width',
                'source': 'ors_segment',
                'segment_index': segment.get('start_index'),
                'road_type': road_type,
                'vehicle_needs': required_width,
                'road_has': estimated_width,
                'severity': severity,
                'message': f"Vehicle ({vehicle_width}m) too wide for {road_type} ({estimated_width}m est.)"
            })
            if severity == 'high':
                high_severity_count += 1
        
        # Surface warning (unpaved roads for heavy vehicles)
        surface = segment.get('surface', 'unknown')
        if surface in ['unpaved', 'gravel', 'dirt', 'ground', 'sand'] and vehicle_weight > 3.5:
            violations.append({
                'type': 'surface',
                'source': 'ors_segment',
                'segment_index': segment.get('start_index'),
                'surface': surface,
                'severity': 'low',
                'message': f"Heavy vehicle ({vehicle_weight}t) on {surface} road may cause issues"
            })
    
    # --- Check Overpass OSM constraints (actual measured limits) ---
    if osm_constraints:
        for constraint in osm_constraints:
            osm_maxheight = constraint.get('maxheight')
            osm_maxwidth = constraint.get('maxwidth')
            osm_maxweight = constraint.get('maxweight')
            constraint_name = constraint.get('name', 'Unknown location')
            is_bridge = constraint.get('is_bridge', False)
            is_tunnel = constraint.get('is_tunnel', False)
            
            location_type = 'bridge' if is_bridge else ('tunnel' if is_tunnel else 'road')
            
            # Height violation (most critical for bridges/tunnels)
            if osm_maxheight and vehicle_height > osm_maxheight:
                violations.append({
                    'type': 'height',
                    'source': 'osm',
                    'osm_id': constraint.get('osm_id'),
                    'location': constraint_name,
                    'location_type': location_type,
                    'vehicle_needs': vehicle_height,
                    'limit': osm_maxheight,
                    'severity': 'critical',
                    'message': f"Vehicle height ({vehicle_height}m) exceeds {location_type} limit ({osm_maxheight}m) at {constraint_name}"
                })
                high_severity_count += 1
            
            # Width violation from OSM
            if osm_maxwidth and vehicle_width > osm_maxwidth:
                violations.append({
                    'type': 'width',
                    'source': 'osm',
                    'osm_id': constraint.get('osm_id'),
                    'location': constraint_name,
                    'vehicle_needs': vehicle_width,
                    'limit': osm_maxwidth,
                    'severity': 'high',
                    'message': f"Vehicle width ({vehicle_width}m) exceeds limit ({osm_maxwidth}m) at {constraint_name}"
                })
                high_severity_count += 1
            
            # Weight violation
            if osm_maxweight and vehicle_weight > osm_maxweight:
                violations.append({
                    'type': 'weight',
                    'source': 'osm',
                    'osm_id': constraint.get('osm_id'),
                    'location': constraint_name,
                    'vehicle_needs': vehicle_weight,
                    'limit': osm_maxweight,
                    'severity': 'high',
                    'message': f"Vehicle weight ({vehicle_weight}t) exceeds limit ({osm_maxweight}t) at {constraint_name}"
                })
                high_severity_count += 1
    
    # --- Build summary ---
    fits = high_severity_count == 0
    
    return {
        'fits': fits,
        'violations': violations,
        'summary': {
            'total_violations': len(violations),
            'high_severity': high_severity_count,
            'critical_count': len([v for v in violations if v.get('severity') == 'critical']),
            'segments_checked': len(segments),
            'osm_constraints_checked': len(osm_constraints) if osm_constraints else 0,
            'can_generate_alternative': not fits
        },
        'vehicle_dimensions': vehicle_dimensions
    }


# ============================================================================
# PHASE 4: INTELLIGENT ALTERNATIVE GENERATION
# ============================================================================

def generate_vehicle_safe_alternatives(origin, destination, vehicle_dimensions, violations, config, logger):
    """
    Generate alternative routes that avoid problematic segments.
    Uses ORS avoid_features to route around issues detected in Phase 3.
    
    Args:
        origin: {'lat': float, 'lng': float}
        destination: {'lat': float, 'lng': float}
        vehicle_dimensions: dict with 'width', 'height', 'weight'
        violations: list of violation dicts from analyze_vehicle_fit_v2
        config: app config dict
        logger: logging instance
    
    Returns:
        list of alternative route dicts with avoidance info
    """
    if not violations:
        return []
    
    vehicle_width = vehicle_dimensions.get('width', 1.8)
    vehicle_height = vehicle_dimensions.get('height', 1.5)
    vehicle_weight = vehicle_dimensions.get('weight', 0)
    
    # Determine what features to avoid based on violations and vehicle size
    avoid_features = []
    avoid_reasons = []
    
    # Check violation types and vehicle dimensions
    has_height_violation = any(v['type'] == 'height' for v in violations)
    has_width_violation = any(v['type'] == 'width' for v in violations)
    has_weight_violation = any(v['type'] == 'weight' for v in violations)
    has_surface_issue = any(v['type'] == 'surface' for v in violations)
    
    # Height issues -> avoid tunnels (most common height restriction)
    if has_height_violation or vehicle_height > 3.5:
        avoid_features.append('tunnels')
        avoid_reasons.append(f"Avoiding tunnels (vehicle height: {vehicle_height}m)")
    
    # Width issues on narrow roads -> avoid unpaved (often narrower)
    if has_width_violation or vehicle_width > 2.5:
        avoid_features.append('unpavedroads')
        avoid_reasons.append(f"Avoiding unpaved roads (vehicle width: {vehicle_width}m)")
    
    # Heavy vehicles -> avoid certain road types
    if has_weight_violation or vehicle_weight > 7.5:
        # Avoid ferries (weight restrictions) and tracks (not built for heavy loads)
        avoid_features.append('ferries')
        avoid_features.append('tracks')
        avoid_reasons.append(f"Avoiding ferries/tracks (vehicle weight: {vehicle_weight}t)")
    
    # Surface issues with heavy vehicles
    if has_surface_issue:
        avoid_features.append('unpavedroads')
        if 'Avoiding unpaved roads' not in str(avoid_reasons):
            avoid_reasons.append("Avoiding unpaved roads due to surface issues")
    
    # Very wide vehicles should also avoid steps and narrow paths
    if vehicle_width > 3.0:
        avoid_features.append('steps')
        avoid_features.append('fords')
        avoid_reasons.append(f"Avoiding steps/fords (very wide vehicle: {vehicle_width}m)")
    
    # Remove duplicates
    avoid_features = list(set(avoid_features))
    
    if not avoid_features:
        logger.info("No avoidance features determined from violations")
        return []
    
    logger.info(f"Generating alternative route avoiding: {avoid_features}")
    
    # Build ORS request with avoidance
    alternatives = []
    
    # Try different avoidance combinations
    avoidance_strategies = [
        {'features': avoid_features, 'name': 'full_avoidance'},
        {'features': avoid_features[:2] if len(avoid_features) > 2 else avoid_features, 'name': 'partial_avoidance'},
    ]
    
    for strategy in avoidance_strategies:
        try:
            alt_route = request_route_with_avoidance(
                origin, destination, 
                strategy['features'], 
                config, logger
            )
            
            if alt_route:
                alternatives.append({
                    'route': alt_route,
                    'strategy': strategy['name'],
                    'avoided_features': strategy['features'],
                    'reasons': avoid_reasons
                })
        except Exception as e:
            logger.warning(f"Failed to generate alternative with {strategy['name']}: {e}")
    
    return alternatives


def request_route_with_avoidance(origin, destination, avoid_features, config, logger, alternatives=False):
    """
    Request a route from ORS with specific features to avoid.
    """
    try:
        # Build transport profile - usually driving-car
        transport_profile = config['ORS_PROFILES'].get('driving', 'driving-car')
        
        headers = {
            'Authorization': config.get('ORS_API_KEY', ''),
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        payload = {
            'coordinates': [
                [origin['lng'], origin['lat']],
                [destination['lng'], destination['lat']]
            ],
            'instructions': 'true',
            'preference': 'recommended',
            'units': 'km',
            'geometry': 'true',
            'extra_info': ['waytype', 'surface'],
            'options': {
                'avoid_features': avoid_features
            }
        }
        
        if alternatives:
            payload['alternative_routes'] = {'target_count': 3}
            
        url = f"https://api.openrouteservice.org/v2/directions/{transport_profile}"
        # logger.info(f"Requesting avoidance route (avoiding {avoid_features})")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            logger.warning(f"ORS avoidance route failed: {response.status_code}")
            return None
        
        data = response.json()
        
        # Use common parsing logic if possible, or replicate for this specific function?
        # Actually, this function largely duplicates get_route_from_ors logic but adds 'options'.
        # We should ideally unify, but to avoid breaking things, let's just handle parsing here too.
        
        if alternatives:
             # Parse multiple routes
            parsed_routes = []
            if 'features' in data:
                 for feature in data['features']:
                     # Extract logic similar to _parse_single_route
                     geometry = feature['geometry']['coordinates']
                     route_props = feature['properties']
                     summary = route_props['summary']
                     segments = route_props.get('segments', [])
                     mock_data = {'features': [feature]}
                     parsed_routes.append(_parse_single_route(geometry, summary, segments, mock_data))
            elif 'routes' in data:
                 for route in data['routes']:
                     # Standard JSON parsing
                     raw_geometry = route.get('geometry')
                     if isinstance(raw_geometry, str):
                        geometry = decode_polyline(raw_geometry)
                     else:
                        geometry = raw_geometry if raw_geometry else []
                     summary = route['summary']
                     segments = route.get('segments', [])
                     mock_data = {'routes': [route]}
                     parsed_routes.append(_parse_single_route(geometry, summary, segments, mock_data))
            
            return parsed_routes
                     
        
        # Single route parsing (existing logic)
        geometry = []
        segments = []
        summary = {}
        
        if 'features' in data:
            # GeoJSON format
            geometry = data['features'][0]['geometry']['coordinates']
            route_props = data['features'][0]['properties']
            summary = route_props['summary']
            segments = route_props.get('segments', [])
        elif 'routes' in data:
            # Standard JSON format
            route_data = data['routes'][0]
            
            # Handle geometry: standard JSON often returns encoded polyline string
            raw_geometry = route_data.get('geometry')
            if isinstance(raw_geometry, str):
                geometry = decode_polyline(raw_geometry)
            else:
                geometry = raw_geometry if raw_geometry else []
            
            summary = route_data['summary']
            segments = route_data.get('segments', [])
        else:
            logger.warning(f"Unknown ORS response format. Keys: {list(data.keys())}")
            return None
        
        route_points = [{'lat': lat, 'lng': lng} for lng, lat in geometry]
        
        # Extract segment metadata for the alternative
        segment_metadata = extract_segment_metadata(data)
        
        return {
            'route_points': route_points,
            'distance_km': summary['distance'] / 1000.0,
            'duration_min': summary['duration'] / 60.0,
            'segment_metadata': segment_metadata,
            'avoided': avoid_features
        }
        
    except Exception as e:
        logger.error(f"Error requesting avoidance route: {e}")
        return None



def get_recommended_avoidance(vehicle_dimensions):
    """
    Get recommended avoid_features based purely on vehicle dimensions.
    Useful for proactive routing before violations are detected.
    
    Args:
        vehicle_dimensions: dict with 'width', 'height', 'weight'
    
    Returns:
        list of recommended avoid_features
    """
    avoid = []
    
    width = vehicle_dimensions.get('width', 1.8)
    height = vehicle_dimensions.get('height', 1.5)
    weight = vehicle_dimensions.get('weight', 0)
    
    # Height-based recommendations
    if height > 4.0:
        avoid.extend(['tunnels'])
    
    # Width-based recommendations
    if width > 2.5:
        avoid.extend(['unpavedroads', 'tracks'])
    if width > 3.0:
        avoid.extend(['steps', 'fords'])
    
    # Weight-based recommendations
    if weight > 7.5:
        avoid.extend(['ferries'])
    if weight > 12:
        avoid.extend(['tracks'])
    
    return list(set(avoid))


def get_avoidances_from_violations(violations):
    """
    Determine ORS avoid_features from a list of violations.
    
    Args:
        violations: list of violation dicts
        
    Returns:
        list of feature strings to avoid
    """
    avoid = []
    
    for v in violations:
        v_type = v.get('type')
        v_source = v.get('source', '')
        v_loc_type = v.get('location_type', '')
        
        # Height violations
        if v_type == 'height':
            avoid.append('tunnels') # Tunnels are the main avoidable height constraint
            
        # Width violations
        elif v_type == 'width':
            if 'unpaved' in v.get('message', '').lower() or 'path' in v.get('road_type', '').lower():
                avoid.append('unpavedroads')
            if 'track' in v.get('road_type', '').lower():
                avoid.append('tracks')
            if v_loc_type == 'bridge':
                # Can't specifically avoid bridges in ORS, but maybe avoid unpaved/tracks helps
                pass
                
        # Weight violations
        elif v_type == 'weight':
            avoid.append('ferries')
            avoid.append('tracks')
            
        # Surface violations
        elif v_type == 'surface':
            avoid.append('unpavedroads')
            
    return list(set(avoid))


def find_safe_route(origin, destination, vehicle_dimensions, config, logger, max_attempts=3):
    """
    Fetch alternative routes from ORS and analyze them for vehicle fit.
    Returns ALL valid routes found (safe or not).
    
    Args:
        origin: {'lat': float, 'lng': float}
        destination: {'lat': float, 'lng': float}
        vehicle_dimensions: dict with constraints
        config: config dict
        logger: logger instance
        max_attempts: ignored (kept for compatibility signature)
        
    Returns:
        dict with success, routes (list)
    """
    # Import Overpass client here to avoid circular imports
    try:
        from modules.overpass_client import get_road_constraints_along_route, find_constraints_on_route
        OVERPASS_AVAILABLE = True
    except ImportError:
        OVERPASS_AVAILABLE = False
        logger.warning("Overpass client not available")

    # Request alternatives from ORS
    logger.info("Requesting 3 alternative routes from ORS...")
    
    # We use empty avoid_features to get standard alternatives
    routes = request_route_with_avoidance(origin, destination, [], config, logger, alternatives=True)
    
    if not routes or not isinstance(routes, list):
        logger.warning("ORS returned no alternatives.")
        return {'success': False, 'routes': []}
        
    logger.info(f"ORS returned {len(routes)} routes. Analyzing fit...")
    
    analyzed_routes = []
    
    for i, route in enumerate(routes):
        route_points = route['route_points']
        segment_metadata = route['segment_metadata']
        
        # Get Overpass constraints
        osm_constraints = []
        if OVERPASS_AVAILABLE:
            try:
                all_constraints = get_road_constraints_along_route(route_points)
                osm_constraints = find_constraints_on_route(route_points, all_constraints)
            except Exception as e:
                logger.warning(f"Overpass query failed for route {i}: {e}")
        
        # Analyze Fit
        fit_result = analyze_vehicle_fit_v2(vehicle_dimensions, route_points, segment_metadata, osm_constraints)
        
        route['fit_analysis'] = fit_result
        route['is_safe'] = fit_result['fits']
        route['id'] = i  # Simple ID
        
        analyzed_routes.append(route)
        
    # Check if we have at least one safe route
    safe_count = len([r for r in analyzed_routes if r['is_safe']])
    logger.info(f"Analysis complete. Safe routes found: {safe_count}/{len(analyzed_routes)}")
    
    return {
        'success': safe_count > 0,
        'routes': analyzed_routes,
        'count': len(analyzed_routes)
    }