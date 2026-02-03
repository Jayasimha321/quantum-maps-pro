# backend/modules/overpass_client.py
"""
Phase 2: Overpass API Integration
Query OpenStreetMap for actual road constraints (maxheight, maxwidth, maxweight)
along a route for vehicle fit analysis.
"""

import requests
import logging
import time
from functools import lru_cache

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'

# Cache TTL for constraint data (5 minutes in production)
CACHE_TTL = 300

# Simple in-memory cache with timestamps
_constraint_cache = {}

def get_road_constraints_along_route(route_points, buffer_meters=100):
    """
    Query Overpass API for actual road constraints along route.
    Returns maxheight, maxwidth, maxweight for bridges/tunnels in the bounding box.
    
    Args:
        route_points: List of {'lat': float, 'lng': float} dicts
        buffer_meters: Buffer around bounding box (default 100m)
    
    Returns:
        List of constraint dicts with OSM data
    """
    if not route_points or len(route_points) < 2:
        return []
    
    # Build bounding box from route
    lats = [p['lat'] for p in route_points]
    lngs = [p['lng'] for p in route_points]
    
    # Add small buffer (roughly 0.001 degree = 111m)
    buffer_deg = buffer_meters / 111000
    min_lat = min(lats) - buffer_deg
    max_lat = max(lats) + buffer_deg
    min_lng = min(lngs) - buffer_deg
    max_lng = max(lngs) + buffer_deg
    
    # Create cache key from rounded bbox
    cache_key = f"{min_lat:.3f},{min_lng:.3f},{max_lat:.3f},{max_lng:.3f}"
    
    # Check cache
    if cache_key in _constraint_cache:
        cached_data, timestamp = _constraint_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logging.info(f"Using cached Overpass data for bbox {cache_key}")
            return cached_data
    
    # Overpass QL query for road restrictions
    query = f'''
    [out:json][timeout:8];
    (
      // Ways with height restrictions
      way["maxheight"]({min_lat},{min_lng},{max_lat},{max_lng});
      
      // Ways with width restrictions
      way["maxwidth"]({min_lat},{min_lng},{max_lat},{max_lng});
      
      // Ways with weight restrictions
      way["maxweight"]({min_lat},{min_lng},{max_lat},{max_lng});
      
      // Bridges with height restrictions
      way["bridge"]["maxheight"]({min_lat},{min_lng},{max_lat},{max_lng});
      
      // Tunnels with height restrictions  
      way["tunnel"]["maxheight"]({min_lat},{min_lng},{max_lat},{max_lng});
      
      // HGV restrictions
      way["hgv"]["hgv"!="yes"]({min_lat},{min_lng},{max_lat},{max_lng});
    );
    out body;
    >;
    out skel qt;
    '''
    
    try:
        logging.info(f"Querying Overpass API for constraints in bbox: {cache_key}")
        response = requests.post(
            OVERPASS_URL, 
            data={'data': query}, 
            timeout=10,
            headers={'User-Agent': 'QuantumMaps/1.0'}
        )
        response.raise_for_status()
        data = response.json()
        
        constraints = []
        nodes = {}
        
        # First pass: collect all nodes
        for element in data.get('elements', []):
            if element['type'] == 'node':
                nodes[element['id']] = {
                    'lat': element['lat'],
                    'lng': element['lon']
                }
        
        # Second pass: process ways with constraints
        for element in data.get('elements', []):
            if element['type'] == 'way':
                tags = element.get('tags', {})
                
                # Only include if it has relevant constraint tags
                if not any(k in tags for k in ['maxheight', 'maxwidth', 'maxweight', 'hgv']):
                    continue
                
                # Get way nodes to determine location
                way_nodes = element.get('nodes', [])
                way_coords = []
                for node_id in way_nodes:
                    if node_id in nodes:
                        way_coords.append(nodes[node_id])
                
                constraints.append({
                    'osm_id': element['id'],
                    'maxheight': parse_dimension(tags.get('maxheight')),
                    'maxwidth': parse_dimension(tags.get('maxwidth')),
                    'maxweight': parse_weight(tags.get('maxweight')),
                    'name': tags.get('name', 'Unnamed'),
                    'highway': tags.get('highway', 'unknown'),
                    'is_bridge': 'bridge' in tags,
                    'is_tunnel': 'tunnel' in tags,
                    'hgv_restriction': tags.get('hgv', None),
                    'coordinates': way_coords
                })
        
        logging.info(f"Found {len(constraints)} road constraints from Overpass API")
        
        # Cache the result
        _constraint_cache[cache_key] = (constraints, time.time())
        
        return constraints
        
    except requests.exceptions.Timeout:
        logging.warning("Overpass API timeout - returning empty constraints")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Overpass API error: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in Overpass query: {e}")
        return []


def parse_dimension(value):
    """
    Parse height/width string like '3.5m', '3.5', or '3m50' to float meters.
    Handles various OSM formats.
    """
    if not value:
        return None
    try:
        # Remove common suffixes
        cleaned = value.lower().replace('m', '').replace("'", '').strip()
        
        # Handle feet/inches format (e.g., "12'6")
        if "'" in value or '"' in value:
            # Convert feet to meters (rough)
            parts = value.replace('"', '').split("'")
            feet = float(parts[0]) if parts[0] else 0
            inches = float(parts[1]) if len(parts) > 1 and parts[1] else 0
            return (feet * 0.3048) + (inches * 0.0254)
        
        return float(cleaned)
    except (ValueError, IndexError):
        logging.debug(f"Could not parse dimension: {value}")
        return None


def parse_weight(value):
    """
    Parse weight string like '7.5t', '7500kg', or '7.5' to float tons.
    """
    if not value:
        return None
    try:
        cleaned = value.lower().strip()
        
        if 'kg' in cleaned:
            return float(cleaned.replace('kg', '').strip()) / 1000
        elif 't' in cleaned:
            return float(cleaned.replace('t', '').strip())
        else:
            # Assume tons if no unit
            return float(cleaned)
    except (ValueError, IndexError):
        logging.debug(f"Could not parse weight: {value}")
        return None


def find_constraints_on_route(route_points, constraints, tolerance_meters=50):
    """
    Find which constraints from Overpass are actually on or near the route.
    
    Args:
        route_points: List of {'lat': float, 'lng': float} dicts
        constraints: List of constraint dicts from get_road_constraints_along_route
        tolerance_meters: How close a constraint must be to route (default 50m)
    
    Returns:
        List of constraints that are on/near the route with distance info
    """
    if not constraints or not route_points:
        return []
    
    # Convert tolerance to approximate degrees
    tolerance_deg = tolerance_meters / 111000
    
    on_route_constraints = []
    
    for constraint in constraints:
        # Check if any constraint coordinate is near route
        constraint_coords = constraint.get('coordinates', [])
        
        for c_coord in constraint_coords:
            for r_point in route_points:
                # Simple distance check (approximate)
                lat_diff = abs(c_coord['lat'] - r_point['lat'])
                lng_diff = abs(c_coord['lng'] - r_point['lng'])
                
                if lat_diff < tolerance_deg and lng_diff < tolerance_deg:
                    on_route_constraints.append({
                        **constraint,
                        'on_route': True,
                        'nearest_route_point': r_point
                    })
                    break
            else:
                continue
            break
    
    return on_route_constraints


def clear_cache():
    """Clear the constraint cache (useful for testing)."""
    global _constraint_cache
    _constraint_cache = {}
    logging.info("Overpass constraint cache cleared")
