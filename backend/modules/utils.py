# backend/modules/utils.py
import json
import numpy as np
from flask import jsonify

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)

def safe_jsonify(data):
    return json.dumps(data, cls=NumpyJSONEncoder, ensure_ascii=False)

def validate_request_data(data, config):
    if not isinstance(data, dict):
        return False, "Invalid request format"

    locations = data.get('locations', [])
    
    if not locations:
        from_loc = data.get('from')
        to_loc = data.get('to')
        waypoints = data.get('waypoints', [])
        
        if from_loc and to_loc:
            locations = [from_loc] + waypoints + [to_loc]
        else:
            return False, "Either 'locations' array or 'from'/'to' locations required"

    if not isinstance(locations, list):
        return False, "Locations must be a list"

    if len(locations) < config['MIN_LOCATIONS']:
        return False, f"At least {config['MIN_LOCATIONS']} locations required"

    if len(locations) > config['MAX_LOCATIONS']:
        return False, f"Maximum {config['MAX_LOCATIONS']} locations allowed"

    for i, loc in enumerate(locations):
        if not isinstance(loc, dict):
            return False, f"Location {i+1} must be an object"
        
        if 'lat' not in loc or 'lng' not in loc:
            return False, f"Location {i+1} missing lat/lng coordinates"
        
        try:
            lat, lng = float(loc['lat']), float(loc['lng'])
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return False, f"Location {i+1} has invalid coordinates"
        except (ValueError, TypeError):
            return False, f"Location {i+1} has invalid coordinate format"

    if data.get('check_vehicle_fit', False):
        vehicle_type = data.get('vehicle_type', 'sedan')
        vehicle_dimensions = data.get('vehicle_dimensions', {})
        
        if vehicle_type == 'custom':
            if not all(k in vehicle_dimensions for k in ['width', 'length', 'height']):
                return False, "Custom vehicle dimensions must include width, length, and height"
            
            try:
                width = float(vehicle_dimensions['width'])
                length = float(vehicle_dimensions['length'])
                height = float(vehicle_dimensions['height'])
                
                if (width > config['VEHICLE_CONSTRAINTS']['max_vehicle_width'] or
                    length > config['VEHICLE_CONSTRAINTS']['max_vehicle_length'] or
                    height > config['VEHICLE_CONSTRAINTS']['max_vehicle_height']):
                    return False, "Vehicle dimensions exceed maximum allowed limits"
                    
            except (ValueError, TypeError):
                return False, "Invalid vehicle dimension format"

    return True, "Valid"

def haversine_distance_km(lat1, lon1, lat2, lon2):
    try:
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        a = (np.sin(dlat/2.0)**2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return R * c
    except Exception as e:
        return 0.0