"""
Polyline decoding utilities for OpenRouteService encoded geometry
Based on Google's polyline encoding algorithm
"""

def decode_polyline(polyline_str):
    """Decode a polyline string into a list of (lat, lng) coordinates."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    
    # Coordinates have variable length when encoded, so just keep
    # track of whether we've hit the end of the string. In each
    # while loop iteration, a single coordinate is decoded.
    while index < len(polyline_str):
        # Gather lat/lon changes, store them in a dictionary to apply them later
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

        coordinates.append((lat / 100000.0, lng / 100000.0))

    return coordinates

def decode_polyline_to_geojson(polyline_str):
    """Decode polyline string to GeoJSON format coordinates."""
    coordinates = decode_polyline(polyline_str)
    # Convert to [lng, lat] format for GeoJSON
    return [[lng, lat] for lat, lng in coordinates]