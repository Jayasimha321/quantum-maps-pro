import numpy as np
from math import radians, cos, sin, asin, sqrt, atan2

def interpolate_route_points(start_point, end_point, num_points=10):
    """
    Interpolate points between two coordinates using great circle path
    """
    lat1, lon1 = radians(start_point['lat']), radians(start_point['lng'])
    lat2, lon2 = radians(end_point['lat']), radians(end_point['lng'])
    
    d = 2 * asin(sqrt(
        sin((lat2 - lat1) / 2) ** 2 +
        cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    ))
    
    points = []
    for i in range(num_points + 1):
        f = i / num_points
        a = sin((1 - f) * d) / sin(d)
        b = sin(f * d) / sin(d)
        x = a * cos(lat1) * cos(lon1) + b * cos(lat2) * cos(lon2)
        y = a * cos(lat1) * sin(lon1) + b * cos(lat2) * sin(lon2)
        z = a * sin(lat1) + b * sin(lat2)
        lat = atan2(z, sqrt(x ** 2 + y ** 2))
        lon = atan2(y, x)
        
        points.append({
            'lat': np.degrees(lat),
            'lng': np.degrees(lon)
        })
    
    return points