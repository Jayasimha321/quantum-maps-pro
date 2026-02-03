# Quick test for Phase 3
from modules.routing import analyze_vehicle_fit_v2

# Test with a truck (3m wide, 4m tall, 10 tons)
vehicle = {'width': 3.0, 'height': 4.0, 'weight': 10}
route_points = []
segment_metadata = {
    'segments': [
        {'road_type': 'street', 'estimated_width': 2.8, 'start_index': 0},
        {'road_type': 'path', 'estimated_width': 1.8, 'start_index': 5}
    ]
}

result = analyze_vehicle_fit_v2(vehicle, route_points, segment_metadata, None)

print("Phase 3 Test Results:")
print(f"  Fits: {result['fits']}")
print(f"  Total violations: {len(result['violations'])}")
print(f"  High severity: {result['summary'].get('high_severity', 0)}")

for v in result['violations']:
    print(f"  - {v['severity']}: {v['message']}")
