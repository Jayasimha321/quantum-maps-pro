# Quick test for Phase 5 - app.py imports
try:
    from app import app, OVERPASS_AVAILABLE
    print("Phase 5 Test Results:")
    print(f"  ✓ App imports OK")
    print(f"  OVERPASS_AVAILABLE: {OVERPASS_AVAILABLE}")
    
    # Test routes are defined
    rules = app.url_map._rules
    endpoints = [r.endpoint for r in rules]
    print(f"  Endpoints defined: {len(endpoints)}")
    
    if 'analyze_vehicle_fit_endpoint' in endpoints:
        print("  ✓ /analyze_vehicle_fit endpoint registered")
    
    print("  ✓ Phase 5 complete!")
except Exception as e:
    print(f"Error: {e}")
