# backend/config.py
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONFIG = {
    # Use environment variable for API key
    'ORS_API_KEY': os.environ.get('ORS_API_KEY', ''),
# ... rest of config.py
    # CORS settings
    'CORS_SETTINGS': {
        'ORIGINS': os.environ.get('CORS_ORIGINS', '*').split(','),
        'METHODS': ['GET', 'POST', 'OPTIONS'],
        'ALLOW_HEADERS': ['Content-Type', 'Authorization']
    },
    'ORS_PROFILES': {
        'driving': 'driving-car',
        'cycling': 'cycling-road',
        'walking': 'foot-walking'
    },
    'SPEED_KMH': {
        'walking': 5.2,
        'cycling': 16.5,
        'driving': 48.0
    },
    'TRAFFIC_FACTORS': {
        'walking': 1.0,
        'cycling': 1.05,
        'driving': 1.25
    },
    'DISTANCE_PENALTIES': {
        'walking': {'threshold': 2.0, 'penalty': 1.3},
        'cycling': {'threshold': 8.0, 'penalty': 1.15},
        'driving': {'threshold': float('inf'), 'penalty': 1.0}
    },
    'QUANTUM_SETTINGS': {
        'default_shots': 2048,         # Increased for better statistics
        'superposition_shots': 4096,   # Even more shots for superposition
        'max_reps': 3,
        'seed': 42,
        'qaoa_p': 1,                   # Reduced to p=1 for shallow circuits
        'qaoa_layers': 1,              # p=1 is optimal on noisy hardware
        'qaoa_optimizer': 'COBYLA',
        'optimizer_maxiter': 30,       # Reduced for faster optimization
        'annealing_reads': 1000,
        'annealing_chain_strength': 2.0,
        'qubo_lagrange': 10.0,
        'use_real_quantum': True,      # Enable real quantum simulation
        'use_gpu': True,               # Enable GPU acceleration (if available)
        'max_qubits': 20,              # Increased for one-hot encoding
        'backend': 'qasm_simulator',   # Qiskit backend
        'penalty_lambda': None,        # Auto-calculate (2 * max_distance)
        'warm_start': True,            # Use classical heuristic for initialization
        'optimization_level': 3,       # Maximum transpilation optimization
        # IBM Quantum Hardware settings
        'use_ibm_hardware': False,     # Enable real quantum hardware
        'resilience_level': 1,         # Error mitigation (0=none, 1=readout, 2=ZNE)
        'ibm_api_token': os.environ.get('IBM_QUANTUM_TOKEN', None)
    },
    'ROAD_NETWORK_SETTINGS': {
        'road_network_density': 0.8,
        'max_curve_factor': 1.0,
        'min_points_per_km': 5,
        'max_points_per_km': 15,
        'urban_probability': 0.7,
        'intersection_density': 0.4,
        'traffic_simulation': True
    },
    'MAX_LOCATIONS': 20,
    'MIN_LOCATIONS': 2,
    'ROUTE_VISUALIZATION': {
        'point_precision': 6,
        'max_alternative_routes': 3,
        'route_smoothing': True
    },
    'VEHICLE_CONSTRAINTS': {
        'max_vehicle_width': 4.0,
        'max_vehicle_height': 4.5,
        'max_vehicle_length': 25.0,
        'safety_buffer': 0.5,
        'min_road_width': 2.5,
        'standard_road_width': 3.5,
        'tunnel_min_height': 4.0
    }
}

VEHICLE_DIMENSIONS = {
    'compact car': {'width': 1.8, 'length': 4.5, 'height': 1.5},
    'sedan': {'width': 1.8, 'length': 4.8, 'height': 1.5},
    'midsize car': {'width': 1.85, 'length': 4.9, 'height': 1.5},
    'large car': {'width': 1.9, 'length': 5.1, 'height': 1.5},
    'suv': {'width': 2.0, 'length': 4.8, 'height': 1.8},
    'small suv': {'width': 1.85, 'length': 4.5, 'height': 1.7},
    'large suv': {'width': 2.1, 'length': 5.2, 'height': 1.9},
    'pickup truck': {'width': 2.0, 'length': 5.8, 'height': 1.8},
    'van': {'width': 2.0, 'length': 5.5, 'height': 2.0},
    'minivan': {'width': 1.95, 'length': 5.1, 'height': 1.8},
    'semi truck': {'width': 2.6, 'length': 16.5, 'height': 4.0},
    'delivery truck': {'width': 2.2, 'length': 7.0, 'height': 2.5},
    'motorcycle': {'width': 0.8, 'length': 2.2, 'height': 1.2},
    'bicycle': {'width': 0.6, 'length': 1.8, 'height': 1.1},
    'bus': {'width': 2.5, 'length': 12.0, 'height': 3.2}
}
