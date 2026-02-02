"""
Test script for optimized QAOA TSP solver
"""
import numpy as np
import sys
import os
import logging
import math
from typing import List, Tuple, Dict
from itertools import permutations

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.quantum_solver_optimized import (
    solve_tsp_qaoa_optimized,
    create_one_hot_encoding,
    nearest_neighbor_heuristic,
    calculate_route_cost,
    QISKIT_AVAILABLE
)

def test_small_tsp():
    """Test with a small 4-city TSP problem"""
    print("=" * 60)
    print("Testing Optimized QAOA TSP Solver")
    print("=" * 60)
    
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available. Cannot run test.")
        return
    
    # Create a simple 4-city distance matrix
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    n = len(distance_matrix)
    print(f"\n📊 Problem: {n} cities")
    print(f"Distance matrix:\n{distance_matrix}")
    
    # Calculate encoding requirements
    num_qubits = create_one_hot_encoding(n)
    print(f"\n🔬 One-hot encoding: {num_qubits} qubits (vs {int(np.ceil(np.log2(math.factorial(n))))} for factorial)")
    
    # Get classical baseline
    classical_route = nearest_neighbor_heuristic(distance_matrix)
    classical_cost = calculate_route_cost(classical_route, distance_matrix)
    print(f"\n📈 Classical (Nearest Neighbor):")
    print(f"   Route: {classical_route}")
    print(f"   Cost: {classical_cost:.2f}")
    
    # Run optimized QAOA
    print(f"\n⚛️  Running Optimized QAOA...")
    try:
        route, algorithm, metadata = solve_tsp_qaoa_optimized(
            distance_matrix,
            shots=1024,
            layers=1,
            use_gpu=False,  # Use CPU for testing
            warm_start=True
        )
        
        cost = calculate_route_cost(route, distance_matrix)
        improvement = ((classical_cost - cost) / classical_cost) * 100
        
        print(f"\n✅ Quantum Result:")
        print(f"   Algorithm: {algorithm}")
        print(f"   Route: {route}")
        print(f"   Cost: {cost:.2f}")
        print(f"   Improvement: {improvement:+.1f}%")
        print(f"\n📊 Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        # Verify route validity
        if len(set(route)) == n and route[0] == 0:
            print(f"\n✅ Route is valid!")
        else:
            print(f"\n⚠️  Route may be invalid")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_tsp()
    sys.exit(0 if success else 1)
