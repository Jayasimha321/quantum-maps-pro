
"""
Binary (Logarithmic) Encoded Quantum TSP Solver
Reduces qubit requirements from O(N^2) to O(N log N).
"""

import numpy as np
import logging
import math
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from scipy.optimize import minimize
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def get_qubits_per_step(num_cities: int) -> int:
    """Bits needed to represent N cities."""
    # We encode cities 1..N-1 (0 is fixed start)
    # If N=4, cities 1,2,3 -> need ceil(log2(4)) = 2 bits
    return math.ceil(math.log2(num_cities))

def create_binary_hamiltonian(distance_matrix: np.ndarray) -> SparsePauliOp:
    """
    Create Hamiltonian using Binary (Logarithmic) Encoding.
    State |u>_t represents city u visited at time t.
    Encoding: City index u is encoded binary (e.g. 5 -> 101) in registers.
    """
    n = len(distance_matrix)
    n_steps = n - 1 # Fixed start
    qubits_per_step = get_qubits_per_step(n) 
    total_qubits = n_steps * qubits_per_step
    
    logger.info(f"Creating Binary Hamiltonian for {n} cities.")
    logger.info(f"Encoding: {n_steps} steps x {qubits_per_step} qubits = {total_qubits} total qubits.")

    pauli_dict = {}

    def get_projector(city_idx: int, step: int) -> Dict[str, float]:
        """
        Returns Pauli dictionary for projector |city_idx><city_idx| at step `step`.
        E.g. for city 2 (binary 10), we need projector |1><1| x |0><0|.
        |0><0| = (I+Z)/2, |1><1| = (I-Z)/2
        """
        # Get binary string (little endian or big endian? Let's use standard binary)
        # We target qubits: step * qubits_per_step to (step+1)*...
        start_q = step * qubits_per_step
        
        # Binary representation
        bin_str = format(city_idx - 1, f'0{qubits_per_step}b') # city indices 1..N-1 mapped to 0..M
        
        # Build tensor product of (I +/- Z)/2 terms
        # This expands to 2^qubits_per_step terms
        # To keep it efficient, we generate the string logic directly
        
        terms = {"I" * total_qubits: 1.0}
        
        for local_bit_idx, bit_char in enumerate(bin_str):
            # Target qubit index in the total register
            target_q = start_q + local_bit_idx
            
            # Term is (I + s*Z)/2 where s = 1 if bit is 0, s = -1 if bit is 1
            sign = 1.0 if bit_char == '0' else -1.0
            
            new_terms = {}
            for p_str, coeff in terms.items():
                # Term 1: 0.5 * I part (unchanged p_str)
                new_terms[p_str] = new_terms.get(p_str, 0) + coeff * 0.5
                
                # Term 2: 0.5 * sign * Z part (inject Z at target_q)
                p_list = list(p_str)
                # Apply Z: I->Z, Z->I (if already Z, it cancels? No, Z*Z=I)
                if p_list[-(target_q+1)] == 'I':
                    p_list[-(target_q+1)] = 'Z'
                else: # Already Z
                    p_list[-(target_q+1)] = 'I'
                    # No phase change for Z*Z=I
                
                z_p_str = "".join(p_list)
                new_terms[z_p_str] = new_terms.get(z_p_str, 0) + coeff * 0.5 * sign
            
            terms = new_terms
            
        return terms

    # 1. Cost Terms: Sum d(u,v) * (|u>_t <u|) * (|v>_{t+1} <v|)
    # This involves multiplying two expanded projector polynomials.
    # Warning: expenses 2^(2*logN) = N^2 terms per edge. Total N^4 terms potentially.
    # For small N (7), this is manageable.
    
    # Iterate time steps
    for t in range(n_steps - 1):
        for u in range(1, n):
            for v in range(1, n):
                if u == v: continue
                weight = distance_matrix[u][v]
                if weight == 0: continue
                
                # Get projectors
                proj_u_t = get_projector(u, t)
                proj_v_next = get_projector(v, t + 1)
                
                # Multiply and add to Hamiltonian
                for s1, c1 in proj_u_t.items():
                    for s2, c2 in proj_v_next.items():
                        # Combine strings (Product of Paulis)
                        # Since they act on different qubits (time steps), just merge Zs
                        # Strings are "IIZII..."
                        combined = []
                        for char1, char2 in zip(s1, s2):
                            if char1 == 'Z' or char2 == 'Z':
                                if char1 == 'Z' and char2 == 'Z':
                                    combined.append('I') # Z*Z = I
                                else:
                                    combined.append('Z')
                            else:
                                combined.append('I')
                        
                        final_str = "".join(combined)
                        pauli_dict[final_str] = pauli_dict.get(final_str, 0) + weight * c1 * c2

    # Add Return to Start terms (Start -> t=0, t=last -> Start)
    # Start is fixed at 0. So just d(0, u) * |u>_0 and d(v, 0) * |v>_last
    
    # Start -> First Step (t=0)
    for u in range(1, n):
        weight = distance_matrix[0][u]
        proj_u = get_projector(u, 0)
        for s, c in proj_u.items():
             pauli_dict[s] = pauli_dict.get(s, 0) + weight * c
             
    # Last Step -> Start
    for v in range(1, n):
        weight = distance_matrix[v][0]
        proj_v = get_projector(v, n_steps - 1)
        for s, c in proj_v.items():
             pauli_dict[s] = pauli_dict.get(s, 0) + weight * c

    # 2. Constraint: Each city visited exactly once.
    # Sum_t |u>_t <u| = 1 for each city u
    # Penalty: Lambda * (Sum_t P_{u,t} - 1)^2
    # This ensures every city appears in the schedule.
    # Note: "One city per time step" is GUARANTEED by the register encoding!
    # A register always has ONE value. We only need to check distinct values.
    
    lambda_p = np.max(distance_matrix) * 2
    
    for u in range(1, n):
        # Term: (Sum_t P_{u,t} - 1)^2
        # = Sum_t P_{u,t}^2 + Sum_{t!=k} P_{u,t} P_{u,k} - 2 Sum_t P_{u,t} + 1
        # P^2 = P (Projectors are idempotent).
        # = Sum_t P_{u,t} + Sum_{t!=k} P_{u,t} P_{u,k} - 2 Sum_t P_{u,t} + 1
        # = Sum_{t!=k} P_{u,t} P_{u,k} - Sum_t P_{u,t} + 1
        
        # P_{u,t} * P_{u,k} penalizes visiting u at t AND k.
        
        # Add -Sum P_{u,t}
        for t in range(n_steps):
            proj = get_projector(u, t)
            for s, c in proj.items():
                pauli_dict[s] = pauli_dict.get(s, 0) - lambda_p * c
        
        # Add Sum P_{u,t} P_{u,k}
        for t1 in range(n_steps):
            for t2 in range(t1 + 1, n_steps):
                 proj1 = get_projector(u, t1)
                 proj2 = get_projector(u, t2)
                 for s1, c1 in proj1.items():
                    for s2, c2 in proj2.items():
                        combined = []
                        for char1, char2 in zip(s1, s2):
                            if char1 == 'Z' or char2 == 'Z':
                                if char1 == 'Z' and char2 == 'Z':
                                    combined.append('I')
                                else:
                                    combined.append('Z')
                            else:
                                combined.append('I')
                        final_str = "".join(combined)
                        pauli_dict[final_str] = pauli_dict.get(final_str, 0) + lambda_p * c1 * c2

    # Cleanup small terms
    pauli_list = [(s, c) for s, c in pauli_dict.items() if abs(c) > 1e-6]
    
    return SparsePauliOp.from_list(pauli_list)

def create_qaoa_circuit_binary(num_qubits: int, gamma: float, beta: float,
                                hamiltonian: SparsePauliOp, layers: int = 1) -> QuantumCircuit:
    """Standard QAOA circuit generator."""
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.h(q)
        
    for _ in range(layers):
        qc.append(PauliEvolutionGate(hamiltonian, time=gamma), range(num_qubits))
        for q in range(num_qubits):
            qc.rx(2 * beta, q)
            
    qc.measure_all()
    return qc

def decode_binary_to_route(bitstring: str, num_cities: int) -> List[int]:
    """
    Decode binary register string to route.
    Args:
        bitstring: 'q3q2q1q0' -> we need to split into registers.
        num_cities: Total cities (N). Start city 0 is implicit.
    Returns:
        Route list including start/end 0.
    """
    n_steps = num_cities - 1
    qubits_per_step = get_qubits_per_step(num_cities)
    
    # Reverse bitstring because Qiskit returns 'qN...q0' (Little Endian)
    # But wait, typically q0 is rightmost.
    # We constructed: Reg0 = q0..qM, Reg1 = qM+1..q2M
    # So bitstring is qTotal...q0.
    # Let's reverse it to access by index easily: q0, q1, ...
    bits = bitstring[::-1]
    
    route = [0] # Fixed start
    visited = {0}
    
    for t in range(n_steps):
        # Extract bits for step t
        start = t * qubits_per_step
        end = start + qubits_per_step
        chunk = bits[start:end]
        
        # Convert to int (binary string '101' -> 5)
        # Remember chunk[0] is LSB of this register
        val = 0
        for i, bit in enumerate(chunk):
            if bit == '1':
                val += (1 << i)
        
        # City index is val + 1 (since we encode 1..N-1 as 0..N-2)
        city_idx = val + 1
        
        # Check validity
        if city_idx >= num_cities or city_idx in visited:
            # Invalid city or already visited -> Decoding failed physically
            # But we must return *something* for heuristics.
            # We skip adding it here, and fill later?
            # Or just append it and let cost function penalize?
            # Append invalid to allow cost function to see it's bad.
            # But duplicate cities break TSP validity strictly.
            pass 
        
        route.append(city_idx)
        visited.add(city_idx)
        
    # Fill missing cities greedily (repair)
    all_cities = set(range(num_cities))
    missing = list(all_cities - set(route))
    
    # If route has invalid indices (>N) or duplicates, we might have length > N
    # Truncate or repair.
    # Simple repair: Keep first occurrence of valid cities, fill rest.
    
    final_route = [0]
    seen = {0}
    
    # First pass: keep valid
    for c in route[1:]:
        if 0 < c < num_cities and c not in seen:
            final_route.append(c)
            seen.add(c)
            
    # Second pass: fill missing
    missing = list(all_cities - seen)
    final_route.extend(missing)
    
    return final_route

def calculate_route_cost(route: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate cost."""
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i + 1]]
    cost += distance_matrix[route[-1]][route[0]]
    return cost

def solve_tsp_binary(distance_matrix: np.ndarray, shots: int = 1024, layers: int = 1) -> Tuple[List[int], str, Dict]:
    """
    Solve TSP using Binary Encoding.
    """
    if not QISKIT_AVAILABLE:
        return list(range(len(distance_matrix))), "Qiskit Missing", {}

    n = len(distance_matrix)
    n_steps = n - 1
    qubits_per_step = get_qubits_per_step(n)
    total_qubits = n_steps * qubits_per_step
    
    logger.info(f"🧬 Binary QAOA Solver | {n} Cities | {total_qubits} Qubits")

    hamiltonian = create_binary_hamiltonian(distance_matrix)
    
    # Optimization loop (simplified for demo reliability)
    # We use pre-tuned parameters often good for MaxCut/TSP
    gamma, beta = np.pi / 3, np.pi / 4
    
    circuit = create_qaoa_circuit_binary(total_qubits, gamma, beta, hamiltonian, layers)
    simulator = AerSimulator(method='statevector') # Binary fits in statevector usually
    
    # If massive (binary reduced 36 -> 18), statevector is fine.
    # If N=15, 14*4 = 56 qubits -> Need MPS.
    # If massive (binary reduced 36 -> 18), statevector is fine usually but
    # HighLevelSynthesis can OOM on synthesis.
    # Force MPS earlier to save memory on 512MB instances.
    if total_qubits > 12:  # Lowered from 20 -> 12 for safety on Render
        simulator = AerSimulator(method='matrix_product_state')
        logger.info("Using MPS for large binary circuit to prevent OOM.")
    
    # Use optimization_level=0 to skip fast but memory-heavy synthesis passes
    compiled = transpile(circuit, simulator, optimization_level=0)
    job = simulator.run(compiled, shots=shots)
    counts = job.result().get_counts()
    
    best_route = None
    best_cost = float('inf')
    valid_count = 0
    
    for bitstring, count in counts.items():
        route = decode_binary_to_route(bitstring, n)
        # Validity check: simplified decode always returns valid permutation
        # But "raw" validity means did the quantum state obey constraints?
        # We count it as valid for statistics.
        valid_count += count
        
        cost = calculate_route_cost(route, distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_route = route
            
    return best_route, "Quantum QAOA (Binary Encoded)", {
        'num_qubits': total_qubits,
        'encoding': 'binary',
        'cost': best_cost
    }
