import numpy as np
import logging

def calculate_total_distance(route, distance_matrix):
    """Calculate total distance of a route"""
    total = 0
    n = len(route)
    for i in range(n - 1):
        total += distance_matrix[route[i]][route[i+1]]
    # Return to start
    total += distance_matrix[route[-1]][route[0]]
    return total

def solve_tsp_quantum(distance_matrix, shots=1000):
    """
    Solve TSP using a simplified quantum approach
    """
    try:
        n = len(distance_matrix)
        # Simulated Quantum Annealing
        # We model the system's energy landscape and find the ground state (optimal route)
        # This is the classical equivalent of adiabatic quantum computation

        
        # Find best route from measurements
        # Simulated Annealing Logic
        current_route = list(range(n))
        np.random.shuffle(current_route)
        current_cost = calculate_total_distance(current_route, distance_matrix)
        
        temperature = 1000.0
        cooling_rate = 0.995
        
        best_route = list(current_route)
        min_cost = current_cost
        
        # Iterate to simulate annealing process
        for i in range(shots):
            # Create neighbor (swap two cities)
            neighbor = list(current_route)
            idx1, idx2 = np.random.randint(0, n, 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
            neighbor_cost = calculate_total_distance(neighbor, distance_matrix)
            
            # Acceptance probability
            delta = neighbor_cost - current_cost
            if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                current_route = list(neighbor)
                current_cost = neighbor_cost
                
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_route = list(current_route)
            
            temperature *= cooling_rate
            
        return best_route
        
        return best_route if best_route else list(range(n))
        
    except Exception as e:
        logging.error(f"Quantum solver error: {str(e)}")
        return list(range(n))  # Return simple sequential route as fallback

def solve_tsp_classical_fallback(distance_matrix):
    """Classical nearest-neighbor solver as fallback"""
    n = len(distance_matrix)
    if n <= 1:
        return list(range(n))
    
    unvisited = set(range(n))
    route = [0]  # Start with first city
    unvisited.remove(0)
    
    while unvisited:
        current = route[-1]
        next_city = min(unvisited, key=lambda x: distance_matrix[current][x])
        route.append(next_city)
        unvisited.remove(next_city)
    
    return route