"""
IBM Quantum Hardware Integration Module
Provides real quantum hardware execution with error mitigation
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# IBM Quantum imports
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
    from qiskit import transpile
    from qiskit.providers import Backend
    IBM_RUNTIME_AVAILABLE = True
    logger.info("✅ IBM Quantum Runtime available")
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    logger.warning("⚠️  IBM Quantum Runtime not available")


def get_least_busy_backend(service: 'QiskitRuntimeService', min_qubits: int = 5) -> Optional['Backend']:
    """
    Get the least busy IBM Quantum backend with sufficient qubits.
    
    Args:
        service: QiskitRuntimeService instance
        min_qubits: Minimum number of qubits required
        
    Returns:
        Least busy backend or None
    """
    try:
        # Get available backends
        backends = service.backends(
            filters=lambda x: x.configuration().n_qubits >= min_qubits
            and not x.configuration().simulator
            and x.status().operational
        )
        
        if not backends:
            logger.warning(f"No operational backends with {min_qubits}+ qubits found")
            return None
        
        # Sort by pending jobs (least busy first)
        least_busy = min(backends, key=lambda b: b.status().pending_jobs)
        
        logger.info(f"Selected backend: {least_busy.name}")
        logger.info(f"  Qubits: {least_busy.configuration().n_qubits}")
        logger.info(f"  Pending jobs: {least_busy.status().pending_jobs}")
        
        return least_busy
        
    except Exception as e:
        logger.error(f"Error selecting backend: {e}")
        return None


def execute_on_ibm_hardware(circuit, shots: int = 1024, 
                             resilience_level: int = 1,
                             api_token: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Execute circuit on real IBM Quantum hardware with error mitigation.
    
    Args:
        circuit: Quantum circuit to execute
        shots: Number of measurements
        resilience_level: Error mitigation level (0=none, 1=readout, 2=ZNE)
        api_token: IBM Quantum API token
        
    Returns:
        Tuple of (counts, metadata)
    """
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError("IBM Quantum Runtime not available")
    
    try:
        # Initialize service
        if api_token:
            service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
        else:
            # Try to use saved credentials
            service = QiskitRuntimeService(channel="ibm_quantum")
        
        # Get least busy backend
        num_qubits = circuit.num_qubits
        backend = get_least_busy_backend(service, min_qubits=num_qubits)
        
        if backend is None:
            raise ValueError("No suitable backend available")
        
        # Configure options with error mitigation
        options = Options()
        options.resilience_level = resilience_level
        options.optimization_level = 3
        options.execution.shots = shots
        
        logger.info(f"🚀 Executing on {backend.name} with resilience_level={resilience_level}")
        
        # Create session and run
        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session, options=options)
            
            # Transpile for hardware
            transpiled = transpile(
                circuit,
                backend=backend,
                optimization_level=3,
                seed_transpiler=42
            )
            
            # Execute
            job = sampler.run(transpiled)
            result = job.result()
            
            # Get counts
            counts = result.quasi_dists[0].binary_probabilities()
            counts = {k: int(v * shots) for k, v in counts.items()}
            
            # Metadata
            metadata = {
                'backend': backend.name,
                'backend_version': backend.version,
                'qubits': num_qubits,
                'circuit_depth': transpiled.depth(),
                'resilience_level': resilience_level,
                'job_id': job.job_id(),
                'execution_time': result.metadata.get('time_taken', 0)
            }
            
            logger.info(f"✅ Job completed: {job.job_id()}")
            logger.info(f"   Circuit depth: {transpiled.depth()}")
            
            return counts, metadata
            
    except Exception as e:
        logger.error(f"IBM Quantum execution failed: {e}")
        raise


def check_ibm_credentials() -> bool:
    """Check if IBM Quantum credentials are configured."""
    if not IBM_RUNTIME_AVAILABLE:
        return False
    
    try:
        service = QiskitRuntimeService(channel="ibm_quantum")
        backends = service.backends()
        return len(backends) > 0
    except Exception:
        return False
