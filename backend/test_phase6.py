
import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    try:
        from modules.routing import find_safe_route, get_avoidances_from_violations
        logger.info("✅ modules.routing imports successful")
    except ImportError as e:
        logger.error(f"❌ modules.routing import failed: {e}")
        return False
        
    try:
        from app import app
        logger.info("✅ app imports successful")
    except ImportError as e:
        logger.error(f"❌ app import failed: {e}")
        return False
        
    return True

def test_get_avoidances():
    from modules.routing import get_avoidances_from_violations
    
    violations = [
        {'type': 'height', 'message': 'Too tall'},
        {'type': 'width', 'road_type': 'path', 'message': 'Too wide for path'},
        {'type': 'weight', 'message': 'Too heavy'}
    ]
    
    avoid = get_avoidances_from_violations(violations)
    logger.info(f"Avoidances generated: {avoid}")
    
    expected = {'tunnels', 'unpavedroads', 'ferries', 'tracks'}
    if set(avoid) == expected:
        logger.info("✅ get_avoidances_from_violations logic correct")
    else:
        logger.error(f"❌ get_avoidances_from_violations logic incorrect. Expected {expected}, got {set(avoid)}")
        return False
        
    return True

if __name__ == "__main__":
    if test_imports() and test_get_avoidances():
        logger.info("Phase 6 Logic Verification Passed")
    else:
        logger.error("Phase 6 Logic Verification Failed")
        sys.exit(1)
