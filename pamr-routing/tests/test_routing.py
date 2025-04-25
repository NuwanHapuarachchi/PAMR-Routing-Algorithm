import pytest
from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter

def test_network_creation():
    """Test network topology creation."""
    network = NetworkTopology(num_nodes=10, connectivity=0.4)
    assert len(network.graph.nodes) == 10
    
def test_path_finding():
    """Test path finding capabilities."""
    network = NetworkTopology(num_nodes=5, connectivity=0.8)
    router = PAMRRouter(network.graph)
    
    # Test path between all pairs
    for src in range(5):
        for dest in range(5):
            if src == dest:
                continue
                
            path, quality = router.find_path(src, dest)
            # Verify path exists and connects source to destination
            assert len(path) >= 2
            assert path[0] == src
            assert path[-1] == dest
            assert quality > 0
