import sys
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from datetime import datetime

# Add parent directory to path so we can import the pamr package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from pamr package
from pamr.core.network import NetworkTopology
from pamr.core.routing import PAMRRouter
from examples.comparison_with_ospf import OSPFRouter

def test_path_quality_improvement():
    """Test the improved PAMR router against OSPF on path quality."""
    print("Testing path quality improvements in PAMR vs OSPF...")
    
    # Create a network for testing
    network = NetworkTopology(num_nodes=30, connectivity=0.3, seed=42)
    
    # Create a copy of the graph for OSPF
    import pickle
    network_data = pickle.dumps(network.graph)
    ospf_graph = pickle.loads(network_data)
    
    # Create routers
    ospf_router = OSPFRouter(ospf_graph)
    
    # Create original PAMR router (with global path disabled)
    original_pamr_router = PAMRRouter(network.graph, alpha=1.0, beta=2.0, gamma=1.0)
    original_pamr_router.use_global_path = False
    
    # Create improved PAMR router
    improved_pamr_router = PAMRRouter(network.graph, alpha=2.0, beta=3.0, gamma=2.5)
    improved_pamr_router.use_global_path = True
    
    # Test multiple source-destination pairs
    num_pairs = 20
    test_pairs = []
    
    # Generate random source-destination pairs
    nodes = list(network.graph.nodes())
    for _ in range(num_pairs):
        src, dst = random.sample(nodes, 2)
        test_pairs.append((src, dst))
    
    # Results storage
    results = {
        'ospf_quality': [],
        'original_pamr_quality': [],
        'improved_pamr_quality': [],
        'ospf_path_length': [],
        'original_pamr_path_length': [],
        'improved_pamr_path_length': []
    }
    
    # Test each pair
    for src, dst in test_pairs:
        # Get paths from all routers
        ospf_path, ospf_quality = ospf_router.find_path(src, dst)
        original_path, original_quality = original_pamr_router.find_path(src, dst)
        improved_path, improved_quality = improved_pamr_router.find_path(src, dst)
        
        # Store results
        results['ospf_quality'].append(ospf_quality)
        results['original_pamr_quality'].append(original_quality)
        results['improved_pamr_quality'].append(improved_quality)
        
        results['ospf_path_length'].append(len(ospf_path) - 1 if ospf_path else 0)
        results['original_pamr_path_length'].append(len(original_path) - 1 if original_path else 0)
        results['improved_pamr_path_length'].append(len(improved_path) - 1 if improved_path else 0)
        
        # Print detailed comparison for this pair
        print(f"\nSource {src} to Destination {dst}:")
        print(f"  OSPF:          Quality = {ospf_quality:.4f}, Path length = {len(ospf_path) - 1 if ospf_path else 0}")
        print(f"  Original PAMR: Quality = {original_quality:.4f}, Path length = {len(original_path) - 1 if original_path else 0}")
        print(f"  Improved PAMR: Quality = {improved_quality:.4f}, Path length = {len(improved_path) - 1 if improved_path else 0}")
        
        if improved_quality > ospf_quality:
            print("  Result: Improved PAMR outperforms OSPF! ✅")
        elif improved_quality > original_quality:
            print("  Result: Improved PAMR better than original PAMR, but still below OSPF")
        else:
            print("  Result: Improvements not effective for this path")
    
    # Calculate average metrics
    avg_ospf_quality = np.mean(results['ospf_quality'])
    avg_original_pamr_quality = np.mean(results['original_pamr_quality'])
    avg_improved_pamr_quality = np.mean(results['improved_pamr_quality'])
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Average OSPF Quality:          {avg_ospf_quality:.4f}")
    print(f"Average Original PAMR Quality: {avg_original_pamr_quality:.4f}")
    print(f"Average Improved PAMR Quality: {avg_improved_pamr_quality:.4f}")
    
    # Calculate improvement percentages
    original_vs_ospf = ((avg_original_pamr_quality / avg_ospf_quality) - 1) * 100
    improved_vs_ospf = ((avg_improved_pamr_quality / avg_ospf_quality) - 1) * 100
    improved_vs_original = ((avg_improved_pamr_quality / avg_original_pamr_quality) - 1) * 100
    
    print(f"Original PAMR vs OSPF:      {original_vs_ospf:.2f}%")
    print(f"Improved PAMR vs OSPF:      {improved_vs_ospf:.2f}%")
    print(f"Improved vs Original PAMR:  {improved_vs_original:.2f}%")
    
    # Create visualization
    create_comparison_visualization(results)
    
def create_comparison_visualization(results):
    """Create visualizations to show the improvement in path quality."""
    # Create output directory if it doesn't exist
    output_dir = "./comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path Quality Comparison
    plt.figure(figsize=(12, 6))
    
    # Bar chart of average qualities
    avg_ospf = np.mean(results['ospf_quality'])
    avg_original = np.mean(results['original_pamr_quality'])
    avg_improved = np.mean(results['improved_pamr_quality'])
    
    plt.bar(['OSPF', 'Original PAMR', 'Improved PAMR'], 
            [avg_ospf, avg_original, avg_improved],
            color=['blue', 'red', 'green'])
    
    plt.title('Average Path Quality Comparison')
    plt.ylabel('Average Path Quality')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate([avg_ospf, avg_original, avg_improved]):
        plt.text(i, v + 0.001, f"{v:.4f}", ha='center')
    
    # Add percentage improvement labels
    improvement = ((avg_improved / avg_ospf) - 1) * 100
    plt.figtext(0.5, 0.01, 
               f"Improved PAMR vs OSPF: {improvement:.2f}% difference in path quality", 
               ha="center", fontsize=12, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quality_path = os.path.join(output_dir, f'pamr_quality_improvement_{timestamp}.png')
    plt.savefig(quality_path)
    plt.close()
    
    # Path Length Comparison
    plt.figure(figsize=(12, 6))
    
    # Box plot of path lengths
    plt.boxplot([results['ospf_path_length'], 
                results['original_pamr_path_length'], 
                results['improved_pamr_path_length']],
               labels=['OSPF', 'Original PAMR', 'Improved PAMR'])
    
    plt.title('Path Length Comparison')
    plt.ylabel('Path Length (Hops)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    length_path = os.path.join(output_dir, f'pamr_path_length_comparison_{timestamp}.png')
    plt.savefig(length_path)
    plt.close()
    
    # Individual quality comparison
    plt.figure(figsize=(14, 7))
    
    x = range(len(results['ospf_quality']))
    plt.plot(x, results['ospf_quality'], 'bo-', label='OSPF')
    plt.plot(x, results['original_pamr_quality'], 'ro-', label='Original PAMR')
    plt.plot(x, results['improved_pamr_quality'], 'go-', label='Improved PAMR')
    
    plt.title('Path Quality for Each Source-Destination Pair')
    plt.xlabel('Test Case Number')
    plt.ylabel('Path Quality')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    pairs_path = os.path.join(output_dir, f'pamr_quality_by_pair_{timestamp}.png')
    plt.savefig(pairs_path)
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")
    print(f"1. Average Path Quality: {quality_path}")
    print(f"2. Path Length Comparison: {length_path}")
    print(f"3. Quality by Pair: {pairs_path}")

if __name__ == "__main__":
    test_path_quality_improvement()