"""
Helper script to load and analyze saved results from compute_vector_multi.py

This allows you to load the saved .pt files and JSON analysis without recomputing.
"""
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_model_results(model_name="gemma"):
    """
    Load all saved results for a model.
    
    Returns:
        dict with keys: 'vector', 'norms', 'analysis'
    """
    base_path = f"/mnt/drive_b/MATS_apply/analogy_vector_{model_name}"
    
    results = {}
    
    # Load vector
    vector_path = f"{base_path}.pt"
    try:
        results['vector'] = torch.load(vector_path)
        print(f"✓ Loaded vector from: {vector_path}")
        print(f"  Shape: {results['vector'].shape}")
    except FileNotFoundError:
        print(f"✗ Vector not found: {vector_path}")
        return None
    
    # Load norms
    norms_path = f"{base_path}_norms.pt"
    try:
        results['norms'] = torch.load(norms_path)
        print(f"✓ Loaded norms from: {norms_path}")
        print(f"  Shape: {results['norms'].shape}")
    except FileNotFoundError:
        print(f"⚠ Norms not found: {norms_path} (computing from vector...)")
        results['norms'] = results['vector'].norm(dim=1)
    
    # Load analysis JSON
    analysis_path = f"{base_path}_analysis.json"
    try:
        with open(analysis_path, 'r') as f:
            results['analysis'] = json.load(f)
        print(f"✓ Loaded analysis from: {analysis_path}")
    except FileNotFoundError:
        print(f"⚠ Analysis JSON not found: {analysis_path}")
        results['analysis'] = None
    
    return results

def compare_models(model_names=["gemma", "llama8b"]):
    """Compare multiple models using saved results."""
    print("="*60)
    print("MODEL COMPARISON FROM SAVED RESULTS")
    print("="*60)
    
    all_results = {}
    for model_name in model_names:
        print(f"\nLoading {model_name}...")
        results = load_model_results(model_name)
        if results:
            all_results[model_name] = results
    
    if len(all_results) < 2:
        print("\n⚠ Need at least 2 models to compare")
        return
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for model_name, results in all_results.items():
        norms = results['norms']
        analysis = results.get('analysis', {})
        
        best_layer = int(norms.argmax().item())
        max_norm = float(norms.max().item())
        mean_norm = float(norms.mean().item())
        
        print(f"\n{model_name.upper()}:")
        print(f"  Best layer: {best_layer}")
        print(f"  Max norm: {max_norm:.4f}")
        print(f"  Mean norm: {mean_norm:.4f}")
        print(f"  Total layers: {len(norms)}")
        
        if analysis:
            print(f"  Vector shape: {analysis.get('vector_shape', 'N/A')}")
    
    # Create comparison plot
    plt.figure(figsize=(14, 7))
    for model_name, results in all_results.items():
        norms = results['norms'].cpu().numpy() if isinstance(results['norms'], torch.Tensor) else results['norms']
        layers = np.arange(len(norms))
        best_layer = int(np.argmax(norms))
        plt.plot(layers, norms, marker='o', linewidth=2, markersize=3, 
                label=f"{model_name.upper()} (best: layer {best_layer})", alpha=0.8)
    
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Euclidean Norm of Analogy Vector", fontsize=12)
    plt.title("Model Comparison: Analogy Vector Norms by Layer", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = "/mnt/drive_b/MATS_apply/model_comparison_from_saved.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_path}")
    plt.close()

def analyze_single_model(model_name="gemma"):
    """Analyze a single model's saved results."""
    print("="*60)
    print(f"ANALYZING SAVED RESULTS: {model_name.upper()}")
    print("="*60)
    
    results = load_model_results(model_name)
    if not results:
        return
    
    vector = results['vector']
    norms = results['norms']
    analysis = results.get('analysis', {})
    
    # Print key statistics
    print("\n" + "="*60)
    print("KEY STATISTICS")
    print("="*60)
    
    best_layer = int(norms.argmax().item())
    max_norm = float(norms.max().item())
    mean_norm = float(norms.mean().item())
    std_norm = float(norms.std().item())
    min_norm = float(norms.min().item())
    min_layer = int(norms.argmin().item())
    
    print(f"Best layer: {best_layer} (norm = {max_norm:.4f})")
    print(f"Worst layer: {min_layer} (norm = {min_norm:.4f})")
    print(f"Mean norm: {mean_norm:.4f}")
    print(f"Std norm: {std_norm:.4f}")
    print(f"Norm range: {min_norm:.4f} to {max_norm:.4f}")
    
    # Top 5 layers
    top_5_indices = norms.argsort(descending=True)[:5]
    print(f"\nTop 5 layers by norm:")
    for i, idx in enumerate(top_5_indices):
        print(f"  {i+1}. Layer {idx.item()}: {norms[idx].item():.4f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    layers = np.arange(len(norms))
    norms_np = norms.cpu().numpy() if isinstance(norms, torch.Tensor) else norms
    plt.plot(layers, norms_np, marker='o', linewidth=2, markersize=4)
    plt.axvline(x=best_layer, color='red', linestyle='--', alpha=0.7, 
               label=f'Best layer ({best_layer})')
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Euclidean Norm of Analogy Vector", fontsize=12)
    plt.title(f"Analogy Vector Norm by Layer: {model_name.upper()}\n"
             f"Best layer: {best_layer} (norm={max_norm:.4f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = f"/mnt/drive_b/MATS_apply/{model_name}_analysis_from_saved.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            models = sys.argv[2:] if len(sys.argv) > 2 else ["gemma", "llama8b"]
            compare_models(models)
        else:
            model_name = sys.argv[1]
            analyze_single_model(model_name)
    else:
        # Default: analyze gemma
        analyze_single_model("gemma")
        print("\n" + "="*60)
        print("Usage:")
        print("  python analyze_saved_results.py gemma")
        print("  python analyze_saved_results.py llama8b")
        print("  python analyze_saved_results.py compare gemma llama8b")
        print("="*60)

