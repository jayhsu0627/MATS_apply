"""
Helper script to load and analyze saved stability analysis results.

This allows you to load the saved .pt files and JSON analysis without recomputing.
"""
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_stability_results(model_name="gemma"):
    """
    Load all saved stability analysis results for a model.
    
    Returns:
        dict with keys: 'similarities', 'analysis'
    """
    # Handle model name aliases
    model_name_map = {
        "llama": "llama8b",  # Map "llama" to "llama8b" for file naming
    }
    actual_model_name = model_name_map.get(model_name, model_name)
    
    base_path = f"/mnt/drive_b/MATS_apply/analogy_vector_{actual_model_name}"
    
    results = {}
    
    # Load similarities - try multiple possible paths
    possible_paths = [
        f"{base_path}_similarities.pt",
        f"/mnt/drive_b/MATS_apply/analogy_vector_{model_name}_similarities.pt",  # Try original name too
    ]
    
    similarities_path = None
    for path in possible_paths:
        try:
            results['similarities'] = torch.load(path).numpy()
            similarities_path = path
            break
        except FileNotFoundError:
            continue
    
    if similarities_path:
        print(f"✓ Loaded similarities from: {similarities_path}")
        print(f"  Shape: {results['similarities'].shape}")
    else:
        print(f"✗ Similarities not found. Tried:")
        for path in possible_paths:
            print(f"    - {path}")
        return None
    
    # Load analysis JSON - try multiple possible paths
    possible_json_paths = [
        f"{base_path}_stability_analysis.json",
        f"/mnt/drive_b/MATS_apply/analogy_vector_{model_name}_stability_analysis.json",
    ]
    
    analysis_path = None
    for path in possible_json_paths:
        try:
            with open(path, 'r') as f:
                results['analysis'] = json.load(f)
            analysis_path = path
            break
        except FileNotFoundError:
            continue
    
    if analysis_path:
        print(f"✓ Loaded analysis from: {analysis_path}")
    else:
        print(f"⚠ Analysis JSON not found. Tried:")
        for path in possible_json_paths:
            print(f"    - {path}")
        results['analysis'] = None
    
    return results

def compare_stability_models(model_names=["gemma", "llama8b"]):
    """Compare stability across multiple models using saved results."""
    print("="*60)
    print("STABILITY COMPARISON FROM SAVED RESULTS")
    print("="*60)
    
    all_results = {}
    for model_name in model_names:
        print(f"\nLoading {model_name}...")
        results = load_stability_results(model_name)
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
        similarities = results['similarities']
        analysis = results.get('analysis', {})
        
        mean_sim = float(np.mean(similarities))
        min_sim = float(np.min(similarities))
        min_layer = int(np.argmin(similarities))
        max_sim = float(np.max(similarities))
        stable_start = analysis.get('stable_region_start') if analysis else None
        
        print(f"\n{model_name.upper()}:")
        print(f"  Mean similarity: {mean_sim:.4f}")
        print(f"  Min similarity: {min_sim:.4f} (at layer {min_layer})")
        print(f"  Max similarity: {max_sim:.4f}")
        if stable_start is not None:
            print(f"  Stable region: layers {stable_start} to {analysis.get('stable_region_end', 'N/A')}")
        print(f"  Turning points: {analysis.get('turning_points', []) if analysis else 'N/A'}")
    
    # Create comparison plot
    plt.figure(figsize=(14, 7))
    for model_name, results in all_results.items():
        similarities = results['similarities']
        layers = np.arange(len(similarities))
        stable_start = results.get('analysis', {}).get('stable_region_start')
        label = f"{model_name.upper()}"
        if stable_start is not None:
            label += f" (stable from {stable_start})"
        plt.plot(layers, similarities, marker='o', linewidth=2, markersize=3, 
                label=label, alpha=0.8)
    
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.3, label='Stable threshold (0.95)')
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Cosine Similarity (Layer L vs L+1)", fontsize=12)
    plt.title("Model Comparison: Analogy Vector Stability", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = "/mnt/drive_b/MATS_apply/stability_comparison_from_saved.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_path}")
    plt.close()

def analyze_single_stability(model_name="gemma"):
    """Analyze a single model's saved stability results."""
    print("="*60)
    print(f"ANALYZING STABILITY RESULTS: {model_name.upper()}")
    print("="*60)
    
    results = load_stability_results(model_name)
    if not results:
        return
    
    similarities = results['similarities']
    analysis = results.get('analysis', {})
    
    # Print key statistics
    print("\n" + "="*60)
    print("KEY STATISTICS")
    print("="*60)
    
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    min_sim = float(np.min(similarities))
    min_layer = int(np.argmin(similarities))
    max_sim = float(np.max(similarities))
    max_layer = int(np.argmax(similarities))
    
    print(f"Mean similarity: {mean_sim:.4f} ± {std_sim:.4f}")
    print(f"Min similarity: {min_sim:.4f} (at layer {min_layer})")
    print(f"Max similarity: {max_sim:.4f} (at layer {max_layer})")
    print(f"Similarity range: {min_sim:.4f} to {max_sim:.4f}")
    
    if analysis:
        stable_start = analysis.get('stable_region_start')
        stable_end = analysis.get('stable_region_end')
        turning_points = analysis.get('turning_points', [])
        
        if stable_start is not None:
            print(f"\nStable region: Layers {stable_start} to {stable_end}")
            print(f"  (Similarity > 0.95 from layer {stable_start} onwards)")
            print(f"  Recommendation: Use layers {stable_start} to {stable_end} for steering")
        else:
            print("\nNo highly stable region found (similarity < 0.95 throughout)")
        
        if turning_points:
            print(f"\nTurning points (computation happening): {turning_points}")
        else:
            print("\nNo major turning points found (vector direction is relatively stable)")
    
    # Plot
    plt.figure(figsize=(12, 6))
    layers = np.arange(len(similarities))
    plt.plot(layers, similarities, marker='o', linewidth=2, markersize=4)
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Stable threshold (0.95)')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold (0.9)')
    
    if analysis:
        stable_start = analysis.get('stable_region_start')
        if stable_start is not None:
            plt.axvline(x=stable_start, color='green', linestyle=':', alpha=0.7, 
                       label=f'Stable region starts (layer {stable_start})')
        
        turning_points = analysis.get('turning_points', [])
        for tp in turning_points:
            if tp < len(similarities):
                plt.axvline(x=tp, color='red', linestyle=':', alpha=0.5)
                plt.text(tp, similarities[tp], f'TP {tp}', rotation=90, fontsize=8)
    
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title(f"Analogy Vector Stability: {model_name.upper()}\n"
             f"Mean similarity: {mean_sim:.4f}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = f"/mnt/drive_b/MATS_apply/{model_name}_stability_from_saved.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            models = sys.argv[2:] if len(sys.argv) > 2 else ["gemma", "llama8b"]
            compare_stability_models(models)
        else:
            model_name = sys.argv[1]
            analyze_single_stability(model_name)
    else:
        # Default: analyze gemma
        analyze_single_stability("gemma")
        print("\n" + "="*60)
        print("Usage:")
        print("  python analyze_stability_results.py gemma")
        print("  python analyze_stability_results.py llama8b  (or 'llama' as alias)")
        print("  python analyze_stability_results.py compare gemma llama8b")
        print("="*60)

