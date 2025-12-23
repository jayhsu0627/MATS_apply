"""
Layer Stability Analysis: Find where analogy vector direction stabilizes.

This implements the "Turning Point" analysis from Google AI Studio suggestions.
We calculate cosine similarity between layer L and L+1 to identify:
- Where computation happens (rapid changes)
- Where information propagates (stable direction)
"""
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import sys
import json

def analyze_layer_stability(vector_path, model_name="gemma", save_plot=True):
    """
    Analyze stability of analogy vector across layers.
    
    Args:
        vector_path: Path to analogy vector .pt file
        model_name: Name of model (for labeling)
        save_plot: Whether to save the plot
    """
    print(f"\n{'='*60}")
    print(f"Layer Stability Analysis: {model_name}")
    print(f"{'='*60}\n")
    
    # Load vector: shape [n_layers, d_model]
    vector = torch.load(vector_path)
    n_layers = vector.shape[0]
    
    print(f"Vector shape: {vector.shape}")
    print(f"Number of layers: {n_layers}\n")
    
    # Calculate Cosine Similarity between Layer L and Layer L+1
    similarities = []
    for i in range(n_layers - 1):
        # Get vectors for layer i and i+1
        vec_l = vector[i]
        vec_l1 = vector[i + 1]
        
        # Calculate cosine similarity
        sim = F.cosine_similarity(
            vec_l.unsqueeze(0), 
            vec_l1.unsqueeze(0)
        )
        similarities.append(sim.item())
    
    similarities = np.array(similarities)
    
    # Find where stability changes
    # High similarity (close to 1.0) = stable propagation
    # Low similarity = computation/transformation happening
    
    # Identify "turning points" - where similarity drops significantly
    similarity_diffs = np.diff(similarities)
    turning_points = []
    
    # Find layers where similarity drops by more than 0.1
    threshold = 0.1
    for i, diff in enumerate(similarity_diffs):
        if diff < -threshold:  # Significant drop
            turning_points.append(i + 1)  # Layer where change happens
    
    # Print analysis
    print("Layer Stability Analysis:")
    print(f"  Mean similarity: {similarities.mean():.4f}")
    print(f"  Min similarity: {similarities.min():.4f} (at layer {similarities.argmin()})")
    print(f"  Max similarity: {similarities.max():.4f} (at layer {similarities.argmax()})")
    
    if turning_points:
        print(f"\n  Turning points (where computation happens): {turning_points}")
    else:
        print(f"\n  No major turning points found (vector direction is relatively stable)")
    
    # Find stable region (where similarity > 0.95)
    stable_region = np.where(similarities > 0.95)[0]
    stable_start = stable_region[0] if len(stable_region) > 0 else None
    stable_end = n_layers - 1 if stable_start is not None else None
    
    if stable_start is not None:
        print(f"\n  Stable region starts at layer {stable_start}")
        print(f"  (Similarity > 0.95 from layer {stable_start} onwards)")
        print(f"  Recommendation: Use layers {stable_start} to {n_layers-1} for steering")
    else:
        print(f"\n  No highly stable region found (similarity < 0.95 throughout)")
    
    # Prepare analysis results
    analysis_results = {
        "model_name": model_name,
        "vector_path": vector_path,
        "n_layers": int(n_layers),
        "similarities": similarities.tolist(),  # Convert numpy to list for JSON
        "mean_similarity": float(similarities.mean()),
        "std_similarity": float(similarities.std()),
        "min_similarity": float(similarities.min()),
        "min_layer": int(similarities.argmin()),
        "max_similarity": float(similarities.max()),
        "max_layer": int(similarities.argmax()),
        "turning_points": [int(tp) for tp in turning_points],
        "turning_point_threshold": 0.1,
        "stable_region_start": int(stable_start) if stable_start is not None else None,
        "stable_region_end": int(stable_end) if stable_end is not None else None,
        "stable_threshold": 0.95,
        "recommended_steering_layers": {
            "start": int(stable_start) if stable_start is not None else None,
            "end": int(stable_end) if stable_end is not None else None,
            "range": list(range(stable_start, n_layers)) if stable_start is not None else []
        }
    }
    
    # Save similarities as .pt file
    similarities_path = vector_path.replace('.pt', '_similarities.pt')
    torch.save(torch.tensor(similarities, dtype=torch.float32), similarities_path)
    print(f"  Similarities saved to: {similarities_path}")
    
    # Save analysis results as JSON
    analysis_path = vector_path.replace('.pt', '_stability_analysis.json')
    try:
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"  Analysis results saved to: {analysis_path}")
    except Exception as e:
        print(f"  ⚠ Warning: Could not save JSON analysis: {e}")
    
    # Plotting
    layers = np.arange(n_layers - 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, similarities, marker='o', linewidth=2, markersize=4)
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Stable threshold (0.95)')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold (0.9)')
    
    # Mark turning points
    if turning_points:
        for tp in turning_points:
            if tp < len(similarities):
                plt.axvline(x=tp, color='red', linestyle=':', alpha=0.5)
                plt.text(tp, similarities[tp] if tp < len(similarities) else 0.5, 
                        f'TP {tp}', rotation=90, fontsize=8)
    
    plt.title(f"Analogy Vector Stability: {model_name.upper()}\n(Cosine Similarity between Layer L and L+1)")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_plot:
        plot_path = vector_path.replace('.pt', '_stability.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {plot_path}")
    
    plt.close()
    
    return {
        "similarities": similarities,
        "turning_points": turning_points,
        "stable_start": stable_start,
        "stable_end": stable_end,
        "mean_similarity": float(similarities.mean()),
        "min_similarity": float(similarities.min()),
        "min_layer": int(similarities.argmin()),
        "max_similarity": float(similarities.max()),
        "max_layer": int(similarities.argmax()),
        "analysis_results": analysis_results
    }

def compare_models():
    """Compare stability across both models."""
    models = {
        "gemma": "/mnt/drive_b/MATS_apply/analogy_vector_gemma.pt",
        "llama": "/mnt/drive_b/MATS_apply/analogy_vector_llama8b.pt"
    }
    
    results = {}
    for model_name, path in models.items():
        try:
            results[model_name] = analyze_layer_stability(path, model_name)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping {model_name}.")
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    # Comparison plot and save comparison results
    if len(results) == 2:
        plt.figure(figsize=(12, 6))
        for model_name, result in results.items():
            similarities = result["similarities"]
            layers = np.arange(len(similarities))
            plt.plot(layers, similarities, marker='o', linewidth=2, 
                    markersize=3, label=model_name.upper(), alpha=0.7)
        
        plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
        plt.title("Model Comparison: Analogy Vector Stability")
        plt.xlabel("Layer")
        plt.ylabel("Cosine Similarity (Layer L vs L+1)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        comparison_plot_path = "/mnt/drive_b/MATS_apply/layer_stability_comparison.png"
        plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {comparison_plot_path}")
        plt.close()
        
        # Save comparison analysis
        # Convert all numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native(item) for item in obj)
            else:
                return obj
        
        comparison_analysis = {
            "models_compared": list(results.keys()),
            "comparison_date": None,  # Could add timestamp if needed
            "results": {
                model_name: convert_to_native(result.get("analysis_results", {})) 
                for model_name, result in results.items()
            },
            "summary": {
                model_name: {
                    "stable_region_start": convert_to_native(result.get("stable_start")),
                    "stable_region_end": convert_to_native(result.get("stable_end")),
                    "mean_similarity": convert_to_native(result.get("mean_similarity")),
                    "turning_points": convert_to_native(result.get("turning_points", []))
                }
                for model_name, result in results.items()
            }
        }
        
        comparison_analysis_path = "/mnt/drive_b/MATS_apply/layer_stability_comparison_analysis.json"
        try:
            with open(comparison_analysis_path, 'w') as f:
                json.dump(comparison_analysis, f, indent=2)
            print(f"Comparison analysis saved to: {comparison_analysis_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not save comparison analysis: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Try to analyze existing vector (works with either name)
    vector_paths = [
        "/mnt/drive_b/MATS_apply/analogy_vector_gemma.pt",
        "/mnt/drive_b/MATS_apply/analogy_vector.pt"  # Fallback to original
    ]
    
    found = False
    for path in vector_paths:
        try:
            analyze_layer_stability(path, "gemma")
            found = True
            break
        except FileNotFoundError:
            continue
    
    if not found:
        print("\nERROR: No analogy vector found!")
        print("Please run compute_vector.py or compute_vector_multi.py first.")
        sys.exit(1)
    
    # Try llama if it exists
    try:
        analyze_layer_stability("/mnt/drive_b/MATS_apply/analogy_vector_llama8b.pt", "llama")
    except FileNotFoundError:
        print("\n(Optional) Llama vector not found. Run compute_vector_multi.py to compare models.")
    
    # Compare if both exist
    try:
        compare_models()
    except:
        pass  # Comparison is optional

