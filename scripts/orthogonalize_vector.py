"""
Orthogonalization: Remove token entanglement from analogy vector.

This implements the "Refusal Paper" move - projecting the analogy vector
orthogonal to problematic token directions (like "Imagine", "like", etc.)
to isolate the "concept" from "lexical triggers".

Based on empirical validation:
- Gemma: Optimal layers 29 (best) or 35 (stable region)
- Llama: Optimal layers 24 (best) or 28 (stable region)
"""
import torch
from transformer_lens import HookedTransformer
import numpy as np
import json

# Configuration: Support multiple models
MODELS = {
    "gemma": {
        "name": "google/gemma-2-9b-it",
        "vector_path": "/mnt/drive_b/MATS_apply/analogy_vector_gemma.pt",
        "optimal_layers": [29, 35],  # Empirically best (29) and stable region middle (35)
    },
    "llama8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "vector_path": "/mnt/drive_b/MATS_apply/analogy_vector_llama8b.pt",
        "optimal_layers": [24, 28],  # Empirically best (24) and stable region middle (28)
    }
}

DEVICE = "cuda"

# Tokens that might cause mode collapse
PROBLEMATIC_TOKENS = [
    " Imagine",  # Note: space matters in tokenization
    " like",
    " similar",
    " analogy",
    " metaphor",
    " think",
]

def orthogonalize_vector(dirty_vector, model, target_tokens, layer_idx):
    """
    Remove projection of dirty_vector onto token directions.
    
    Formula: v_clean = v - (v · u) / (u · u) * u
    where u is the unembedding direction for a token.
    
    Args:
        dirty_vector: [d_model] tensor - the original analogy vector
        model: HookedTransformer model
        target_tokens: List of token strings to orthogonalize against
        layer_idx: Which layer's vector we're cleaning
    
    Returns:
        clean_vector: [d_model] tensor - orthogonalized vector
        removed_projections: dict of how much was removed for each token
    """
    clean_vector = dirty_vector.clone()
    removed_projections = {}
    
    # Get unembedding matrix: [d_model, d_vocab]
    W_U = model.W_U  # Shape: [d_model, d_vocab]
    
    for token_str in target_tokens:
        try:
            # Get token ID
            token_id = model.to_single_token(token_str)
            
            # Get the direction in residual stream that writes this token
            # This is the column of W_U corresponding to the token
            token_dir = W_U[:, token_id]  # Shape: [d_model]
            
            # Calculate projection: (v · u) / (u · u) * u
            dot_product = torch.dot(clean_vector, token_dir)
            norm_sq = torch.dot(token_dir, token_dir)
            
            if norm_sq > 1e-10:  # Avoid division by zero
                projection = (dot_product / norm_sq) * token_dir
                
                # Remove the projection
                clean_vector = clean_vector - projection
                
                removed_projections[token_str] = {
                    "dot_product": dot_product.item(),
                    "projection_norm": projection.norm().item(),
                    "removed_fraction": (projection.norm() / dirty_vector.norm()).item()
                }
            else:
                removed_projections[token_str] = {"error": "Token direction has zero norm"}
                
        except Exception as e:
            removed_projections[token_str] = {"error": str(e)}
            print(f"Warning: Could not process token '{token_str}': {e}")
    
    return clean_vector, removed_projections

def orthogonalize_for_model(model_key, model_config):
    """Orthogonalize analogy vector for a single model."""
    model_name = model_config["name"]
    vector_path = model_config["vector_path"]
    optimal_layers = model_config["optimal_layers"]
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_key.upper()}")
    print(f"Model: {model_name}")
    print(f"Optimal layers: {optimal_layers}")
    print(f"{'='*60}\n")
    
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE, dtype="bfloat16")
    model.eval()
    
    # Load dirty vector
    try:
        dirty_vector = torch.load(vector_path)  # Shape: [n_layers, d_model]
    except FileNotFoundError:
        print(f"⚠ Warning: {vector_path} not found. Skipping {model_key}.")
        del model
        torch.cuda.empty_cache()
        return None
    
    print(f"Loaded dirty vector: {dirty_vector.shape}")
    n_layers = dirty_vector.shape[0]
    
    # Clean vector for each layer
    clean_vector = torch.zeros_like(dirty_vector)
    all_removed_projections = {}
    
    print(f"Orthogonalizing against tokens: {PROBLEMATIC_TOKENS}")
    print(f"Processing {n_layers} layers...\n")
    
    for layer_idx in range(n_layers):
        layer_vec = dirty_vector[layer_idx]
        clean_vec, removed = orthogonalize_vector(
            layer_vec, model, PROBLEMATIC_TOKENS, layer_idx
        )
        clean_vector[layer_idx] = clean_vec
        all_removed_projections[layer_idx] = removed
        
        # Highlight optimal layers
        is_optimal = layer_idx in optimal_layers
        marker = " ⭐" if is_optimal else ""
        
        if layer_idx % 5 == 0 or layer_idx == n_layers - 1 or is_optimal:
            # Print summary for this layer
            total_removed = sum(
                r.get("removed_fraction", 0) 
                for r in removed.values() 
                if isinstance(r, dict) and "removed_fraction" in r
            )
            print(f"Layer {layer_idx:2d}: Removed {total_removed*100:.2f}% of vector magnitude{marker}")
    
    # Save clean vector
    output_path = vector_path.replace('.pt', '_clean.pt')
    torch.save(clean_vector, output_path)
    print(f"\n✓ Clean vector saved to: {output_path}")
    
    # Analysis: How much did we remove?
    dirty_norms = dirty_vector.norm(dim=1)
    clean_norms = clean_vector.norm(dim=1)
    reduction = (1 - clean_norms / dirty_norms) * 100
    
    print(f"\n{'='*60}")
    print(f"ORTHOGONALIZATION SUMMARY: {model_key.upper()}")
    print(f"{'='*60}")
    print(f"Average reduction: {reduction.mean():.2f}%")
    print(f"Max reduction: {reduction.max():.2f}% (at layer {reduction.argmax().item()})")
    print(f"Min reduction: {reduction.min():.2f}% (at layer {reduction.argmin().item()})")
    
    # Show reduction at optimal layers
    print(f"\nReduction at optimal layers:")
    for opt_layer in optimal_layers:
        if opt_layer < len(reduction):
            print(f"  Layer {opt_layer}: {reduction[opt_layer]:.2f}% reduction")
    
    # Which tokens contributed most?
    print(f"\nToken removal summary (averaged across layers):")
    token_totals = {}
    for layer_removed in all_removed_projections.values():
        for token, info in layer_removed.items():
            if isinstance(info, dict) and "removed_fraction" in info:
                if token not in token_totals:
                    token_totals[token] = []
                token_totals[token].append(info["removed_fraction"])
    
    for token, fractions in sorted(token_totals.items(), 
                                  key=lambda x: np.mean(x[1]), 
                                  reverse=True):
        avg_removed = np.mean(fractions) * 100
        print(f"  {token:15s}: {avg_removed:5.2f}% average removal")
    
    # Save analysis
    analysis_path = output_path.replace('.pt', '_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump({
            "model": model_key,
            "model_name": model_name,
            "optimal_layers": optimal_layers,
            "reduction_per_layer": reduction.tolist(),
            "reduction_at_optimal_layers": {
                f"layer_{layer}": float(reduction[layer]) 
                for layer in optimal_layers if layer < len(reduction)
            },
            "token_removals": {
                token: {
                    "avg_removed_percent": float(np.mean(fractions) * 100),
                    "std_removed_percent": float(np.std(fractions) * 100)
                }
                for token, fractions in token_totals.items()
            }
        }, f, indent=2)
    print(f"\n✓ Analysis saved to: {analysis_path}")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "model_key": model_key,
        "output_path": output_path,
        "analysis_path": analysis_path,
        "reduction": reduction,
        "optimal_layers": optimal_layers
    }

def main():
    """Orthogonalize analogy vectors for all configured models."""
    print("="*60)
    print("ORTHOGONALIZATION: Remove Token Entanglement")
    print("="*60)
    print("\nThis will process all configured models.")
    print("Optimal layers (from empirical validation) will be highlighted.\n")
    
    results = []
    
    for model_key, model_config in MODELS.items():
        try:
            result = orthogonalize_for_model(model_key, model_config)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n✗ Error processing {model_key}: {e}")
            print("Skipping to next model...\n")
            continue
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        for result in results:
            print(f"\n{result['model_key'].upper()}:")
            print(f"  Clean vector: {result['output_path']}")
            print(f"  Analysis: {result['analysis_path']}")
            avg_reduction = result['reduction'].mean().item()
            print(f"  Average reduction: {avg_reduction:.2f}%")
            print(f"  Optimal layers: {result['optimal_layers']}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

