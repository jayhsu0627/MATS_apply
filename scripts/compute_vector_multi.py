import torch
import json
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Configuration: Support multiple models
MODELS = {
    "gemma": {
        "name": "google/gemma-2-9b-it",
        "output_file": "/mnt/drive_b/MATS_apply/analogy_vector_gemma.pt"
    },
    "llama8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "output_file": "/mnt/drive_b/MATS_apply/analogy_vector_llama8b.pt"
    },
    # "llama70b": {
    #     "name": "meta-llama/Llama-3.1-70B-Instruct",
    #     "output_file": "/mnt/drive_b/MATS_apply/analogy_vector_llama70b.pt"
    # }
}

DATASET_PATH = "/mnt/drive_b/MATS_apply/analogy_dataset_100.json"
DEVICE = "cuda"

def load_data():
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    return data

def format_instruction(prompt, model):
    """
    Applies the correct chat template for instruction tuned models.
    This ensures the model sees the prompt exactly as it was trained.
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = model.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted

def compute_vector_for_model(model_name, model_path, output_path):
    """Compute analogy vector for a single model."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}\n")
    
    print(f"Loading {model_path} into VRAM...")
    if "70B" in model_path or "70b" in model_path:
        print("Detected 70B model: loading with 4-bit quantization for memory efficiency.")
        model = HookedTransformer.from_pretrained(
            model_path,
            device=DEVICE,
            dtype="bfloat16",
            quantization_config={"load_in_4bit": True}
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_path, 
            device=DEVICE, 
            dtype="bfloat16"
        )
    model.eval()

    data = load_data()
    print(f"Loaded {len(data)} prompt pairs.")

    # Storage for activations: List of tensors [Layers, d_model]
    literal_activations = []
    analogy_activations = []

    print("Extracting activations...")
    
    with torch.no_grad():
        for item in tqdm(data, desc=f"Processing {model_name}"):
            # 1. Format Prompts
            lit_prompt = format_instruction(item['literal_prompt'], model)
            ana_prompt = format_instruction(item['analogy_prompt'], model)

            # 2. Run Literal Prompt
            _, cache_lit = model.run_with_cache(
                lit_prompt, 
                names_filter=lambda x: x.endswith("resid_post")
            )
            
            # 3. Run Analogy Prompt
            _, cache_ana = model.run_with_cache(
                ana_prompt, 
                names_filter=lambda x: x.endswith("resid_post")
            )

            # 4. Extract Last Token Activations across all layers
            lit_act = cache_lit.stack_activation("resid_post")[:, 0, -1, :]
            ana_act = cache_ana.stack_activation("resid_post")[:, 0, -1, :]

            literal_activations.append(lit_act)
            analogy_activations.append(ana_act)

    # 5. Stack and Compute Mean Difference
    literal_tensor = torch.stack(literal_activations)
    analogy_tensor = torch.stack(analogy_activations)

    # Calculate Mean Vectors
    mean_literal = literal_tensor.mean(dim=0)
    mean_analogy = analogy_tensor.mean(dim=0)

    # The Magic Vector: "Analogy" minus "Literal"
    analogy_steering_vector = mean_analogy - mean_literal

    # 6. Save the vector
    torch.save(analogy_steering_vector, output_path)
    print(f"\nVector saved! Shape: {analogy_steering_vector.shape}")
    print(f"Saved to: {output_path}")

    # 7. Analysis: Which layer has the biggest difference?
    diff_norms = analogy_steering_vector.norm(dim=1)
    best_layer = diff_norms.argmax().item()
    max_norm = diff_norms[best_layer].item()
    
    print(f"\nLayer Analysis for {model_name}:")
    print(f"  Layer with largest 'Analogy' divergence: {best_layer}")
    print(f"  Maximum norm: {max_norm:.4f}")
    print(f"  Total layers: {len(diff_norms)}")
    
    # Print top 5 layers
    top_layers = diff_norms.argsort(descending=True)[:5]
    print(f"\n  Top 5 layers by norm:")
    for i, layer in enumerate(top_layers):
        print(f"    {i+1}. Layer {layer.item()}: {diff_norms[layer].item():.4f}")
    
    # 8. Save analysis results
    # Convert norms to float32 to avoid BFloat16 serialization issues
    diff_norms_cpu = diff_norms.cpu().float()  # Convert to float32
    
    analysis_results = {
        "model_name": model_name,
        "model_path": model_path,
        "vector_shape": list(analogy_steering_vector.shape),
        "n_layers": len(diff_norms),
        "layer_norms": diff_norms_cpu.numpy().tolist(),  # Convert to list for JSON
        "best_layer": int(best_layer),
        "max_norm": float(max_norm),
        "mean_norm": float(diff_norms.mean().item()),
        "std_norm": float(diff_norms.std().item()),
        "min_norm": float(diff_norms.min().item()),
        "min_layer": int(diff_norms.argmin().item()),
        "top_5_layers": [int(layer.item()) for layer in top_layers],
        "top_5_norms": [float(diff_norms[layer].item()) for layer in top_layers],
    }
    
    # Save norms as separate .pt file (as float32)
    norms_path = output_path.replace('.pt', '_norms.pt')
    torch.save(diff_norms_cpu, norms_path)
    print(f"  Layer norms saved to: {norms_path}")
    
    # Save analysis results as JSON
    analysis_path = output_path.replace('.pt', '_analysis.json')
    try:
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"  Analysis results saved to: {analysis_path}")
    except Exception as e:
        print(f"  ⚠ Warning: Could not save JSON analysis: {e}")
        print(f"     (This is non-critical - vector and norms are saved)")
    
    # 9. Create and save plot
    plot_path = output_path.replace('.pt', '_norms_plot.png')
    plt.figure(figsize=(12, 6))
    layers = np.arange(len(diff_norms))
    # Use float32 version for plotting
    norms_plot = diff_norms_cpu.numpy()
    plt.plot(layers, norms_plot, marker='o', linewidth=2, markersize=4)
    plt.axvline(x=best_layer, color='red', linestyle='--', alpha=0.7, 
               label=f'Best layer ({best_layer})')
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Euclidean Norm of Analogy Vector", fontsize=12)
    plt.title(f"Analogy Vector Norm by Layer: {model_name.upper()}\n"
             f"Best layer: {best_layer} (norm={max_norm:.4f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {plot_path}")
    
    # Clean up model from memory
    del model
    torch.cuda.empty_cache()
    
    # Return float32 version of norms to avoid BFloat16 issues in main()
    return analogy_steering_vector, diff_norms_cpu, analysis_results

def main():
    """Compute vectors for all models."""
    all_results = {}
    all_analysis = {}
    
    for model_key, model_config in MODELS.items():
        try:
            vector, norms, analysis = compute_vector_for_model(
                model_key,
                model_config["name"],
                model_config["output_file"]
            )
            # Norms are already in float32 from compute_vector_for_model
            all_results[model_key] = {
                "vector": vector,
                "norms": norms,  # Already float32
                "best_layer": int(norms.argmax().item())
            }
            all_analysis[model_key] = analysis
        except Exception as e:
            print(f"\nERROR processing {model_key}: {e}")
            print("Skipping to next model...\n")
            continue
    
    # Compare models if both succeeded
    if len(all_results) >= 2:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        for model_key, results in all_results.items():
            max_norm = float(results['norms'].max().item())
            print(f"{model_key.upper()}: Best layer = {results['best_layer']}, Max norm = {max_norm:.4f}")
        
        # Create comparison plot
        comparison_plot_path = "/mnt/drive_b/MATS_apply/model_comparison_norms.png"
        plt.figure(figsize=(14, 7))
        for model_key, results in all_results.items():
            # Convert to float32 if needed
            norms = results["norms"]
            if isinstance(norms, torch.Tensor):
                norms = norms.cpu().float().numpy()
            else:
                norms = np.array(norms)
            layers = np.arange(len(norms))
            plt.plot(layers, norms, marker='o', linewidth=2, markersize=3, 
                    label=f"{model_key.upper()} (best: layer {results['best_layer']})", alpha=0.8)
        
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel("Euclidean Norm of Analogy Vector", fontsize=12)
        plt.title("Model Comparison: Analogy Vector Norms by Layer", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nComparison plot saved to: {comparison_plot_path}")
        
        # Save combined analysis
        combined_analysis_path = "/mnt/drive_b/MATS_apply/model_comparison_analysis.json"
        try:
            with open(combined_analysis_path, 'w') as f:
                json.dump(all_analysis, f, indent=2)
            print(f"Combined analysis saved to: {combined_analysis_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not save combined analysis: {e}")
    
    print("\n" + "="*60)
    print("COMPUTATION COMPLETE")
    print("="*60)
    print("\nSaved files per model:")
    print("  - analogy_vector_<model>.pt (the steering vector)")
    print("  - analogy_vector_<model>_norms.pt (layer norms tensor)")
    print("  - analogy_vector_<model>_analysis.json (statistics)")
    print("  - analogy_vector_<model>_norms_plot.png (visualization)")
    if len(all_results) >= 2:
        print("  - model_comparison_norms.png (comparison plot)")
        print("  - model_comparison_analysis.json (combined analysis)")

if __name__ == "__main__":
    main()

