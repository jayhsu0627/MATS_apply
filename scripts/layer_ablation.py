"""
Layer Ablation: Test steering effectiveness across all layers.

This creates the "Steering Success vs. Layer" plot mentioned in the plan.
"""
import torch
import numpy as np
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm

MODEL_NAME = "google/gemma-2-9b-it"
DEVICE = "cuda"

TEST_PROMPT = "Explain how TCP/IP handshakes work."
STEERING_COEFFICIENT = 4.0  # Moderate steering strength
LAYER_INTERVAL = 5  # Test every 5 layers (0, 5, 10, ..., 40)

def format_instruction(prompt, model):
    """Format prompt for instruction-tuned model."""
    messages = [{"role": "user", "content": prompt}]
    formatted = model.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted

def count_analogy_keywords(text):
    """Count analogy-related keywords as proxy metric."""
    keywords = ["like", "imagine", "similar", "analogy", "metaphor", "think of", "as if"]
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

def test_layer(model, prompt, vector, layer, coefficient):
    """Test steering at a specific layer."""
    formatted = format_instruction(prompt, model)
    tokens = model.to_tokens(formatted, prepend_bos=False).to(DEVICE)
    
    layer_vector = vector[layer]
    
    def steering_hook(resid_post, hook):
        resid_post += coefficient * layer_vector
        return resid_post
    
    hook_name = f"blocks.{layer}.hook_resid_post"
    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        generated_ids = model.generate(tokens, max_new_tokens=80, temperature=0.1)
    
    generated_text = model.tokenizer.decode(generated_ids[0])
    if generated_text.startswith(formatted):
        generated_text = generated_text[len(formatted):].strip()
    
    keyword_count = count_analogy_keywords(generated_text)
    return keyword_count, generated_text

def main():
    """Run layer ablation study."""
    print("="*60)
    print("LAYER ABLATION STUDY")
    print("="*60)
    
    # Load model
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype="bfloat16")
    model.eval()
    
    # Load vector
    vector_path = "/mnt/drive_b/MATS_apply/analogy_vector_gemma.pt"
    try:
        analogy_vector = torch.load(vector_path)
    except FileNotFoundError:
        vector_path = "/mnt/drive_b/MATS_apply/analogy_vector.pt"
        analogy_vector = torch.load(vector_path)
    
    n_layers = analogy_vector.shape[0]
    print(f"\nModel: {MODEL_NAME}")
    print(f"Total layers: {n_layers}")
    print(f"Test layers: {list(range(0, n_layers, LAYER_INTERVAL))}")
    print(f"Steering coefficient: {STEERING_COEFFICIENT}")
    print(f"Test prompt: {TEST_PROMPT}\n")
    
    # Test baseline (no steering)
    formatted = format_instruction(TEST_PROMPT, model)
    tokens = model.to_tokens(formatted, prepend_bos=False).to(DEVICE)
    baseline_ids = model.generate(tokens, max_new_tokens=80, temperature=0.1)
    baseline_text = model.tokenizer.decode(baseline_ids[0])
    if baseline_text.startswith(formatted):
        baseline_text = baseline_text[len(formatted):].strip()
    baseline_keywords = count_analogy_keywords(baseline_text)
    
    print(f"Baseline (no steering): {baseline_keywords} keywords")
    print(f"Baseline text: {baseline_text[:150]}...\n")
    
    # Test each layer
    test_layers = list(range(0, n_layers, LAYER_INTERVAL))
    if test_layers[-1] != n_layers - 1:
        test_layers.append(n_layers - 1)  # Always include last layer
    
    results = []
    
    for layer in tqdm(test_layers, desc="Testing layers"):
        keyword_count, response = test_layer(
            model, TEST_PROMPT, analogy_vector, layer, STEERING_COEFFICIENT
        )
        results.append({
            "layer": layer,
            "keywords": keyword_count,
            "response": response[:100]
        })
        
        if layer % 10 == 0 or layer == test_layers[-1]:
            print(f"Layer {layer:2d}: {keyword_count} keywords")
    
    # Extract data for plotting
    layers = [r["layer"] for r in results]
    keywords = [r["keywords"] for r in results]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(layers, keywords, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=baseline_keywords, color='gray', linestyle='--', 
               label=f'Baseline ({baseline_keywords} keywords)', alpha=0.7)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Analogy Keywords Count", fontsize=12)
    plt.title(f"Steering Effectiveness by Layer\n(Coef={STEERING_COEFFICIENT}, Prompt: '{TEST_PROMPT[:30]}...')", 
             fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = "/mnt/drive_b/MATS_apply/layer_ablation_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    plt.close()
    
    # Find best layer
    best_idx = np.argmax(keywords)
    best_layer = layers[best_idx]
    best_keywords = keywords[best_idx]
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline keywords: {baseline_keywords}")
    print(f"Best layer: {best_layer} ({best_keywords} keywords)")
    print(f"Improvement: +{best_keywords - baseline_keywords} keywords")
    
    # Print top 3 layers
    sorted_results = sorted(results, key=lambda x: x["keywords"], reverse=True)
    print(f"\nTop 3 layers:")
    for i, r in enumerate(sorted_results[:3]):
        print(f"  {i+1}. Layer {r['layer']:2d}: {r['keywords']} keywords")
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

