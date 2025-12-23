"""
Sanity Checks: Baseline tests to verify steering is specific and meaningful.

Tests:
1. Random vector baseline - does any vector work?
2. Simplicity confounder - is it just "simpler language"?
3. Negative steering - does subtracting remove analogies?
4. Clean vs. dirty vector comparison - does orthogonalization help?

Based on empirical validation:
- Gemma: Optimal layers 29 (best) or 35 (stable region)
- Llama: Optimal layers 24 (best) or 28 (stable region)
- Validated coefficients: 0.6-1.0 (no mode collapse)
"""
import torch
import numpy as np
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import json

# Configuration: Support multiple models
MODELS = {
    "gemma": {
        "name": "google/gemma-2-9b-it",
        "vector_path": "/mnt/drive_b/MATS_apply/analogy_vector_gemma.pt",
        "clean_vector_path": "/mnt/drive_b/MATS_apply/analogy_vector_gemma_clean.pt",
        "optimal_layers": [29, 35],  # Empirically best (29) and stable region middle (35)
    },
    "llama8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "vector_path": "/mnt/drive_b/MATS_apply/analogy_vector_llama8b.pt",
        "clean_vector_path": "/mnt/drive_b/MATS_apply/analogy_vector_llama8b_clean.pt",
        "optimal_layers": [24, 28],  # Empirically best (24) and stable region middle (28)
    }
}

DEVICE = "cuda"

# Test prompts
TECHNICAL_PROMPT = "Explain how TCP/IP handshakes work."
POETRY_PROMPT = "Write a short poem about the ocean."
ANALOGY_PROMPT = "Explain DNS using an analogy."  # Should already use analogy
# Better prompt for negative steering test (doesn't request analogy)
LITERAL_PROMPT = "Explain DNS technically."  # Should NOT use analogy by default

# Validated coefficients (from layer ablation: 0.6-1.0 work without mode collapse)
# Updated based on findings: need more granular testing around validated range
COEFFICIENTS = [-1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # More granular around validated range
VALIDATED_COEFFICIENTS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]  # Focus on validated range + transition to collapse
MODE_COLLAPSE_TEST_COEFFS = [2.0, 3.0, 4.0, 6.0]  # Test mode collapse range

def format_instruction(prompt, model):
    """Format prompt for instruction-tuned model."""
    messages = [{"role": "user", "content": prompt}]
    formatted = model.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted

def generate_with_steering(model, prompt, vector, layer, coefficient):
    """Generate text with steering vector applied."""
    formatted = format_instruction(prompt, model)
    tokens = model.to_tokens(formatted, prepend_bos=False).to(DEVICE)
    
    def steering_hook(resid_post, hook):
        resid_post += coefficient * vector
        return resid_post
    
    hook_name = f"blocks.{layer}.hook_resid_post"
    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        generated_ids = model.generate(tokens, max_new_tokens=80, temperature=0.1)
    
    generated_text = model.tokenizer.decode(generated_ids[0])
    if generated_text.startswith(formatted):
        generated_text = generated_text[len(formatted):].strip()
    
    return generated_text

def count_analogy_keywords(text):
    """Count analogy-related keywords as proxy metric."""
    keywords = ["like", "imagine", "similar", "analogy", "metaphor", "think of", "as if"]
    text_lower = text.lower()
    count = sum(1 for kw in keywords if kw in text_lower)
    return count

def test_random_baseline(model, analogy_vector, layer, test_prompt, model_name=""):
    """Test if random vector with same norm produces similar effects."""
    print("\n" + "="*60)
    print("TEST 1: Random Vector Baseline")
    print("="*60)
    
    # Get analogy vector for target layer
    analogy_vec = analogy_vector[layer]
    analogy_norm = analogy_vec.norm().item()
    
    # Generate random vector with same norm
    d_model = analogy_vec.shape[0]
    random_vec = torch.randn(d_model, device=DEVICE, dtype=analogy_vec.dtype)
    random_vec = random_vec / random_vec.norm() * analogy_norm
    
    print(f"Analogy vector norm: {analogy_norm:.4f}")
    print(f"Random vector norm: {random_vec.norm().item():.4f}")
    print(f"Layer: {layer}")
    print(f"\nTest prompt: {test_prompt}\n")
    
    results = {"analogy": [], "random": []}
    
    # Test with more granular coefficients around validated range
    test_coeffs = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]  # More granular testing
    
    for coeff in test_coeffs:
        # Test with analogy vector
        analogy_response = generate_with_steering(
            model, test_prompt, analogy_vec, layer, coeff
        )
        analogy_keywords = count_analogy_keywords(analogy_response)
        results["analogy"].append({
            "coeff": coeff,
            "keywords": analogy_keywords,
            "response": analogy_response[:150]
        })
        
        # Test with random vector
        random_response = generate_with_steering(
            model, test_prompt, random_vec, layer, coeff
        )
        random_keywords = count_analogy_keywords(random_response)
        results["random"].append({
            "coeff": coeff,
            "keywords": random_keywords,
            "response": random_response[:150]
        })
        
        print(f"Coeff {coeff:4.2f}:")
        print(f"  Analogy vector: {analogy_keywords} keywords")
        print(f"  Random vector:  {random_keywords} keywords")
        print()
    
    # Plot comparison
    analogy_keywords = [r["keywords"] for r in results["analogy"]]
    random_keywords = [r["keywords"] for r in results["random"]]
    coeffs = [r["coeff"] for r in results["analogy"]]
    
    plt.figure(figsize=(8, 5))
    plt.plot(coeffs, analogy_keywords, marker='o', label='Analogy Vector', linewidth=2)
    plt.plot(coeffs, random_keywords, marker='s', label='Random Vector', linewidth=2, linestyle='--')
    plt.xlabel("Steering Coefficient")
    plt.ylabel("Analogy Keywords Count")
    plt.title(f"Random Vector Baseline Test\n{model_name} - Layer {layer}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = f"/mnt/drive_b/MATS_apply/sanity_random_baseline_{model_name}_layer{layer}.png"
    plt.savefig(output_path, dpi=150)
    print(f"✓ Plot saved: {output_path}")
    plt.close()
    
    return results

def test_simplicity_confounder(model, analogy_vector, layer, model_name=""):
    """Test if vector just makes text simpler vs. forcing analogies."""
    print("\n" + "="*60)
    print("TEST 2: Simplicity Confounder")
    print("="*60)
    print("Applying vector to poetry - does it simplify or add metaphors?\n")
    print(f"Layer: {layer}\n")
    
    analogy_vec = analogy_vector[layer]
    results = []
    
    # Test with validated range + transition to mode collapse
    test_coeffs = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    for coeff in test_coeffs:
        response = generate_with_steering(
            model, POETRY_PROMPT, analogy_vec, layer, coeff
        )
        keyword_count = count_analogy_keywords(response)
        word_count = len(response.split())
        
        results.append({
            "coeff": coeff,
            "response": response,
            "keywords": keyword_count,
            "word_count": word_count
        })
        
        print(f"Coeff {coeff:4.1f}:")
        print(f"  Keywords: {keyword_count}, Words: {word_count}")
        print(f"  Response: {response[:200]}...")
        print()
    
    # Check if keyword count increases (analogy) vs. word count decreases (simplification)
    baseline_keywords = results[0]["keywords"]
    baseline_words = results[0]["word_count"]
    
    print("Analysis:")
    for r in results[1:]:
        keyword_change = r["keywords"] - baseline_keywords
        word_change = r["word_count"] - baseline_words
        print(f"  Coeff {r['coeff']:4.1f}: Keywords {keyword_change:+d}, Words {word_change:+d}")
    
    return results

def test_negative_steering(model, analogy_vector, layer, model_name=""):
    """Test if subtracting vector removes analogies from literal prompts."""
    print("\n" + "="*60)
    print("TEST 3: Negative Steering")
    print("="*60)
    print("Testing bidirectional effect:")
    print("  - Positive coeff on literal prompt: should ADD analogies")
    print("  - Negative coeff on literal prompt: should REMOVE analogies (if any)\n")
    print(f"Layer: {layer}\n")
    
    analogy_vec = analogy_vector[layer]
    results = []
    
    # Use LITERAL_PROMPT (doesn't request analogy) to better test bidirectional effect
    # Test bidirectional effect with literal prompt (doesn't request analogy)
    test_coeffs = [-1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    for coeff in test_coeffs:
        response = generate_with_steering(
            model, LITERAL_PROMPT, analogy_vec, layer, coeff
        )
        keyword_count = count_analogy_keywords(response)
        
        results.append({
            "coeff": coeff,
            "response": response,
            "keywords": keyword_count
        })
        
        print(f"Coeff {coeff:5.1f}: {keyword_count} keywords")
        print(f"  {response[:150]}...")
        print()
    
    # Plot
    coeffs = [r["coeff"] for r in results]
    keywords = [r["keywords"] for r in results]
    
    plt.figure(figsize=(8, 5))
    plt.plot(coeffs, keywords, marker='o', linewidth=2)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='No steering')
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    plt.xlabel("Steering Coefficient (negative = subtracting analogy direction)")
    plt.ylabel("Analogy Keywords Count")
    plt.title(f"Bidirectional Steering Test\n{model_name} - Layer {layer}\n(Literal prompt: should increase with positive, decrease with negative)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = f"/mnt/drive_b/MATS_apply/sanity_negative_steering_{model_name}_layer{layer}.png"
    plt.savefig(output_path, dpi=150)
    print(f"✓ Plot saved: {output_path}")
    plt.close()
    
    # Analysis
    baseline_keywords = results[len(results)//2]["keywords"] if len(results) > 0 else 0  # Coeff 0.0
    print("Analysis:")
    for r in results:
        if r["coeff"] < 0:
            change = r["keywords"] - baseline_keywords
            print(f"  Coeff {r['coeff']:5.1f} (negative): {r['keywords']} keywords (change: {change:+d})")
    for r in results:
        if r["coeff"] > 0:
            change = r["keywords"] - baseline_keywords
            print(f"  Coeff {r['coeff']:5.1f} (positive): {r['keywords']} keywords (change: {change:+d})")
    
    return results

def test_clean_vs_dirty(model, dirty_vector, clean_vector, layer, test_prompt, model_name=""):
    """Test if clean vector reduces mode collapse while preserving analogies."""
    print("\n" + "="*60)
    print("TEST 4: Clean vs. Dirty Vector Comparison")
    print("="*60)
    print("Does orthogonalization reduce mode collapse while preserving analogies?\n")
    print(f"Layer: {layer}\n")
    
    dirty_vec = dirty_vector[layer]
    clean_vec = clean_vector[layer]
    
    results = {"dirty": [], "clean": []}
    
    for coeff in VALIDATED_COEFFICIENTS + MODE_COLLAPSE_TEST_COEFFS:  # Include validated range + mode collapse range
        # Test dirty vector
        dirty_response = generate_with_steering(
            model, test_prompt, dirty_vec, layer, coeff
        )
        dirty_keywords = count_analogy_keywords(dirty_response)
        # Check for mode collapse (repetitive "Imagine")
        imagine_count = dirty_response.lower().count("imagine")
        mode_collapse = imagine_count > 3  # More than 3 "imagine" = potential collapse
        
        results["dirty"].append({
            "coeff": coeff,
            "keywords": dirty_keywords,
            "imagine_count": imagine_count,
            "mode_collapse": mode_collapse,
            "response": dirty_response[:150]
        })
        
        # Test clean vector
        clean_response = generate_with_steering(
            model, test_prompt, clean_vec, layer, coeff
        )
        clean_keywords = count_analogy_keywords(clean_response)
        clean_imagine_count = clean_response.lower().count("imagine")
        clean_mode_collapse = clean_imagine_count > 3
        
        results["clean"].append({
            "coeff": coeff,
            "keywords": clean_keywords,
            "imagine_count": clean_imagine_count,
            "mode_collapse": clean_mode_collapse,
            "response": clean_response[:150]
        })
        
        print(f"Coeff {coeff:4.1f}:")
        print(f"  Dirty: {dirty_keywords} keywords, {imagine_count} 'imagine', collapse={mode_collapse}")
        print(f"  Clean: {clean_keywords} keywords, {clean_imagine_count} 'imagine', collapse={clean_mode_collapse}")
        print()
    
    # Plot comparison
    coeffs = [r["coeff"] for r in results["dirty"]]
    dirty_keywords = [r["keywords"] for r in results["dirty"]]
    clean_keywords = [r["keywords"] for r in results["clean"]]
    dirty_imagine = [r["imagine_count"] for r in results["dirty"]]
    clean_imagine = [r["imagine_count"] for r in results["clean"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Keywords comparison
    ax1.plot(coeffs, dirty_keywords, marker='o', label='Dirty Vector', linewidth=2)
    ax1.plot(coeffs, clean_keywords, marker='s', label='Clean Vector', linewidth=2, linestyle='--')
    ax1.set_xlabel("Steering Coefficient")
    ax1.set_ylabel("Analogy Keywords Count")
    ax1.set_title(f"Keywords: Clean vs. Dirty\n{model_name} - Layer {layer}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # "Imagine" count comparison
    ax2.plot(coeffs, dirty_imagine, marker='o', label='Dirty Vector', linewidth=2)
    ax2.plot(coeffs, clean_imagine, marker='s', label='Clean Vector', linewidth=2, linestyle='--')
    ax2.axhline(y=3, color='red', linestyle=':', alpha=0.5, label='Mode collapse threshold')
    ax2.set_xlabel("Steering Coefficient")
    ax2.set_ylabel("'Imagine' Count")
    ax2.set_title(f"Mode Collapse Check\n{model_name} - Layer {layer}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"/mnt/drive_b/MATS_apply/sanity_clean_vs_dirty_{model_name}_layer{layer}.png"
    plt.savefig(output_path, dpi=150)
    print(f"✓ Plot saved: {output_path}")
    plt.close()
    
    return results

def run_sanity_checks_for_model(model_key, model_config):
    """Run all sanity checks for a single model."""
    model_name = model_config["name"]
    vector_path = model_config["vector_path"]
    clean_vector_path = model_config["clean_vector_path"]
    optimal_layers = model_config["optimal_layers"]
    
    print("\n" + "="*70)
    print(f"SANITY CHECKS: {model_key.upper()}")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Optimal layers: {optimal_layers}")
    print("="*70)
    
    # Load model
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE, dtype="bfloat16")
    model.eval()
    
    # Load vectors
    try:
        dirty_vector = torch.load(vector_path)
        print(f"✓ Loaded dirty vector: {vector_path}")
    except FileNotFoundError:
        print(f"✗ Dirty vector not found: {vector_path}")
        del model
        torch.cuda.empty_cache()
        return None
    
    clean_vector = None
    try:
        clean_vector = torch.load(clean_vector_path)
        print(f"✓ Loaded clean vector: {clean_vector_path}")
    except FileNotFoundError:
        print(f"⚠ Clean vector not found: {clean_vector_path}")
        print("   Will skip clean vs. dirty comparison")
    
    # Use empirically best layer (first in optimal_layers)
    TARGET_LAYER = optimal_layers[0]
    
    print(f"\nTarget Layer: {TARGET_LAYER} (empirically best)")
    print(f"Validated coefficients: {VALIDATED_COEFFICIENTS}\n")
    
    all_results = {
        "model": model_key,
        "model_name": model_name,
        "target_layer": TARGET_LAYER,
        "optimal_layers": optimal_layers
    }
    
    # Test 1: Random baseline
    all_results["random_baseline"] = test_random_baseline(
        model, dirty_vector, TARGET_LAYER, TECHNICAL_PROMPT, model_key
    )
    
    # Test 2: Simplicity confounder
    all_results["simplicity"] = test_simplicity_confounder(
        model, dirty_vector, TARGET_LAYER, model_key
    )
    
    # Test 3: Negative steering
    all_results["negative"] = test_negative_steering(
        model, dirty_vector, TARGET_LAYER, model_key
    )
    
    # Test 4: Clean vs. dirty (if clean vector exists)
    if clean_vector is not None:
        all_results["clean_vs_dirty"] = test_clean_vs_dirty(
            model, dirty_vector, clean_vector, TARGET_LAYER, TECHNICAL_PROMPT, model_key
        )
    
    del model
    torch.cuda.empty_cache()
    
    return all_results

def main():
    """Run all sanity checks for all configured models."""
    print("="*70)
    print("SANITY CHECKS: Verifying Steering Specificity")
    print("="*70)
    print("\nThis will test:")
    print("  1. Random vector baseline (specificity)")
    print("  2. Simplicity confounder (analogy vs. simplification)")
    print("  3. Negative steering (bidirectional effect)")
    print("  4. Clean vs. dirty vector (orthogonalization benefit)")
    print("\nUsing empirically validated layers and coefficients.\n")
    
    all_model_results = {}
    
    for model_key, model_config in MODELS.items():
        try:
            results = run_sanity_checks_for_model(model_key, model_config)
            if results:
                all_model_results[model_key] = results
        except Exception as e:
            print(f"\n✗ Error processing {model_key}: {e}")
            print("Skipping to next model...\n")
            continue
    
    # Save all results
    if all_model_results:
        results_path = "/mnt/drive_b/MATS_apply/sanity_check_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_model_results, f, indent=2, default=str)
        print(f"\n✓ All results saved to: {results_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SANITY CHECK SUMMARY")
    print("="*70)
    print("\n1. Random Baseline:")
    print("   ✓ If analogy vector >> random vector, steering is specific")
    
    print("\n2. Simplicity Confounder:")
    print("   ✓ If keywords increase on poetry, it's forcing analogies")
    print("   ✓ If only word count decreases, it's just simplification")
    
    print("\n3. Negative Steering:")
    print("   ✓ If negative coefficients reduce keywords, effect is bidirectional")
    
    print("\n4. Clean vs. Dirty Vector:")
    print("   ✓ If clean vector reduces 'imagine' count but preserves keywords,")
    print("     orthogonalization successfully isolated concept from tokens")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

