"""
LLM-as-a-Judge Evaluation System

Replaces keyword-based evaluation with LLM scoring of analogy quality.
This provides a much cleaner metric than counting "like", "imagine", etc.
"""
import torch
import json
from transformer_lens import HookedTransformer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Configuration
STUDENT_MODEL_NAME = "google/gemma-2-9b-it"  # Model being evaluated
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"  # Judge model
DEVICE = "cuda"

# Test prompts (neutral, technical prompts)
TEST_PROMPTS = [
    "Explain how TCP/IP handshakes work.",
    "What is photosynthesis?",
    "How does a CPU execute instructions?",
    "Explain quantum entanglement.",
    "What is the difference between DNA and RNA?",
]

# Steering coefficients to test
COEFFICIENTS = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

# Target layer (will be determined from layer analysis)
TARGET_LAYER = 30  # Default, should be updated based on stability analysis

def format_instruction(prompt, model):
    """Format prompt for instruction-tuned model."""
    messages = [{"role": "user", "content": prompt}]
    formatted = model.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted

def judge_analogy_quality(prompt, response, judge_llm):
    """
    Use LLM to judge analogy quality in response.
    
    Returns score 1-5 and reasoning.
    """
    judge_prompt = f"""I will give you a prompt and a response. Rate the response on a scale of 1-5 based on how well it uses an **analogy or metaphor** to explain the concept.

1: No analogy (literal definition only).
2: Minimal analogy (brief mention, not developed).
3: Moderate analogy (some analogy used, but could be better).
4: Good analogy (clear, helpful analogy that aids understanding).
5: Excellent analogy (concrete, well-developed analogy that makes the concept clear).

**Prompt:**
{prompt}

**Response:**
{response}

Output JSON only: {{"score": 3, "reasoning": "..."}}"""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    outputs = judge_llm.generate([judge_prompt], sampling_params)
    judge_output = outputs[0].outputs[0].text.strip()
    
    # Extract JSON
    try:
        # Find JSON in output
        start = judge_output.find('{')
        end = judge_output.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = judge_output[start:end]
            result = json.loads(json_str)
            return result.get("score", 3), result.get("reasoning", "")
    except:
        pass
    
    # Fallback: try to extract score
    try:
        score = int(judge_output.split('"score":')[1].split(',')[0].strip())
        return score, judge_output
    except:
        return 3, judge_output  # Default neutral score

def generate_with_steering(model, prompt, vector, layer, coefficient):
    """Generate text with steering vector applied."""
    formatted = format_instruction(prompt, model)
    tokens = model.to_tokens(formatted, prepend_bos=False).to(DEVICE)
    
    layer_vector = vector[layer]
    
    def steering_hook(resid_post, hook):
        # Add to all positions for stability
        resid_post += coefficient * layer_vector
        return resid_post
    
    hook_name = f"blocks.{layer}.hook_resid_post"
    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        generated_ids = model.generate(tokens, max_new_tokens=100, temperature=0.1)
    
    generated_text = model.tokenizer.decode(generated_ids[0])
    # Remove prompt from output
    if generated_text.startswith(formatted):
        generated_text = generated_text[len(formatted):].strip()
    
    return generated_text

def evaluate_steering(student_model, judge_llm, vector_path, layer, use_clean=True):
    """
    Evaluate steering effectiveness using LLM-as-Judge.
    
    Args:
        student_model: The model being steered
        judge_llm: The judge model (vLLM)
        vector_path: Path to analogy vector
        layer: Layer to apply steering
        use_clean: Whether to use clean (orthogonalized) vector
    """
    # Load vector
    if use_clean:
        vector_path = vector_path.replace('.pt', '_clean.pt')
    
    try:
        vector = torch.load(vector_path)
    except FileNotFoundError:
        print(f"Warning: {vector_path} not found. Using original vector.")
        vector_path = vector_path.replace('_clean.pt', '.pt')
        vector = torch.load(vector_path)
    
    print(f"\n{'='*60}")
    print(f"LLM-as-Judge Evaluation")
    print(f"{'='*60}")
    print(f"Student Model: {STUDENT_MODEL_NAME}")
    print(f"Judge Model: {JUDGE_MODEL_NAME}")
    print(f"Vector: {vector_path}")
    print(f"Layer: {layer}")
    print(f"Test Prompts: {len(TEST_PROMPTS)}")
    print(f"Coefficients: {COEFFICIENTS}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for prompt in tqdm(TEST_PROMPTS, desc="Processing prompts"):
        prompt_results = {"prompt": prompt, "coefficients": []}
        
        for coeff in COEFFICIENTS:
            # Generate with steering
            response = generate_with_steering(
                student_model, prompt, vector, layer, coeff
            )
            
            # Judge the response
            score, reasoning = judge_analogy_quality(prompt, response, judge_llm)
            
            prompt_results["coefficients"].append({
                "coefficient": coeff,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "score": score,
                "reasoning": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
            })
            
            all_results.append({
                "prompt": prompt,
                "coefficient": coeff,
                "score": score
            })
        
        # Print sample for first prompt
        if prompt == TEST_PROMPTS[0]:
            print(f"\nSample (Prompt: '{prompt}'):")
            for result in prompt_results["coefficients"][:3]:
                print(f"  Coeff {result['coefficient']:4.1f}: Score {result['score']}/5")
                print(f"    Response: {result['response'][:100]}...")
    
    # Aggregate results
    scores_by_coeff = {}
    for result in all_results:
        coeff = result["coefficient"]
        if coeff not in scores_by_coeff:
            scores_by_coeff[coeff] = []
        scores_by_coeff[coeff].append(result["score"])
    
    # Calculate means
    coeffs_sorted = sorted(scores_by_coeff.keys())
    mean_scores = [np.mean(scores_by_coeff[c]) for c in coeffs_sorted]
    std_scores = [np.std(scores_by_coeff[c]) for c in coeffs_sorted]
    
    # Plot: The "Money Plot"
    plt.figure(figsize=(10, 6))
    plt.errorbar(coeffs_sorted, mean_scores, yerr=std_scores, 
                marker='o', linewidth=2, markersize=8, capsize=5)
    plt.xlabel("Steering Coefficient", fontsize=12)
    plt.ylabel("Average Analogy Score (1-5)", fontsize=12)
    plt.title(f"Steering Effectiveness: {STUDENT_MODEL_NAME}\n(Layer {layer}, LLM-as-Judge Evaluation)", 
             fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='Neutral (3.0)')
    plt.legend()
    plt.tight_layout()
    
    plot_path = f"/mnt/drive_b/MATS_apply/analogy_score_plot_layer{layer}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    plt.close()
    
    # Save results
    results_path = f"/mnt/drive_b/MATS_apply/judge_scores_layer{layer}.json"
    with open(results_path, 'w') as f:
        json.dump({
            "model": STUDENT_MODEL_NAME,
            "layer": layer,
            "vector_path": vector_path,
            "results": all_results,
            "summary": {
                "coefficients": coeffs_sorted,
                "mean_scores": mean_scores,
                "std_scores": std_scores
            }
        }, f, indent=2)
    print(f"✓ Results saved to: {results_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for coeff, mean_score, std_score in zip(coeffs_sorted, mean_scores, std_scores):
        print(f"Coeff {coeff:5.1f}: {mean_score:.2f} ± {std_score:.2f}")
    
    return {
        "coefficients": coeffs_sorted,
        "mean_scores": mean_scores,
        "std_scores": std_scores
    }

def main():
    """Run LLM-as-Judge evaluation."""
    print("Loading models...")
    
    # Load student model (TransformerLens)
    student_model = HookedTransformer.from_pretrained(
        STUDENT_MODEL_NAME, device=DEVICE, dtype="bfloat16"
    )
    student_model.eval()
    
    # Load judge model (vLLM)
    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    judge_llm = LLM(model=JUDGE_MODEL_NAME, tensor_parallel_size=1, 
                   gpu_memory_utilization=0.90)
    
    # Evaluate with original vector
    vector_path = "/mnt/drive_b/MATS_apply/analogy_vector_gemma.pt"
    results_original = evaluate_steering(
        student_model, judge_llm, vector_path, TARGET_LAYER, use_clean=False
    )
    
    # Try with clean vector if it exists
    try:
        results_clean = evaluate_steering(
            student_model, judge_llm, vector_path, TARGET_LAYER, use_clean=True
        )
        
        # Compare
        print(f"\n{'='*60}")
        print("COMPARISON: Original vs. Clean Vector")
        print(f"{'='*60}")
        for coeff in results_original["coefficients"]:
            idx = results_original["coefficients"].index(coeff)
            orig_score = results_original["mean_scores"][idx]
            clean_score = results_clean["mean_scores"][idx]
            diff = clean_score - orig_score
            print(f"Coeff {coeff:5.1f}: Original={orig_score:.2f}, Clean={clean_score:.2f}, Diff={diff:+.2f}")
    except FileNotFoundError:
        print("\nClean vector not found. Run orthogonalize_vector.py first.")
    
    del student_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

