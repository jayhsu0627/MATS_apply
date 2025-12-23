import torch
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-9b-it"
DEVICE = "cuda"

def main():
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype="bfloat16")
    model.eval()

    # Load vector and grab Layer 30
    analogy_vector = torch.load("analogy_vector.pt")[30]
    
    # Define target tokens we want to measure (The "Analogy" signature)
    # Note: Spaces before words matter in tokenization
    target_words = [" like", " imagine", " similar", " think", " metaphor", " analogy"]
    target_ids = [model.to_single_token(w) for w in target_words]

    # Test Prompt
    prompt = "Explain how a CPU works."
    formatted_prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Sweep coefficients
    coeffs = np.linspace(-2.0, 4.0, 20) # From negative steering to positive
    probs = []

    print("Running quantitative sweep...")
    for coeff in coeffs:
        def steering_hook(resid_post, hook):
            # Add vector to the last token position only for the logic check
            resid_post[:, -1, :] += coeff * analogy_vector
            return resid_post

        # Run forward pass (no generation, just get logits of the first generated token)
        logits = model.run_with_hooks(
            formatted_prompt,
            fwd_hooks=[(f"blocks.30.hook_resid_post", steering_hook)]
        )
        
        # Get probability of target words
        # logits shape: [1, seq_len, d_vocab] -> get last token
        last_token_logits = logits[0, -1, :]
        last_token_probs = torch.softmax(last_token_logits, dim=0)
        
        # Sum probabilities of "like", "imagine", etc.
        target_prob = last_token_probs[target_ids].sum().item()
        probs.append(target_prob)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(coeffs, probs, marker='o')
    plt.title("Effect of Steering on Probability of Analogy Keywords")
    plt.xlabel("Steering Coefficient")
    plt.ylabel("Combined Probability of {'like', 'imagine', 'similar'}")
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid(True)
    plt.savefig("/mnt/drive_b/MATS_apply/steering_quant_plot.png")
    print("Plot saved to steering_quant_plot.png")

if __name__ == "__main__":
    main()