import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-9b-it"
DEVICE = "cuda"

def main():
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype="bfloat16")
    model.eval()

    # Load the vector
    analogy_vector = torch.load("/mnt/drive_b/MATS_apply/analogy_vector.pt")

    # TEST PROMPT: Something boring/technical
    test_prompt = "Explain how TCP/IP handshakes work."
    
    # Format properly
    formatted_prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": test_prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Tokenize once for reuse
    tokens = model.to_tokens(formatted_prompt, prepend_bos=False).to(DEVICE)

    print(f"\n--- Baseline (No Steering) ---")
    baseline_ids = model.generate(tokens, max_new_tokens=50, temperature=0)
    baseline_text = model.tokenizer.decode(baseline_ids[0])
    print(baseline_text)

    # Try steering at different layers
    # Layers to test: Early-Mid (20), Mid-Late (30), Late (41)
    target_layers = [20, 30, 41]
    # steering_coefficient = 20.0 # Start strong, decrease if model breaks
    steering_coefficient = 20.0 # Start strong, decrease if model breaks

    for layer in target_layers:
        print(f"\n--- Steering at Layer {layer} (Coeff: {steering_coefficient}) ---")
        
        # Define the hook function
        # We need to grab the vector for THIS specific layer
        layer_vector = analogy_vector[layer]
        
        def steering_hook(resid_post, hook):
            # resid_post shape: [batch, seq_len, d_model]
            # Add vector to ALL tokens (simplest way for stability)
            # Or add only to the last token: resid_post[:, -1, :] += ...
            resid_post += steering_coefficient * layer_vector
            return resid_post

        # Run with hooks using the context manager; we still call model.generate for text
        # Hook name: blocks.{layer}.hook_resid_post
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            steered_ids = model.generate(tokens, max_new_tokens=50, temperature=0)

        steered_text = model.tokenizer.decode(steered_ids[0])
        print(steered_text)

if __name__ == "__main__":
    main()