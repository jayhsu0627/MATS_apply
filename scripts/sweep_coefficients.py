import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-9b-it"
DEVICE = "cuda"

def main():
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype="bfloat16")
    model.eval()

    # Load vector
    analogy_vector = torch.load("analogy_vector.pt")

    # Focus on the promising layer
    TARGET_LAYER = 30
    layer_vector = analogy_vector[TARGET_LAYER]

    prompt = "Explain how TCP/IP handshakes work."
    formatted_prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Tokenize once for reuse in generation
    tokens = model.to_tokens(formatted_prompt, prepend_bos=False).to(DEVICE)

    # Sweep exponentially-ish: 1, 2, 5, 8, 12
    coeffs = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    print(f"--- Prompt: {prompt} ---")
    print(f"--- Steering at Layer {TARGET_LAYER} ---\n")

    for coeff in coeffs:
        def steering_hook(resid_post, hook):
            # Add to all positions (simplest stable method)
            resid_post += coeff * layer_vector
            return resid_post

        # Use hooks as a context while calling generate
        with model.hooks(fwd_hooks=[(f"blocks.{TARGET_LAYER}.hook_resid_post", steering_hook)]):
            generated_ids = model.generate(tokens, max_new_tokens=60, temperature=0.1)

        generated_text = model.tokenizer.decode(generated_ids[0])
        # Optionally strip the original prompt from the front for readability
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        print(f"[Coeff {coeff}]: {generated_text}")

if __name__ == "__main__":
    main()