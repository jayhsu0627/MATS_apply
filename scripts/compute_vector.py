import torch
import json
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm

# 1. Configuration
MODEL_NAME = "google/gemma-2-9b-it" 
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
    # We want the prompt text ending right before the model starts generating
    formatted = model.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted

def main():
    print(f"Loading {MODEL_NAME} into VRAM...")
    # HookedTransformer allows us to access internal states easily
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, 
        device=DEVICE, 
        dtype="bfloat16" # Save memory, standard for Llama/Gemma
    )
    model.eval()

    data = load_data()
    print(f"Loaded {len(data)} prompt pairs.")

    # Storage for activations: List of tensors [Layers, d_model]
    literal_activations = []
    analogy_activations = []

    print("Extracting activations...")
    
    # We iterate one by one to avoid padding issues affecting the 'last token' position
    with torch.no_grad():
        for item in tqdm(data):
            # 1. Format Prompts
            lit_prompt = format_instruction(item['literal_prompt'], model)
            ana_prompt = format_instruction(item['analogy_prompt'], model)

            # 2. Run Literal Prompt
            # run_with_cache returns (logits, cache). We only need cache.
            # We want the residual stream state at the end of every block: 'blocks.{layer}.hook_resid_post'
            _, cache_lit = model.run_with_cache(
                lit_prompt, 
                # names_filter=lambda x: x.endswith("hook_resid_post")
                names_filter=lambda x: x.endswith("resid_post")
            )
            
            # 3. Run Analogy Prompt
            _, cache_ana = model.run_with_cache(
                ana_prompt, 
                # names_filter=lambda x: x.endswith("hook_resid_post")
                names_filter=lambda x: x.endswith("resid_post")
            )

            # 4. Extract Last Token Activations across all layers
            # cache.stack_activation returns tensor: [n_layers, batch_size, seq_len, d_model]
            # We grab the LAST token (pos = -1)

            # In TransformerLens, stack_activation takes the *base* name like "resid_post",
            # and internally expands to "blocks.{layer}.hook_resid_post".
            # Extract and squeeze to remove batch dim -> Shape: [n_layers, d_model]
            lit_act = cache_lit.stack_activation("resid_post")[:, 0, -1, :]
            ana_act = cache_ana.stack_activation("resid_post")[:, 0, -1, :]

            literal_activations.append(lit_act)
            analogy_activations.append(ana_act)

    # 5. Stack and Compute Mean Difference
    # Shape: [N_samples, n_layers, d_model]
    literal_tensor = torch.stack(literal_activations)
    analogy_tensor = torch.stack(analogy_activations)

    # Calculate Mean Vectors
    mean_literal = literal_tensor.mean(dim=0)
    mean_analogy = analogy_tensor.mean(dim=0)

    # The Magic Vector: "Analogy" minus "Literal"
    # This vector represents the "direction" of abstracting/simplifying
    analogy_steering_vector = mean_analogy - mean_literal

    # 6. Save the vector
    torch.save(analogy_steering_vector, "/mnt/drive_b/MATS_apply/analogy_vector.pt")
    print(f"Vector saved! Shape: {analogy_steering_vector.shape}")
    print("Shape format: [n_layers, d_model]")

    # 7. Quick Analysis: Which layer has the biggest difference?
    # We calculate the Euclidean norm of the difference vector at each layer
    diff_norms = analogy_steering_vector.norm(dim=1)
    best_layer = diff_norms.argmax().item()
    print(f"\nLayer with largest 'Analogy' divergence: {best_layer}")
    print("You should probably target this layer (and surrounding ones) for steering.")

if __name__ == "__main__":
    main()