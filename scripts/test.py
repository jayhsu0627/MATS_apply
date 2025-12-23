import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose which model you want to run: "gemma" (Gemma-2-9B-IT) or "llama" (Llama-3.1-70B-Instruct)
MODEL_CHOICE = "gemma"  # or "llama"
# MODEL_CHOICE = "llama"  # or "llama"
MODEL_CHOICE = "llama-instruct"

if MODEL_CHOICE == "gemma":
    model_id = "google/gemma-2-9b-it"
elif MODEL_CHOICE == "llama":
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
elif MODEL_CHOICE == "llama-instruct":
    model_id = "meta-llama/Llama-3.1-70B-Instruct"
else:
    raise ValueError("MODEL_CHOICE must be 'gemma' or 'llama'")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_id} on {device} ...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Define a short dialogue
messages = [
    {
        "role": "user",
        "content": (
            "Complete the analogy: Electricity is to Wire as Water is to..."
            # "Hi! Can you explain what mechanistic interpretability is in 2–3 sentences?"
        ),
    },
]
# Use the model’s chat template (works for both Gemma-2-9B-IT and Llama-3.1-8B-Instruct)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== Full model output ===\n")
print(generated_text)

# Optionally, you can just print the assistant’s last turn by splitting on the user prompt