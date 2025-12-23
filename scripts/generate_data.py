import json
from vllm import LLM, SamplingParams

# 1. Setup the "Teacher" Model (Llama-3.1-70B)
# Note: With 96GB VRAM, you can try loading in 8-bit or 4-bit.
# If you don't want to deal with quantization config right now, 
# 'Qwen/Qwen2.5-72B-Instruct-AWQ' (4-bit) is a great out-of-the-box option.
model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ" 

print(f"Loading {model_name} onto GPU...")
llm = LLM(model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.90)


# outputs = llm.generate(prompts, sampling_params)

# # 4. Save
# generated_text = outputs[0].outputs[0].text
# # Basic cleaning to ensure JSON validity if the model chatters
# start_idx = generated_text.find('[')
# end_idx = generated_text.rfind(']') + 1
# clean_json = generated_text[start_idx:end_idx]

# with open("/mnt/drive_b/MATS_apply/analogy_dataset.json", "w") as f:
#     f.write(clean_json)

# print("Dataset generated and saved to analogy_dataset.json")


domains = [
    "Computer Science and Cybersecurity",
    "Quantum Physics and Cosmology",
    "Cellular Biology and Genetics",
    "Macroeconomics and Finance",
    "Linear Algebra and Calculus"
]

all_data = []

for domain in domains:
    print(f"Generating concepts for: {domain}...")
    
    # Update prompt to force specific domain
    system_prompt = "You are a dataset generator. Output valid JSON lists only."
    # Build the prompt without f-string braces conflicts
    user_prompt = (
        f"Generate 20 unique complex concepts specifically related to **{domain}**.\n"
        "For each, provide a literal explanation and an analogy explanation.\n"
        "Format:\n"
        "[\n"
        "{\n"
        '    "concept": "DNS",\n'
        '    "literal_prompt": "Define DNS technically.",\n'
        '    "analogy_prompt": "Explain DNS using a phonebook analogy.",\n'
        '    "literal_response": "DNS translates domain names into IP addresses so browsers can load internet resources.",\n'
        '    "analogy_response": "DNS is like the contacts app on your phone; you click a name (domain), and it looks up the actual phone number (IP address) to make the call."\n'
        "}\n"
        "]\n"
        "Make sure the topics are diverse (Physics, CS, Math, Bio). Do not repeat concepts.\n"
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=4000)

    prompts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"]
    
    outputs = llm.generate(prompts, sampling_params)
    text = outputs[0].outputs[0].text
    
    # Extract JSON
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        batch_json = json.loads(text[start:end])
        all_data.extend(batch_json)
    except Exception as e:
        print(f"Error parsing batch for {domain}: {e}")

# Save combined 100 prompts
with open("/mnt/drive_b/MATS_apply/analogy_dataset_100.json", "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Done! Generated {len(all_data)} pairs.")