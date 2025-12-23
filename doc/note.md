tested (start from 5:30 am)
1. download two models
2. ask a simple analogy "Complete the analogy: Electricity is to Wire as Water is to...", both answer 'pipe'

prepare datasets: 
1. 100 pairs (generate_date.py)
2. model 'Qwen/Qwen2.5-72B-Instruct-AWQ'

'This synthetic dataset looks excellent. It perfectly fits the "Literal vs. Explanatory Analogy" criteria required for finding the steering vector.
Here is a quick quality audit of why these work well for your specific mechanistic interpretability project:
1. Clear Separation of "Mode"
* Literal: The responses are dry, academic, and definition-heavy ("process by which," "phenomenon where," "set of rules"). This activates the model's "encyclopedic retrieval" circuits.
* Analogy: The responses consistently use structural mapping markers ("is like," "similar to," "just as"). This activates the "abstraction/mapping" circuits.
* Why this matters: When you subtract the mean of the Literal activations from the Analogy activations, the "content" (e.g., Biology vs. Physics) will largely cancel out because it is present in both prompts. What remains is the pure "Analogy Direction."
2. Diverse Domains
You have successfully covered Biology (Photosynthesis), Physics (Quantum Entanglement), Math (Euler's Identity), and CS (Binary Code).
* Why this matters: This ensures the vector you find isn't just a "Science Explanation Vector" or a "Coding Vector." It forces the vector to represent the general abstract capability of analogy, rather than domain-specific knowledge.
3. Concrete "Source Domains"
The analogies map to very distinct, concrete domains: Kitchens, Dice, Trains, Recipes, Light Switches.
* Why this matters: This prevents the model from cheating by using a "near" analogy (e.g., explaining DNA using RNA). It forces a "far" transfer (Biology -> Card Games), which requires a deeper level of abstraction in the residual stream.
4. Zero "Leakage"
I checked the literal responses. None of them accidentally slip into metaphor.'



3. then captured some activations, saved as pt (compute_vector.py)



then do the steering (start with steering strength 20.0, maybetoo high) (test_steering.py and sweep_coefficients.py)
test from -0.5 to 20, and the next token significantly changed after strength=1


==pause at 7:45 ==


==start at 15:17 ==
1. do measure two models
2. do stability analysis and comparison (compute_vector_multi.py) (layer_stability_analysis.py)
3. layer_ablation 

we decrease from 0.8 or +1.6 to 0.5 -> (0.6 1.0) (since has `imagine` like keywords existed in each layer)
STEERING_COEFFICIENT = {
    "gemma": 0.5,      # Safe for Gemma (below 1.0 threshold)
    "llama8b": 0.5,    # Llama may handle higher coefficients (test if needed)
}


