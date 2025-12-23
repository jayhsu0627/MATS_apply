This is a robust project direction. It fits squarely into Neel’s **"Concept Representations"** and **"Model Biology"** interests while leveraging your powerful hardware to skip the "toy model" phase and go straight to frontier open-weights models (like Gemma-2 or Llama-3).

Here is a 20-hour research plan tailored to your hardware and Neel’s specific research taste (pragmatism, causality, and simplicity).

---

# Project Proposal: Mechanistic Localization of the "Analogy-Making" Direction

### 0. Hardware & Environment Check
**Your Advantage:** An RTX 6000 Ada/Blackwell (96GB VRAM) is a massive advantage.
*   **Capability:** You can load **Llama-3-8B**, **Gemma-2-9B**, or even **Llama-3-70B (quantized)** entirely in VRAM with context.
*   **Implication for Plan:** Do *not* waste time analyzing tiny toy models (like 2-layer transformers) unless necessary. Neel prefers inspecting real models if compute allows. You can do "real" research on a "real" model immediately.
*   **Tools:**
    *   **Library:** **TransformerLens** is mandatory (Neel wrote it). It fits perfectly with your hardware.
    *   **LLM Assistant:** Use Claude 3.5 Sonnet or GPT-4o to write your boilerplate code and generate your synthetic dataset (see below).

---

### 1. Strong Motivation (The "Why")
*To capture Neel's interest, we must frame this as a scientific inquiry into how models represent abstract concepts, with safety implications.*

**The Pitch:**
"Large Language Models are exceptionally good at simplifying complex technical concepts using analogies (e.g., explaining 'current' as 'water flow'). This requires mapping a high-dimensional, abstract concept onto a lower-dimensional, familiar schema. **Is this 'Analogy-Making' capability mediated by a specific direction or subspace in the residual stream?** If we can isolate this direction, can we causally steer a model to be more explanatory or, conversely, detect when a model is using a false analogy to manipulate a user?"

**Safety Relevance:** Deceptive alignment often involves persuasion. Analogies are a primary tool of persuasion. Understanding the mechanism of analogy generation is understanding the mechanism of simplification and potentially manipulation.

---

### 2. Dataset, Tools, and Target Model

**Target Model:** **Gemma-2-9B-IT** (Instruction Tuned).
*   *Why:* Neel’s team recently released [Gemma Scope](https://neuronpedia.org/gemma-scope), identifying millions of features for this model family. It is the current "pet model" for his lab.

**The Dataset (You must build this, Phase 1):**
You need a **paired synthetic dataset** to find the difference between "Literal Explanation" and "Analogical Explanation."
*   Use GPT-4/Claude to generate 100 entries of the following JSON structure:
    ```json
    {
      "concept": "DNS propagation",
      "literal_prompt": "Explain DNS propagation technically.",
      "analogy_prompt": "Explain DNS propagation using an analogy.",
      "literal_response": "It is the time it takes for DNS records to update across servers...",
      "analogy_response": "It is like updating a phone book in every library in the world..."
    }
    ```

---

### 3. Methodology: The Shortest Path to "Research Taste"
*Neel values **Causal Interventions** (Steering) over pure Probing. Probing tells you a feature exists; Steering tells you the model uses it.*

**We will use "Activation Steering" (specifically Mean-Difference Steering).** This is simpler than training a probe and often more robust for a 20-hour sprint.

#### The Draft Plan (20 Hours)

**Phase 1: Exploration & Data (Hours 0-5)**
1.  **Generate Data:** Use an LLM to generate 50–100 pairs of (Complex Topic, Literal Prompt, Analogy Prompt).
2.  **Run Inference:** Run Gemma-2-9B through TransformerLens on these pairs.
3.  **Cache Activations:** Cache the residual stream activations at the *last token of the prompt* (right before generation starts).
4.  **Compute the "Analogy Vector":** Calculate the mean difference:
    $$ \vec{v}_{analogy} = \text{Mean}(\text{Activations}_{\text{Analogy}}) - \text{Mean}(\text{Activations}_{\text{Literal}}) $$
    *   Do this per layer. You will likely find a specific layer range (e.g., middle layers) where this vector norm is highest.

**Phase 2: The "Model Biology" Experiment (Hours 6-12)**
*This is the core "Science."*
1.  **The Intervention:** Take a **neutral** prompt (e.g., "Explain how a CPU works").
2.  **Add the Vector:** Inject $\vec{v}_{analogy}$ into the residual stream at the critical layer identified in Phase 1.
    *   *Formula:* $Act_{new} = Act_{old} + \alpha \cdot \vec{v}_{analogy}$ (where $\alpha$ is a steering coefficient, try 5.0, 10.0, etc.).
3.  **Observe Generation:** Does the model spontaneously generate an analogy *without* being asked for one?
    *   *Success Criteria:* If the model starts talking about "brains" or "traffic cops" when explaining a CPU, you have found the mechanism.

**Phase 3: Rigor & Sanity Checks (Hours 13-17)**
*Neel looks for skepticism. Don't just show it works; try to break it.*
1.  **The "Simplicity" Confounder:** Is your vector just a "dumb down" vector?
    *   *Test:* Apply the vector to a request for a poem. Does it make the poem simple, or does it force metaphors?
2.  **Layer Specificity:** Does applying this vector at Layer 0 work? (Probably not). Does Layer 20 work? (Maybe too late). Plot a graph of "Steering Success vs. Layer."
3.  **Baseline:** Compare against a random vector. Does adding random noise also cause analogies? (Unlikely, but you must prove it).

**Phase 4: Distillation (Hours 18-20)**
1.  **Executive Summary:** Write the 1-page summary.
2.  **The "Money Plot":** Create one graph showing the probability of "analogy-like words" (like *like, as, imagine, similar to*) increasing as you increase the steering coefficient $\alpha$.

---

### 4. The "Toy Example" to Start With

If you want to verify your pipeline in the first hour, try this simple setup:

**Task:** Sentiment Steering (The "Hello World" of steering).
1.  **Data:** 10 prompts saying "I love this" and 10 saying "I hate this."
2.  **Vector:** `Mean(Positive) - Mean(Negative)`.
3.  **Test:** Feed the prompt "I think this movie is" and add the vector.
    *   If it auto-completes "amazing," your code works.
    *   If it auto-completes "terrible," you subtracted instead of added.
    *   If it outputs garbage, your layer selection or coefficient is wrong.

**Once this works, swap the data for your "Analogy vs. Literal" dataset.**

### Why this fits Neel's Taste:
1.  **It's Mechanistic:** You aren't just prompting; you are manipulating the internal linear algebra of the transformer.
2.  **It's "Model Biology":** You are treating the model like a specimen, poking it to see how it reacts.
3.  **It avoids training:** Training probes takes time and compute (even with your GPU). Mean-difference steering is instant and mathematically cleaner for a short project.
4.  **It addresses "Representation":** It tests the hypothesis that "Analogical Reasoning" is a linear feature in the residual stream.