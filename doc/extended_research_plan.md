This is a fantastic hardware setup. Having an **RTX 6000 Ada/Blackwell with 96GB VRAM** puts you in a unique position: you can run inference on Llama-3-70B (quantized) or run full-precision training/patching experiments on Llama-3-8B/Gemma-2-9B with huge batch sizes locally, without dealing with the latency or cost of API-based remote execution (like NDIF).

Here is the extended research plan, optimized to align with Neel Nanda’s "Research Taste" (pragmatism, baselines, and concept representation) using the "Explore, Understand, Distill" framework.

---

# Research Plan: The Mechanistic Geometry of Analogy

## 0. Prior Knowledge & Hardware Utilization

**Your Advantage:** With 96GB VRAM, you do not need to rely on API-based activation patching (like nnsight's remote server). You can do everything locally using `TransformerLens` or `nnsight` (local mode). This allows for extremely fast feedback loops.

*   **Model Choice:** **Llama-3.1-8B-Instruct** or **Gemma-2-9B-IT**. These models are capable enough to generate high-quality analogies but small enough to fit fully in memory with ample room for caching huge activation batches.
*   **Library:** **TransformerLens** (Neel's library). It is the standard for this type of work and will make your code immediately readable to him.

---

## 1. Strong Motivation (The "Why")

*Reference: "Concept Representations" and "Model Biology" from Neel's interests.*

**The Pitch:**
"Analogy-making is not just a rhetorical device; it is the fundamental mechanism of **zero-shot generalization**. When an LLM explains quantum spin using spinning tops, it is performing a **structure mapping**—extracting an abstract relationship from a source domain and projecting it onto a target domain.

If we can mechanistically locate the **'Abstract Relation Vector'** (the representation of the relationship itself, independent of the specific nouns), we unlock:
1.  **Safety/Steering:** Preventing models from making persuasive but misleading analogies (e.g., in political radicalization).
2.  **Knowledge Transfer:** Understanding how models apply 'reasoning patterns' to out-of-distribution data.

This project investigates if analogies are computed via a **linear subspace** representing the relationship (the 'Structure Mapping Hypothesis') or via simple associative retrieval, and whether we can intervene to break or force analogies."

---

## 2. Data, Tools, and Target

### Dataset: The "Analogy Transport" Dataset (Custom)
You need a clean dataset to isolate the *mechanism*, not just the behavior. Do not use generic text. Construct a dataset of the form $A:B :: C:[D]$.

*   **Structure:** `(Source_Subject, Source_Object, Target_Subject, Target_Object, Relation_Type)`
*   **Example:**
    *   *Relation:* Functional Role
    *   *Prompt:* "The Engine is to the Car what the [Heart] is to the [Body]."
*   **Generation:** Use **Claude 3.5 Sonnet** (via API or web) to generate 1,000 tuples of varying difficulty (Physical, Abstract, Social, Mathematical).
*   **Control:** Generate "False Analogies" where the surface features match but the structure doesn't, to test for robustness.

### Tools
*   **TransformerLens:** For hook points and activation caching.
*   **PyTorch:** For training linear probes.
*   **Cursor (IDE):** Highly recommended by Neel for coding efficiency.

---

## 3. Analysis: The Shortest Path to Neel's Research Taste

Neel explicitly warns against over-complication (e.g., training SAEs immediately). He prefers **baselines** and **simple methods first**. We will use **Linear Probes** and **Activation Patching**.

### The "Explore, Understand, Distill" Plan (20 Hours)

#### Phase 1: Exploration (Hours 1-5) - The Behavioral Baseline
*Goal: Prove the model actually understands the analogy structure vs. simple associations.*

1.  **Zero-Shot Test:** Feed Llama-3-8B prompts like: *"Complete the analogy: Electricity is to Wire as Water is to..."*
2.  **Measure Logits:** Look at the logit rank of "Pipe" (correct) vs. "River" (associative distractor) vs. "Hose".
3.  **Task Vector Extraction (The Toy Example):**
    *   Take the mean difference of activations: $\mu(A:B) - \mu(C:D)$.
    *   Does adding this "Analogy Vector" to a neutral prompt induce analogy-like behavior?
    *   *Neel's Taste Check:* This mimics the "Function Vectors" work. It’s a simple, powerful baseline.

#### Phase 2: Understanding (Hours 6-15) - The Mechanistic Core
*Goal: Locate the circuit. Is the relationship computed in specific heads?*

**Method: Activation Patching (The "Denoising" Experiment)**
*   **Prompt A (Correct):** "Bird is to Air as Fish is to Water."
*   **Prompt B (Corrupted):** "Bird is to Air as Fish is to Sky." (Wrong target domain)
*   **The Experiment:** Run Prompt B. Patch activations from Prompt A into Prompt B layer-by-layer.
*   **Metric:** Logit difference between "Water" and "Sky".
*   **Hardware Advantage:** With 96GB VRAM, you can patch **every layer and head simultaneously** across a batch of 100 examples in seconds. Do not loop one by one; batch it.

**Hypothesis to Test:**
*   **Early Layers:** Resolve the entity concepts (Fish, Water).
*   **Middle Layers:** Compute the *relationship* (Medium/Environment).
*   **Late Layers:** Apply the relationship to the Target Subject (Fish + Environment = Water).

**Linear Probe (The "Geometry" Test):**
Train a linear probe on the middle layers to distinguish between different *types* of analogies (e.g., "Part-to-Whole" vs. "Opposites" vs. "Cause-Effect").
*   *Success Criteria:* If a probe can classify the relationship type with high accuracy *before* the answer is generated, the model represents the abstract structure explicitly.

#### Phase 3: Distillation (Hours 16-20) - The Narrative
*Goal: Write the Executive Summary.*
*   **The Narrative:** "Llama-3 represents abstract analogies linearly in Layers X-Y. We found a 'Relation Vector' that, when patched, forces the model to interpret unrelated concepts through that specific analogy lens."
*   **The Artifact:** A single clear graph showing the "Patching Heatmap" (Layers vs. Token Position).

---

## 4. Specific "First Toy Example" to Try (Hour 1)

Do not start with complex paragraph explanations. Start with the classic **Word Analogy**.

**Code Snippet Concept (TransformerLens):**

```python
# Pseudo-code plan
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device="cuda")

# 1. Define clean and corrupt prompts
# Relation: Capital City
clean_prompt = "Paris is to France as London is to UK"
# Corrupt relationship
corrupt_prompt = "Paris is to France as London is to France" 

# 2. Run with cache
clean_logits, clean_cache = model.run_with_cache(clean_prompt)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompt)

# 3. Metric: Logit diff between "UK" and "France" at the last token position
def metric(logits):
    return logits[0, -1, model.to_single_token(" UK")] - logits[0, -1, model.to_single_token(" France")]

# 4. Patching (The "Neel" way: Activating the circuit)
# Patch residual stream from Clean -> Corrupt at every layer.
# Find where the "Capital City" relationship is moved to the final token.
```

### Potential Barrier & Solution
*   **Barrier:** The model might just be memorizing bigrams (London -> UK) rather than computing the analogy `A:B::C:?`.
*   **Solution:** Use abstract or invented entities in the prompt.
    *   *Prompt:* "In a fictional world, a 'Glorp' lives in a 'Flim'. A 'Zorp' lives in a 'Flam'. A 'Glorp' is to 'Flim' as 'Zorp' is to..."
    *   If the model predicts "Flam", it is doing **In-Context Learning (ICL)** of the relationship, not memorization. This effectively targets **Induction Heads**, a favorite topic of Neel's.

## Summary Checklist for Success
1.  **Don't over-engineer.** Start with word analogies, not paragraphs.
2.  **Use your VRAM.** Batch everything. Run high-resolution patching scans.
3.  **Focus on the "Relation Vector".** Can you isolate the direction that means "Opposite of"?
4.  **Write cleanly.** Use the structure: Question -> Hypothesis -> Experiment -> Result -> Interpretation.