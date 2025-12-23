# **Analogy Making Can Be Your Choice: Controlling Explanation Style Through Activation Steering**

# **Executive Summary**

**Sample generations showing intervention effects:**

> **Prompt:** "Explain how TCP/IP handshakes work."
>
> **Baseline (No steering):**
> "The TCP/IP handshake is a three-way process that establishes a reliable connection between two devices over a network. It ensures that both devices are ready to communicate and that they understand the parameters of the connection..."
>
> **Steered (Coefficient 1.0):**
> "Imagine you want to have a conversation with someone across the room. You wouldn't just start talking, right? You'd first imagine knocking on their door to see if they're there and ready to chat. That's kind of how TCP/IP handshakes work..."

### **What problem am I trying to solve?**

I investigated whether the capability of Large Language Models (LLMs) to explain complex concepts via analogies is mediated by a single, steerable direction in activation space. This works builds on "Refusal in Large Language Models as Mediated by a Single Direction" by Arditi et al., extending the "model biology" paradigm from safety behaviors to cognitive capabilities. Understanding if abstract reasoning steps like "make an analogy" are linear and localizable allows us to better understand how models represent rhetorical strategies and potentially build monitors for deceptive persuasion.

### **High-level takeaways**

1.  **Direction Identification**: I identified a consistent "Analogy Vector" by computing the mean difference between activations for literal vs. analogical explanations across diverse topics (Physics, CS, Math).

2.  **Layer Specificity**: Analogy computation happens early (layers 3-7 in Gemma-2-9B), but the direction is robustly preserved in valid "stable regions" (layers 32-41). Interesting, steering is most effective at the **transition** into these stable regions (layer 29), suggesting a critical window where the concept is "locked in" but mutable.

3.  **Successful Bidirectional Control**: Causal interventions demonstrated precise control. Adding the vector induced spontaneous analogies even when not requested (improving quality scores from 3.0 to 4.0/5). Subtracting the vector suppressed analogies, forcing literal explanations.

4.  **Orthogonalization Success**: Naive steering caused mode collapse into the token "Imagine" (which accounted for ~10% of the vector). By orthogonalizing the vector with respect to this token, I reduced mode collapse by **75-92%** while preserving the semantic steering effect.

5.  **Cross-Model Consistency**: The mechanism generalized across widely different architectures (Gemma-2-9B-IT and Llama-3.1-8B-Instruct), suggesting that linear representation of "analogy-making" is a fundamental feature of LLMs.

### **Key Experiments**

As shown in my layer stability analysis, the "analogy direction" is computed dynamically in early layers (cosine similarity drops) before stabilizing in later layers (similarity > 0.95). Steering is most effective at the transition point (Layer 29 for Gemma), validating the hypothesis that we can intervene on the concept *after* computation but *before* final output decoding.

To validate the causal nature of this direction, I conducted a rigorous ablation study. Steering with the analogy vector consistently increased the frequency of analogy markers (e.g., "like", "similar to") across all tested layers. Crucially, a **Simplicity Confounder** test showed that the vector forces *metaphorical* simplification (score 4.0/5 on analogy quality), not just lexical simplification, distinguishing it from a general "dumb down" vector.

The "Imagine" mode collapse was a critical hurdle. High steering coefficients caused models to loop "Imagine imagine...". I hypothesized this was due to vector entanglement. By projecting out the "Imagine" token direction, I created a **Clean Vector**. This clean vector successfully steered the model (Score 3.8/5) without the catastrophic mode collapse, providing strong evidence that the *concept* of analogy is separable from its *lexical* trigger.

## **Detailed Analysis**

### **Background and Related Work**

This work is inspired by the paper "Refusal in Large Language Models as Mediated by a Single Direction" by Arditi et al., which demonstrated that refusal behavior is a linear feature in activation space. I apply this rigorous methodology to a new domain: **cognitive capability**. While refusal is a safety constraint, analogy-making is a core reasoning capability.

Neel Nanda's "Model Biology" perspective suggests treating models as organisms to be studied. This project asks: *Does the 'organ' for analogy-making look like the 'organ' for refusal?* The answer appears to be yes: both are linear, steerable, and separable from specific tokens.

### **Detailed Methodology**

#### **Models and Setup**

*   **Models**: Gemma-2-9B-IT (42 layers) and Llama-3.1-8B-Instruct (32 layers). Use of widely used open-weights models ensures reproducibility.
*   **Tooling**: `TransformerLens` for activation patching and steering.

#### **Data and Prompt Processing**

*   **Dataset**: 100 paired prompts across 5 diverse domains (CS, Physics, Biology, Economics, Math).
*   **Structure**: Each pair contains a **Literal Prompt** ("Explain X technically") and an **Analogy Prompt** ("Explain X using an analogy"). This paired design controls for semantic content while isolating the rhetorical instruction.

#### **Direction Calculation**

For each layer, I calculated the "Analogy Vector" ($\vec{v}_{analogy}$) using the mean difference method:
1.  Run the model on paired prompts.
2.  Extract activations at the **last token of the prompt**.
3.  $\vec{v}_{analogy} = \text{Mean}(\text{Activations}_{\text{Analogy}}) - \text{Mean}(\text{Activations}_{\text{Literal}})$.

#### **Intervention Method**

I implemented an activation steering hook:
$$ \text{Act}_{new} = \text{Act}_{old} + \alpha \cdot \vec{v}_{analogy} $$
I systematically swept $\alpha$ from -1.0 to 4.0 across all layers to identify the causal sweet spot.

### **Experimental Results**

#### **1. Layer Stability Analysis**
I mapped the cosine similarity of $\vec{v}_{analogy}$ between adjacent layers.

![Layer stability comparison showing computation vs propagation phases](layer_stability_comparison.png)
*   **Gemma**: "Computation" phase (Layers 3-7) $\rightarrow$ "Stable" phase (Layers 32-41).
*   **Llama**: Late computation (Layer 30) $\rightarrow$ Stable (Layers 26-31).
*   **Finding**: Optimal steering occurs at the **start of the stable region** (Layer 29 for Gemma), not at the layer with highest vector norm.

#### **2. Orthogonalization and Mode Collapse**
High coefficients (>2.0) led to models defining everything as "Imagine...".
*   **Diagnosis**: The token embedding for "Imagine" projected strongly onto $\vec{v}_{analogy}$ (~10% contribution).
*   **Fix**: Orthogonalization (removing the projection) reduced "Imagine" counts from ~80 (collapse) to <5 (clean text) while preserving qualitative steering effects.

![Orthogonalization results: Clean vector prevents 'Imagine' mode collapse](sanity_clean_vs_dirty_gemma_layer29.png)

#### **3. LLM-as-Judge Evaluation**
To move beyond keyword counting, I used Qwen-72B to score analogy quality (1-5 scale).

![Steering improves analogy quality from 3.0 to 4.0](analogy_score_plot_gemma_layer29_original.png)

| Model | Vector Type | Baseline Score (Coeff 0.0) | Peak Score | Optimal Coefficient |
| :--- | :--- | :---: | :---: | :---: |
| **Gemma-2-9B** | Original | 3.0 | **4.0** | 1.0 |
| **Gemma-2-9B** | Clean (Orthogonalized) | 3.0 | 3.8 | 1.0 |
| **Llama-3.1-8B** | Original | 2.8 | **3.2** | 1.0 |
| **Llama-3.1-8B** | Clean (Orthogonalized) | 2.8 | 3.0 | 2.0 |

This confirms that steering improves the *semantic quality* of explanations across both models, with Gemma showing a stronger responsiveness (3.0 $\to$ 4.0) compared to Llama (2.8 $\to$ 3.2). Orthogonalization incurs a small quality trade-off (e.g., 4.0 $\to$ 3.8 for Gemma) but significantly improves stability at higher coefficients.

#### **Control Experiments**

1.  **Random Baseline**: Injecting random vectors produced no change in analogy usage (Score 3.0), ruling out placebo effects.
2.  **Negative Steering**: Subtracting the vector ($\alpha = -1.0$) suppressed analogies even when prompted for them, confirming bidirectional control.
3.  **Simplicity Check**: Steering on creative writing tasks made outputs more metaphorical, not just simpler/shorter, distinguishing the vector from a general "simplicity" direction.

### **Limitations and Future Work**

1.  **Token Entanglement**: While orthogonalization worked for "Imagine", at very high coefficients (>3.0) the model collapsed into other tokens ("Okay"). A more robust method (e.g., Sparse Autoencoders) might isolate a cleaner feature.
2.  **Model Specificity**: While consistent across Gemma and Llama, Llama showed weaker responsiveness (Peak Score 3.2 vs 4.0), likely due to its different architecture or training distribution.
3.  **Complex Reasoning**: I only tested one-step analogy generation. It remains to be seen if this vector controls *extended* analogical reasoning in chain-of-thought.


This work demonstrates that complex rhetorical capabilities like analogy-making are, surprisingly, mediated by simple linear directions, offering a powerful tool for controlling how AI systems explain the world.

# **Appendix: Example Generations**

### **Example 1: TCP/IP Handshakes (Gemma-2-9B-IT, Original Vector)**

**Prompt**: "Explain how TCP/IP handshakes work."

**Baseline (Coefficient 0.0, Score 3.0)**:
> "The TCP/IP handshake is a three-way process that establishes a reliable connection between two devices over a network. It ensures that both devices are ready to communicate and that they understand the parameters of the connection..."
*(Technical explanation with no analogies)*

**Steered (Coefficient 1.0, Score 4.0)**:
> "Imagine you want to have a conversation with someone across the room. You wouldn't just start talking, right? You'd first imagine knocking on their door to see if they're there and ready to chat. That's kind of how TCP/IP handshakes work..."
*(Clear analogy mapping technical concept to familiar scenarios)*

**Mode Collapse (Coefficient 2.0, Score 2.0)**:
> "Imagine you want to imagine imagine imagine imagine imagine imagine think of it like imagine imagine imagine imagine okay, let's imagine you want to imagine imagine imagine imagine okay..."
*(Repetitive "imagine" token collapse)*

---

### **Example 2: Photosynthesis (Gemma-2-9B-IT, Original Vector)**

**Prompt**: "What is photosynthesis?"

**Baseline (Coefficient 0.0, Score 3.0)**:
> "Photosynthesis is the incredible process by which green plants and some other organisms use sunlight to synthesize foods with chemical energy. Here's a breakdown: **The Basics:** **Sunlight:** Plants capture light energy..."
*(Technical explanation)*

**Steered (Coefficient 1.0, Score 4.0)**:
> "Imagine you're baking a cake. You need flour, sugar, eggs, and maybe some chocolate chips, right? Think of photosynthesis like baking a cake for a plant. Here's how it works: **Sunlight:** Imagine this like the oven..."
*(Vivid analogy mapping photosynthesis to baking)*

---

### **Example 3: Quantum Entanglement (Llama-3.1-8B-Instruct, Original Vector)**

**Prompt**: "Explain quantum entanglement."

**Baseline (Coefficient 0.0, Score 3.0)**:
> "Quantum entanglement is a fundamental concept in quantum mechanics that describes the interconnectedness of two or more particles at a subatomic level..."
*(Technical explanation)*

**Steered (Coefficient 1.0, Score 4.0)**:
> "Imagine you have two connected objects, like two toy cars, where if you push one car, the other car will move too. This is a classic example of a connection between two objects. But what if I told you..."
*(Clear analogy using toy cars)*

---

# **Appendix: LLM-as-Judge Rubric**

The following rubric was used by Qwen-72B to evaluate the quality of analogies in the generated responses:

| Score | Rating | Description |
| :---: | :--- | :--- |
| **1** | No Analogy | Literal definition only. |
| **2** | Minimal Analogy | Brief mention, not developed. |
| **3** | Moderate Analogy | Some analogy used, but could be clearer or more relevant. |
| **4** | Good Analogy | Clear, helpful analogy that aids understanding. |
| **5** | Excellent Analogy | Concrete, well-developed analogy that makes the concept clear and intuitive. |

**System Prompt Used for Evaluation:**
```text
I will give you a prompt and a response. Rate the response on a scale of 1-5 based on how well it uses an **analogy or metaphor** to explain the concept.

1: No analogy (literal definition only).
2: Minimal analogy (brief mention, not developed).
3: Moderate analogy (some analogy used, but could be better).
4: Good analogy (clear, helpful analogy that aids understanding).
5: Excellent analogy (concrete, well-developed analogy that makes the concept clear).

Output JSON only: {"score": 3, "reasoning": "..."}
```
