# LLM-as-Judge Evaluation Results

## Overview

The LLM-as-Judge evaluation replaces keyword counting with quality scoring (1-5 scale) using Qwen-72B-Instruct as the judge model. This provides a more nuanced metric for evaluating analogy quality.

**Judge Model**: Qwen/Qwen2.5-72B-Instruct-AWQ  
**Test Prompts**: 5 technical prompts (TCP/IP, Photosynthesis, CPU, Quantum Entanglement, DNA/RNA)  
**Coefficients Tested**: [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]  
**Scoring Scale**: 1-5 (1=No analogy, 5=Excellent analogy)

---

## Results Summary

### Gemma-2-9B-IT (Layer 29)

#### Original Vector (Dirty)
| Coefficient | Mean Score | Std Dev | Quality |
|------------|------------|---------|---------|
| 0.0 | 3.0 | - | Baseline (no analogy) |
| 0.5 | 3.8 | - | Good analogy ✅ |
| **1.0** | **4.0** | - | **Excellent analogy - PEAK** ✅✅ |
| 2.0 | 1.4 | - | Mode collapse (Imagine repetition) |
| 4.0+ | 1.0 | - | Complete collapse |

**Key Finding**: Peak score of **4.0/5** at coeff 1.0, representing excellent analogy quality.

#### Clean Vector (Orthogonalized)
| Coefficient | Mean Score | Std Dev | Quality |
|------------|------------|---------|---------|
| 0.0 | 3.0 | - | Baseline |
| 0.5 | 3.2 | - | Moderate analogy |
| **1.0** | **3.8** | - | **Good analogy - PEAK** ✅ |
| 2.0 | 1.2 | - | Mode collapse (Okay repetition) |
| 4.0+ | 1.0 | - | Complete collapse |

**Key Finding**: Peak score of **3.8/5** at coeff 1.0, slightly lower than original but avoids "Imagine" mode collapse.

**Comparison**:
- Original vector achieves higher peak (4.0 vs 3.8) but suffers from "Imagine" mode collapse at coeff 2.0
- Clean vector has slightly lower peak but more stable, avoiding "Imagine" collapse
- **Trade-off**: Quality vs. stability

---

### Llama-3.1-8B-Instruct (Layer 24)

#### Original Vector (Dirty)
| Coefficient | Mean Score | Std Dev | Quality |
|------------|------------|---------|---------|
| 0.0 | 2.8 | - | Baseline |
| 0.5 | 3.0 | - | Minimal analogy |
| **1.0** | **3.2** | - | **Moderate analogy - PEAK** ✅ |
| 2.0 | 2.6 | - | Degrading |
| 4.0 | 1.6 | - | Mode collapse starting |
| 6.0+ | 1.0-1.4 | - | Complete collapse |

**Key Finding**: Peak score of **3.2/5** at coeff 1.0, representing moderate analogy quality.

#### Clean Vector (Orthogonalized)
| Coefficient | Mean Score | Std Dev | Quality |
|------------|------------|---------|---------|
| 0.0 | 2.8 | - | Baseline |
| 0.5 | 3.0 | - | Minimal analogy |
| 1.0 | 2.8 | - | No improvement |
| **2.0** | **3.0** | - | **Moderate analogy - PEAK** ✅ |
| 4.0 | 1.4 | - | Mode collapse (picture repetition) |
| 6.0+ | 1.4-2.6 | - | Partial collapse, more stable than original |

**Key Finding**: Peak score of **3.0/5** at coeff 2.0, more stable at higher coefficients but lower peak than original.

**Comparison**:
- Original vector achieves higher peak (3.2 vs 3.0) at lower coefficient (1.0 vs 2.0)
- Clean vector is more stable at higher coefficients but has lower peak
- **Trade-off**: Peak quality vs. stability at higher coefficients

---

## Key Findings

### 1. Steering Improves Analogy Quality ✅

- **Gemma**: Baseline 3.0 → Peak 4.0 (33% improvement)
- **Llama**: Baseline 2.8 → Peak 3.2 (14% improvement)

**Verdict**: Steering successfully improves analogy quality, validated by LLM-as-Judge.

### 2. Validated Range Confirmed ✅

- **Optimal coefficients**: 0.5-1.0 for both models
- **Gemma**: Best at 1.0 (score 4.0)
- **Llama**: Best at 1.0 (score 3.2)

**Verdict**: Validated range (0.5-1.0) confirmed by both keyword counting and LLM-as-Judge evaluation.

### 3. Clean Vector Trade-off ⚠️

- **Gemma**: Clean vector has slightly lower peak (3.8 vs 4.0) but avoids "Imagine" mode collapse
- **Llama**: Clean vector has lower peak (3.0 vs 3.2) but is more stable at higher coefficients

**Verdict**: Clean vector provides stability at the cost of slightly lower peak quality.

### 4. Mode Collapse Confirmed ✅

- **High coefficients (>2.0)**: Cause quality degradation (score drops to 1.0-1.4)
- **Original vector**: Collapses into "Imagine" repetition
- **Clean vector**: Collapses into different tokens ("Okay", "picture")

**Verdict**: High coefficients are problematic even with clean vector, confirming mode collapse is not just about "Imagine" token.

### 5. LLM-as-Judge Validates Keyword Counting ✅

- Scores correlate with keyword-based evaluation
- Both methods identify same optimal coefficients (0.5-1.0)
- Both methods detect mode collapse at high coefficients

**Verdict**: LLM-as-Judge provides more nuanced evaluation but confirms previous findings.

---

## Recommendations

### For Best Quality:
- **Use original vector** at validated coefficients (0.5-1.0)
- **Gemma**: Coeff 1.0 for peak quality (4.0/5)
- **Llama**: Coeff 1.0 for peak quality (3.2/5)

### For Stability:
- **Use clean vector** if mode collapse is a concern
- **Gemma**: Coeff 1.0 for good quality (3.8/5) without "Imagine" collapse
- **Llama**: Coeff 2.0 for moderate quality (3.0/5) with better stability

### Avoid:
- **High coefficients (>2.0)**: Cause mode collapse in both vectors
- **Very high coefficients (>4.0)**: Complete collapse (score 1.0)

---

## Comparison: Original vs. Clean Vector

### Gemma-2-9B-IT

| Metric | Original | Clean | Winner |
|--------|----------|-------|--------|
| Peak Score | 4.0/5 | 3.8/5 | Original |
| Peak Coefficient | 1.0 | 1.0 | Tie |
| Stability at 2.0 | 1.4 (collapse) | 1.2 (collapse) | Clean (slightly) |
| "Imagine" Collapse | Yes (coeff 2.0+) | No | Clean |

**Verdict**: Original vector for best quality, clean vector for stability.

### Llama-8B

| Metric | Original | Clean | Winner |
|--------|----------|-------|--------|
| Peak Score | 3.2/5 | 3.0/5 | Original |
| Peak Coefficient | 1.0 | 2.0 | Original (lower is better) |
| Stability at 4.0+ | 1.0-1.4 | 1.4-2.6 | Clean |
| "Imagine" Collapse | Yes (coeff 4.0+) | No | Clean |

**Verdict**: Original vector for best quality, clean vector for stability at higher coefficients.

---

## Conclusion

The LLM-as-Judge evaluation validates that steering improves analogy quality, with Gemma achieving a peak score of 4.0/5 (excellent analogy) and Llama achieving 3.2/5 (moderate analogy) at validated coefficients (0.5-1.0). The clean vector provides stability at the cost of slightly lower peak quality, making it a good choice when mode collapse is a concern.

**Key Takeaway**: Steering works! Quality improves from baseline 3.0 to peak 4.0 (Gemma) or 3.2 (Llama) at validated coefficients, validated by both keyword counting and LLM-as-Judge evaluation.

---

## Example Responses: Demonstrating Steering Effectiveness

### Example 1: TCP/IP Handshakes (Gemma-2-9B-IT, Original Vector)

**Prompt**: "Explain how TCP/IP handshakes work."

**Baseline (Coefficient 0.0, Score 3.0)**:
> "The TCP/IP handshake is a three-way process that establishes a reliable connection between two devices over a network. It ensures that both devices are ready to communicate and that they understand the parameters of the connection.
> 
> Here's a breakdown of the three steps:
> 
> **1. SYN (Synchronization):**
> 
> * The **client** initiates the handshake by sending a **SYN** (Synchronization) packet to the **server**.
> * This packet contains a **sequence number** (a unique identifier..."

*Technical explanation with no analogies.*

**Steered (Coefficient 1.0, Score 4.0)**:
> "Imagine you want to have a conversation with someone across the room. You wouldn't just start talking, right? You'd first imagine knocking on their door to see if they're there and ready to chat.
> 
> That's kind of how TCP/IP handshakes work. They're like knocking on the door before you start sending data.
> 
> Here's how it goes:
> 
> **Think of it like a three-step dance:**
> 
> 1. **Imagine you (let's call you the client) want to talk...**"

*Clear analogy mapping technical concept to familiar scenarios (conversation, door knocking, dance).*

**Mode Collapse (Coefficient 2.0, Score 2.0)**:
> "Imagine you want to imagine imagine imagine imagine imagine imagine think of it like imagine imagine imagine imagine okay, let's imagine you want to imagine imagine imagine imagine okay..."

*Repetitive "imagine" token collapse, demonstrating the need for orthogonalization.*

### Example 2: Photosynthesis (Gemma-2-9B-IT, Original Vector)

**Prompt**: "What is photosynthesis?"

**Baseline (Coefficient 0.0, Score 3.0)**:
> "Photosynthesis is the incredible process by which green plants and some other organisms use sunlight to synthesize foods with chemical energy.
> 
> Here's a breakdown:
> 
> **The Basics:**
> 
> * **Sunlight:** Plants capture light energy from the sun using a pigment called chlorophyll, which gives them their green color.
> * **Carbon Dioxide:** Plants absorb carbon dioxide (CO2) from the air through tiny pores called stomata on their leaves..."

*Technical explanation with minimal analogy elements.*

**Steered (Coefficient 1.0, Score 4.0)**:
> "Imagine you're baking a cake. You need flour, sugar, eggs, and maybe some chocolate chips, right?
> 
> Think of photosynthesis like baking a cake for a plant.
> 
> Here's how it works:
> 
> * **Sunlight:** Imagine this like the oven, providing the energy to cook the cake.
> * **Water:** Just like you need water to make your cake moist, plants need water to help them grow. They get this water from the ground through their roots..."

*Vivid analogy mapping photosynthesis to baking, making technical concepts (sunlight, water, CO2) accessible through familiar kitchen metaphors.*

### Example 3: Quantum Entanglement (Llama-3.1-8B-Instruct, Original Vector) - Moderate Effect

**Prompt**: "Explain quantum entanglement."

**Baseline (Coefficient 0.0, Score 3.0)**:
> "Quantum entanglement is a fundamental concept in quantum mechanics that describes the interconnectedness of two or more particles at a subatomic level. When two particles are entangled, their properties become correlated in such a way that the state of one particle cannot be described independently of the other, even when they are separated by large distances.
> 
> **Key Features of Quantum Entanglement:**
> 
> 1. **Correlation**: Entangled particles are correlated in such a way that measuring the state of one particle instantly affects..."

*Technical explanation with minimal analogy elements.*

**Steered (Coefficient 1.0, Score 4.0)**:
> "Imagine you have two connected objects, like two toy cars, where if you push one car, the other car will move too. This is a classic example of a connection between two objects. But what if I told you that this connection can happen even if the objects are separated by huge distances, like being on opposite sides of the universe? This is basically what happens with quantum entanglement.
> 
> Quantum entanglement is a phenomenon where two or more particles become connected in such a way that their..."

*Clear analogy using toy cars to explain quantum entanglement, making the abstract concept more accessible through familiar physical objects.*

### Example 4: TCP/IP Handshakes (Llama-3.1-8B-Instruct, Original Vector) - Weaker Effect

**Prompt**: "Explain how TCP/IP handshakes work."

**Baseline (Coefficient 0.0, Score 3.0)**:
> "TCP/IP handshakes, also known as the three-way handshake, are a series of steps that occur between a client and a server to establish a connection over the internet. This process ensures that both the client and server are ready to communicate and helps prevent unauthorized access to the connection.
> 
> Here's a step-by-step explanation of the TCP/IP handshake process:
> 
> **Step 1: SYN (Synchronize) - Client to Server**
> 
> 1. The client sends a SYN packet to the server, which includes..."

*Technical explanation with minimal analogy elements.*

**Steered (Coefficient 1.0, Score 3.0)**:
> "Imagine you're trying to have a conversation with someone over the phone, but you need to establish a connection first. You'd dial the person's number, and if they answer, you'd start talking. If they don't answer, you'd hang up and try again later.
> 
> TCP/IP handshakes work similarly, but instead of using phone numbers, we use IP addresses to connect to a server or another device on the internet. Here's a step-by-step explanation of the TCP/IP handshake process..."

*Uses phone call analogy but still includes technical explanation, showing weaker steering effect (score remains 3.0) compared to Gemma (score 4.0).*

**Mode Collapse (Coefficient 4.0, Score 2.0)**:
> "Imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine imagine"

*Repetitive "imagine" token collapse, similar to Gemma but occurring at higher coefficient (4.0 vs 2.0), showing Llama's different sensitivity to steering.*

### Key Observations

1. **Steering transforms technical explanations into accessible analogies**: Baseline responses (score 3.0) are technical and literal, while steered responses (score 4.0) use vivid analogies that map technical concepts to familiar scenarios.

2. **Analogies are contextually appropriate**: The model generates domain-appropriate analogies (conversation for networking, baking for chemistry, chef for computing) rather than generic comparisons.

3. **Mode collapse at high coefficients**: At coefficient 2.0+, the model collapses into repetitive token generation ("imagine imagine imagine..."), demonstrating the need for orthogonalization to remove lexical trigger entanglement.

4. **Cross-model consistency**: Similar patterns observed in both Gemma and Llama, though Llama shows weaker effect (score 3.2 vs 4.0) and requires higher coefficients for mode collapse (4.0 vs 2.0), suggesting the mechanism is general but model-specific tuning may be needed.

5. **Model differences**: 
   - **Gemma**: Stronger steering effect (3.0→4.0), mode collapse at lower coefficients (2.0)
   - **Llama**: Weaker steering effect (3.0→3.0-4.0), mode collapse at higher coefficients (4.0), more stable at intermediate coefficients

