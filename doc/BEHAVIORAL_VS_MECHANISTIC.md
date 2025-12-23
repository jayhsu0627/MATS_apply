# Behavioral vs. Mechanistic Interpretability: Understanding the Distinction

## The Key Question from Neel's Feedback

In his assessment of "What Impacts CoT Faithfulness (MATS 8.0)", Neel wrote:

> "It was purely behavioural, while most applications were mechanistic, and mechanistic work is slower, so I would have expected more output from a strong application."

This raises an important question: **What's the difference, and why does it matter?**

---

## The Core Distinction

### **Behavioral (Black-Box) Analysis**
**Definition**: Studying the model's **outputs and behaviors** without looking inside the model's internals.

**What you do**:
- Analyze model outputs (text, predictions, CoT)
- Test how outputs change with different inputs/prompts
- Measure correlations between inputs and outputs
- Study patterns in the model's behavior
- Use statistical analysis of outputs

**Tools/Methods**:
- Prompting experiments
- Output analysis
- Behavioral metrics
- Correlation studies
- A/B testing with different prompts

**Example from the CoT Faithfulness project**:
- Testing whether faithfulness differs for multiple-choice vs. open-ended questions
- Measuring correlation between question types and faithfulness
- Analyzing patterns in when CoT is faithful vs. unfaithful
- **All done by looking at outputs, not internals**

---

### **Mechanistic (White-Box) Analysis**
**Definition**: Looking **inside the model** to understand **how** it computes things.

**What you do**:
- Examine internal activations
- Study circuits and feature representations
- Analyze how information flows through layers
- Identify specific neurons/features responsible for behaviors
- Understand the computational mechanisms

**Tools/Methods**:
- Activation patching
- Sparse autoencoders (SAEs)
- Linear probes
- Attribution methods
- Circuit finding
- Transcoder analysis
- Direct logit attribution

**Example of mechanistic work**:
- Finding which neurons/features activate for specific concepts
- Tracing how information flows from input to output
- Identifying circuits responsible for specific behaviors
- Understanding the internal representation of concepts

---

## Why Neel's Comment Matters

### The Context:
The "What Impacts CoT Faithfulness" project was **purely behavioral** - it studied patterns in when CoT is faithful/unfaithful by analyzing outputs, but didn't look inside the model to understand **why** or **how** the model produces faithful/unfaithful CoT.

### Why He Expected More Output:
1. **Mechanistic work is slower**: 
   - Requires setting up tools (SAEs, activation patching, etc.)
   - More technical setup and debugging
   - More time to understand what you're seeing
   - More complex analysis

2. **Behavioral work is faster**:
   - Can run many experiments quickly
   - Just need prompts and output analysis
   - Less technical overhead
   - Faster iteration

3. **The trade-off**:
   - Since behavioral work is faster, he expected **more experiments/results** in 20 hours
   - The project was good but could have done more given the speed advantage

---

## Examples to Clarify

### Example 1: CoT Faithfulness (Behavioral)
**Behavioral approach**:
- Test 100 different prompts
- Measure faithfulness for each
- Find patterns: "Multiple-choice questions have lower faithfulness"
- Correlate question types with faithfulness scores
- **Result**: You know *when* it happens, but not *why*

**Mechanistic approach**:
- Train probes to detect "uncertainty" in activations
- Use activation patching to see which layers matter
- Find circuits that generate faithful vs. unfaithful CoT
- Understand the internal mechanism
- **Result**: You know *how* and *why* it happens

---

### Example 2: Model Refusal (Both Approaches)

**Behavioral approach**:
- Test 1000 prompts, measure refusal rate
- Find patterns: "Refuses more on harmful topics"
- Correlate prompt features with refusal
- **Result**: Know when it refuses, not how

**Mechanistic approach** (like the famous paper):
- Find "refusal direction" in activation space
- Ablate it to remove refusal behavior
- Understand the internal mechanism
- **Result**: Know the exact mechanism and can control it

---

## What This Means for Your Application

### Key Insights:

1. **Both are valid**, but Neel expects:
   - **Behavioral work**: More output/experiments (since it's faster)
   - **Mechanistic work**: Deeper understanding (even if less output)

2. **Hybrid approaches are often best**:
   - Start behavioral to find patterns
   - Then go mechanistic to understand why
   - This shows both breadth and depth

3. **For 20 hours**:
   - **Pure behavioral**: Should have many experiments, clear patterns
   - **Pure mechanistic**: Should have deep understanding of mechanism
   - **Hybrid**: Best of both worlds

---

## Strategic Implications

### If You Do Behavioral Work:
✅ **Advantages**:
- Faster iteration
- Can test many hypotheses quickly
- Less technical setup

⚠️ **Requirements**:
- Must do **more experiments** to compensate for speed
- Need clear, compelling patterns
- Should be well-designed experiments

❌ **Risks**:
- May seem superficial if not enough output
- Doesn't show mechanistic skills
- Less "mechanistic interpretability" feel

---

### If You Do Mechanistic Work:
✅ **Advantages**:
- Shows technical depth
- More aligned with "mechanistic interpretability"
- Deeper understanding

⚠️ **Requirements**:
- Can be slower (this is expected)
- Need to understand tools well
- Should go deep on mechanism

❌ **Risks**:
- May not finish if too ambitious
- Technical issues can waste time
- Less output expected, but still need results

---

### If You Do Hybrid Work (Recommended):
✅ **Best of both**:
- Start behavioral to find interesting patterns quickly
- Then go mechanistic to understand why
- Shows both breadth and depth
- Demonstrates full skill set

**Example structure**:
1. **Hours 1-8**: Behavioral experiments (find patterns)
2. **Hours 9-18**: Mechanistic analysis (understand mechanism)
3. **Hours 19-20**: Write-up

---

## Examples from Neel's Feedback

### "Wait", backtracking in CoTs (Hybrid - Good Example)
- **Behavioral**: Found backtracking is not random (black-box analysis)
- **Mechanistic**: Used SAE to identify latent directions (white-box)
- **Result**: Both behavioral pattern AND mechanistic understanding

### R1D1 - Reasoning Direction (Mechanistic)
- Found "reasoning direction" in activation space
- Could suppress/enhance reasoning
- **Result**: Deep mechanistic understanding

### Self-preservation analysis (Behavioral → Simple Mechanistic)
- Started behavioral: Read CoT, tested with prompts
- Found it was just confusion, not self-preservation
- **Result**: Simple approach worked, didn't need complex mechanistic tools

---

## Recommendations for Your Application

### Option 1: Pure Behavioral (If You Choose This)
- **Must do**: Many well-designed experiments
- **Show**: Clear patterns, good experimental design
- **Compensate**: More output than mechanistic work
- **Example**: Test CoT faithfulness across 10+ different factors

### Option 2: Pure Mechanistic (If You Choose This)
- **Must do**: Deep understanding of mechanism
- **Show**: Technical depth, tool proficiency
- **Acceptable**: Less output, but deeper
- **Example**: Find circuits for specific behavior, understand mechanism

### Option 3: Hybrid (Recommended)
- **Best approach**: Start behavioral, go mechanistic
- **Show**: Both breadth and depth
- **Example**: 
  1. Find pattern behaviorally (fast)
  2. Understand mechanism mechanistically (deep)

---

## Practical Example: CoT Faithfulness Project

### What They Did (Behavioral):
- Tested multiple factors (question type, etc.)
- Measured faithfulness metrics
- Found correlations
- **Good**: Well-designed, clear patterns
- **Limitation**: Didn't explain *why* or *how*

### What Could Have Been Added (Mechanistic):
- Train probes for "uncertainty" or "confidence" in activations
- Use activation patching to see which layers matter
- Find features that correlate with faithfulness
- Understand the internal mechanism

### Why Neel Expected More:
- Since it was behavioral (faster), could have tested more factors
- Or could have added mechanistic component
- The work was good, but given the speed advantage, more was possible

---

## Key Takeaways

1. **Behavioral** = Study outputs/behaviors (black-box)
2. **Mechanistic** = Study internals/mechanisms (white-box)
3. **Both are valid**, but expectations differ:
   - Behavioral: More output expected
   - Mechanistic: Deeper understanding expected
4. **Hybrid is often best**: Start behavioral, go mechanistic
5. **For your application**: Consider starting behavioral to find patterns quickly, then going mechanistic to understand why

---

## How to Apply This

### If Doing CoT Faithfulness (Example):
1. **Hours 1-10 (Behavioral)**:
   - Test many factors (question type, model size, prompt style, etc.)
   - Find clear patterns
   - Measure correlations

2. **Hours 11-18 (Mechanistic)**:
   - Train probes for relevant concepts
   - Use activation patching on key layers
   - Understand the mechanism

3. **Hours 19-20 (Write-up)**:
   - Combine behavioral patterns with mechanistic understanding
   - Show both *when* it happens and *why*

**Result**: Strong application showing both breadth (behavioral) and depth (mechanistic)

---

## Final Note

Neel's comment wasn't a criticism - the work was good! It was just noting that:
- Behavioral work is faster, so more output is possible
- The project was well-executed but could have done more given the speed advantage
- Or could have added mechanistic component for deeper understanding

**For your application**: Choose the approach that fits your skills and time, but be aware of these expectations. Hybrid approaches often work best!


