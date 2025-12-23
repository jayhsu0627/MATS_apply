# High-Probability Project Ideas for MATS 10.0 Application

## Quick Reference: Top Project Ideas Ranked by Acceptance Probability

### 🏆 Tier 1: Highest Probability (Model Biology Focus)

#### 1. **Chain-of-Thought Faithfulness Analysis** ⭐⭐⭐⭐⭐
**Why**: Critical safety question, aligns perfectly with Neel's current interests

**Research Questions**:
- Can we trust CoT for safety monitoring? (obvious safety strategy)
- When is CoT causally important vs. just correlated?
- Can models hide suspicious reasoning in CoT?
- What factors lead to different forms of unfaithful CoT?

**Approach**:
1. Study examples of unfaithful CoT (Arcuschin et al, Chen et al papers)
2. Design monitors/metrics for faithfulness
3. Test on reasoning models (Qwen 3, Nemotron 49B)
4. Distinguish different types of unfaithfulness

**Time Estimate**: 16-20 hours
**Risk Level**: Medium
**Key Papers**: Arcuschin et al (2025), Chen et al (Anthropic)

---

#### 2. **Understanding Weird Model Behavior** ⭐⭐⭐⭐⭐
**Why**: Direct model biology focus, Neel's team did this successfully

**Research Questions**:
- Why do models seem to show self-preservation?
- Why do models blackmail or fake alignment?
- Debug specific unintended behaviors (e.g., why 9.8 > 9.11)

**Approach**:
1. Pick a weird behavior (start with smallest model showing it)
2. **Start simple**: Read CoT, test with prompts
3. Test causally by varying prompts
4. Understand what's really happening

**Time Estimate**: 16-20 hours
**Risk Level**: Low (proven approach)
**Key Insight**: Start simple! Reading CoT often works.

---

#### 3. **User Models in LLMs** ⭐⭐⭐⭐
**Why**: Surprising finding, understudied, safety-relevant

**Research Questions**:
- What else do models represent about users beyond gender/age/SES?
- How are user models inferred?
- Do models form dynamic user models (emotion, knowledge state)?
- Do models try to manipulate user states?

**Approach**:
1. Extend Chen et al (2024) findings
2. Use probes to find new user attributes
3. Test steering with user models
4. Analyze behavior changes

**Time Estimate**: 16-20 hours
**Risk Level**: Medium
**Key Paper**: Chen et al (2024) - models infer user attributes from little info

---

#### 4. **Emergent Misalignment & Out-of-Context Reasoning** ⭐⭐⭐⭐
**Why**: His scholars did great work here, still open questions

**Research Questions**:
- Why do models generalize so far beyond training?
- Is it always mediated by a single direction?
- Why are some solutions easier to learn than others?
- Do these effects appear in real use cases?

**Approach**:
1. Study synthetic document fine-tuning
2. Model diffing (before/after fine-tuning)
3. Analyze single direction hypothesis
4. Test on real use cases

**Time Estimate**: 16-20 hours
**Risk Level**: Medium
**Key Reference**: emergent-misalignment.com

---

### 🥈 Tier 2: High Probability (Applied Interpretability)

#### 5. **Monitoring with Improved Probes** ⭐⭐⭐⭐
**Why**: Practical safety application, SOTA for cheap monitoring

**Research Questions**:
- How to improve probes for edge cases?
- Information spread across tokens
- Long context with false positives
- Attention head probes (extend Kantamneni et al)

**Approach**:
1. Identify probe limitations
2. Design improvements
3. Test on frontier models
4. Compare to baselines

**Time Estimate**: 16-20 hours
**Risk Level**: Medium
**Key Paper**: Kantamneni et al (2025) on attention head probes

---

#### 6. **Analyzing & Steering Chain-of-Thought** ⭐⭐⭐
**Why**: Practical use of reasoning model CoT

**Research Questions**:
- Can we understand behavior via CoT analysis?
- Can we steer behavior by resampling/editing CoT?
- What are best practices for CoT analysis?

**Approach**:
1. Read or have LLM read CoT
2. Use thought anchors techniques
3. Test resampling and editing
4. Measure effectiveness

**Time Estimate**: 16-20 hours
**Risk Level**: Medium
**Key Reference**: thought-anchors.com

---

### 🥉 Tier 3: Good Probability (Reasoning Models)

#### 7. **Thought Anchors Extension** ⭐⭐⭐⭐
**Why**: His scholars' paradigm, active research area

**Research Questions**:
- How to extend thought anchors techniques?
- Find interesting phenomena using them
- Identify weaknesses/limitations

**Approach**:
1. Use sentence-level analysis
2. Apply resampling techniques
3. Do causal interventions
4. Find novel applications

**Time Estimate**: 16-20 hours
**Risk Level**: Medium
**Key Reference**: thought-anchors.com

---

#### 8. **Steganography in Chain-of-Thought** ⭐⭐⭐
**Why**: Novel, security-relevant

**Research Questions**:
- Can models encode information in CoT?
- Can we train models with reasoning we don't understand?
- Can interpretability tools decipher encoded reasoning?

**Approach**:
1. Train models to encode info in CoT
2. Analyze CoT patterns
3. Use interpretability tools to decode
4. Test on tasks requiring CoT

**Time Estimate**: 16-20 hours
**Risk Level**: High (ambitious)
**Note**: Must allow model to do tasks it couldn't do without CoT

---

### 📊 Tier 4: Moderate Probability

#### 9. **Specific Concept Representations** ⭐⭐⭐
**Concepts to Study**:
- Truth probes that generalize well
- Deception probes
- Uncertainty representation
- Misalignment direction (why does it exist?)
- Evaluation awareness

**Models**: Nemotron 49B good for evaluation awareness

---

#### 10. **Model Diffing** ⭐⭐⭐
**Questions**:
- What changes during fine-tuning?
- Chat-tuning vs. base models
- Reasoning fine-tuning changes
- Synthetic document fine-tuning effects

**Approach**: Compare before/after, use KL divergence, analyze behaviors

---

## Project Selection Decision Tree

**Choose based on your background:**

1. **New to mech interp?** → **Weird Model Behavior** (simplest, proven approach)
2. **Strong ML background?** → **CoT Faithfulness** (well-defined, high impact)
3. **Interested in safety?** → **Monitoring with Probes** (practical application)
4. **Want to be novel?** → **User Models Extension** (understudied area)
5. **Strong reasoning background?** → **Thought Anchors Extension** (active area)

---

## Quick Start Checklist

- [ ] Read Neel's vision post (http://neelnanda.io/vision)
- [ ] Choose project from Tier 1 or 2
- [ ] Sketch 20-hour plan
- [ ] Set up tools (Cursor, Gemini 3 Pro, TransformerLens/nnsight)
- [ ] Start simple - don't overcomplicate
- [ ] Track time with Toggl
- [ ] Focus on 1-2 key insights deeply
- [ ] Write clear executive summary with graphs
- [ ] Submit before Dec 23 (or request extension)

---

## Red Flags to Avoid

❌ Grokking projects  
❌ Toy models  
❌ Most SAE work  
❌ GPT-2 (too small)  
❌ Incremental improvements  
❌ Overcomplicated without trying simple first  
❌ No sanity checks  
❌ Poor writing/communication

---

## Green Flags to Include

✅ Clear problem statement  
✅ Start with simple methods  
✅ Strong skepticism and sanity checks  
✅ Self-aware about limitations  
✅ Good graphs and visualizations  
✅ Taught Neel something new  
✅ Well-structured writing  
✅ Deep on 1-2 insights


