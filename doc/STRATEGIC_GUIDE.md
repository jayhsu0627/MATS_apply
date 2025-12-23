# MATS 10.0 Application Strategy Guide
## Maximizing Acceptance Probability for Neel Nanda's Program

### Executive Summary

**The Opportunity**: Work with Neel Nanda (Google DeepMind MI team leader) - a highly compensated opportunity for a 3rd year PhD student. Top ~34 candidates get 5-week paid exploration phase, ~8 get 12-week paid research phase with high publication potential.

**The Challenge**: 16-20 hours to create a research project that teaches Neel something new and demonstrates strong research potential.

**Key Insight**: Neel's research interests have SHIFTED significantly since 2024. Understanding this shift is critical for success.

---

## Critical Requirements & What Neel Values

### Application Format
- **16-20 hours** max for research (additional 2 hours for executive summary)
- Submit via Google Doc with:
  - Executive summary (1-3 pages, max 600 words) with graphs
  - Full write-up showing progress and learnings
  - Clear structure, good graphs, bullet points
- **Deadline**: Dec 23, 11:59pm PT (late apps until Jan 2)

### What Neel Values Most (in order of importance)

1. **Clarity** - If he understands your claims, evidence, and conclusions → top 20% instantly
2. **Good Taste** - Choosing interesting questions and getting traction
3. **Truth-Seeking & Skepticism** - Questioning results, alternative explanations, sanity checks
4. **Technical Depth & Practicality** - Good handle on tools, willingness to code
5. **Simplicity** - Try obvious methods first, avoid unnecessary complexity
6. **Prioritization** - Deep on 1-2 insights vs. superficial on many
7. **Productivity** - Fast feedback loops, efficient execution
8. **Teaching Him Something New** - Ideal application teaches him something

### What Neel is NOT Interested In (Avoid These!)

❌ **Grokking**  
❌ **Toy models**  
❌ **Most SAE work** (he's moved away from this)  
❌ **Ambitious reverse-engineering** (too pessimistic about this now)  
❌ **Incremental SAE improvements**  
❌ **IOI-style circuit finding on random problems**  
❌ **GPT-2** (too small/dumb for most tasks now)

---

## Neel's Current Research Interests (2024+ Shift)

### His Three Main Categories (in priority order):

1. **Model Biology** - Studying high-level qualitative properties of model behavior
   - Treating models like biological organisms
   - Focus on weird, emergent, safety-relevant phenomena
   - **This is his NEW focus area**

2. **Applied Interpretability** - Practical, real-world applications for safety
   - Not just downstream tasks for grounding
   - Choose problems that actually matter
   - Show interpretability helps in practice

3. **Basic Science** - Still interested but **higher bar** now
   - Less focus than before
   - Must be compelling

### Key Philosophy Shift:
- **From**: Ambitious complete reverse-engineering
- **To**: Pragmatic approaches with clear AGI Safety applications
- **Emphasis**: Start simple, do the obvious thing first

---

## High-Probability Project Ideas (Ranked by Alignment)

### Tier 1: Model Biology Projects (Highest Alignment)

#### 1. **Understanding Weird Model Behavior** ⭐⭐⭐⭐⭐
- **Why it's good**: Directly aligns with his new focus, practical safety relevance
- **Examples**:
  - Why models seem to show self-preservation (his team did this - start simple!)
  - Models that blackmail or fake alignment
  - Debugging unintended behaviors (e.g., why 9.8 > 9.11)
- **Approach**: Start with smallest model showing behavior, read CoT, test causally
- **Key**: Start simple! Reading CoT and prompting often works

#### 2. **Reasoning Model Chain-of-Thought Faithfulness** ⭐⭐⭐⭐⭐
- **Why it's good**: Critical safety question, active research area
- **Questions to explore**:
  - Can we trust CoT for safety monitoring?
  - When is CoT causally important vs. just correlated?
  - Can models hide suspicious reasoning in CoT?
  - What factors lead to unfaithful CoT?
- **Approach**: Study examples of unfaithful CoT, design monitors/metrics
- **Models**: Qwen 3, Nemotron 49B (good reasoning models)

#### 3. **User Models in LLMs** ⭐⭐⭐⭐
- **Why it's good**: Surprising finding, safety-relevant, understudied
- **Questions**:
  - What else do models represent about users?
  - How are these inferred?
  - Do models form dynamic user models (emotion, knowledge)?
  - Do models try to manipulate user states?
- **Approach**: Use probes, steering, analyze behavior changes
- **Reference**: Chen et al (2024) - models infer gender, age, SES from little info

#### 4. **Emergent Misalignment & Out-of-Context Reasoning** ⭐⭐⭐⭐
- **Why it's good**: His scholars did great work here, still open questions
- **Questions**:
  - Why do models generalize so far beyond training?
  - Is it always a single direction? (his scholars suggest this)
  - Why are some solutions easier to learn?
  - Do these effects appear in real use cases?
- **Approach**: Study synthetic document fine-tuning, model diffing
- **Reference**: emergent-misalignment.com

### Tier 2: Applied Interpretability Projects

#### 5. **Monitoring with Probes** ⭐⭐⭐⭐
- **Why it's good**: Practical safety application, SOTA for cheap monitoring
- **Questions**:
  - How to improve probes for edge cases?
  - Information spread across tokens
  - Long context with false positives
  - Attention head probes (Kantamneni et al starting point)
- **Approach**: Design better probes, test on frontier models

#### 6. **Analyzing & Steering Chain-of-Thought** ⭐⭐⭐
- **Why it's good**: Practical use of reasoning model CoT
- **Questions**:
  - Can we understand behavior via CoT?
  - Can we steer by resampling/editing CoT?
  - Use thought anchors techniques
- **Approach**: Read CoT, use resampling, causal interventions

#### 7. **Conditional Steering & Abliteration** ⭐⭐⭐
- **Why it's good**: Practical techniques with safety applications
- **Questions**:
  - Where else can conditional steering apply?
  - How else can abliteration be used?
  - Training data attribution for debugging
- **Approach**: Apply techniques to new problems, measure effectiveness

### Tier 3: Reasoning Models (High Interest Area)

#### 8. **Thought Anchors Extension** ⭐⭐⭐⭐
- **Why it's good**: His scholars' paradigm, active research area
- **Questions**:
  - How to extend thought anchors techniques?
  - Find interesting phenomena using them
  - Identify weaknesses/limitations
- **Approach**: Use sentence-level analysis, resampling, causal interventions
- **Reference**: thought-anchors.com

#### 9. **Steganography in CoT** ⭐⭐⭐
- **Why it's good**: Novel, security-relevant
- **Questions**:
  - Can models encode info in CoT?
  - Train models with reasoning we don't understand
  - Use interpretability to decipher it
- **Approach**: Train models, analyze CoT, use interp tools
- **Note**: Must allow model to do tasks it couldn't do without CoT

### Tier 4: Concept Representations

#### 10. **Specific Concept Representations** ⭐⭐⭐
- **Why it's good**: Safety-relevant concepts
- **Concepts to study**:
  - Truth probes that generalize
  - Deception probes
  - Uncertainty representation
  - Misalignment direction (why does it exist?)
  - Evaluation awareness
- **Approach**: Train probes, analyze representations, test generalization
- **Models**: Nemotron 49B good for evaluation awareness

#### 11. **Model Diffing** ⭐⭐⭐
- **Why it's good**: Powerful isolation technique
- **Questions**:
  - What changes during fine-tuning?
  - Chat-tuning vs. base models
  - Reasoning fine-tuning changes
  - What happens during synthetic document fine-tuning?
- **Approach**: Compare before/after, use KL divergence, analyze specific behaviors
- **Reference**: His past scholar's work on diffing chat finetuning

### Tier 5: Circuit Analysis (Lower Priority)

#### 12. **Attribution Graphs** ⭐⭐
- **Why it's lower**: Less focus than before, but still interesting
- **Questions**:
  - Are attribution graphs useful for model biology?
  - Find interesting things on Neuronpedia graphs
  - Overcome limitations
  - When is precision important?
- **Approach**: Use graphs on Neuronpedia, test hypotheses
- **Note**: He's more skeptical now, need strong results

#### 13. **Baseline Methods** ⭐⭐
- **Why it's good**: Simple methods often overlooked
- **Questions**:
  - Automate linear probe testing
  - Scale CoT reading/analysis
  - Best practices for prompt observation
  - Can we automate hypothesis generation + validation?
- **Approach**: Build automation, test effectiveness

---

## Strategic Advice for 3rd Year PhD Student

### Your Advantages:
1. **Research experience** - You know how to do research
2. **Time management** - PhD teaches prioritization
3. **Technical skills** - Likely strong coding/ML background
4. **Academic writing** - Can write clearly

### Your Challenges:
1. **Mech interp may be new** - But Neel values potential over experience
2. **20 hours is tight** - Need efficient execution
3. **Competition** - Many strong applicants

### Maximizing Your Chances:

#### 1. **Choose the Right Project**
- **Best**: Model Biology project on weird behavior or CoT faithfulness
- **Why**: Aligns with his current excitement, practical safety relevance
- **Avoid**: SAE work, toy models, grokking

#### 2. **Start Simple, Be Pragmatic**
- Read CoT first (often works!)
- Use prompting before complex techniques
- Train linear probes before SAEs
- **Key**: Show you tried obvious things first

#### 3. **Focus on Teaching Him Something**
- Don't just confirm obvious hypotheses
- Find surprising patterns
- Question your results deeply
- Show self-awareness of limitations

#### 4. **Excellent Communication**
- Executive summary must stand alone
- Use graphs effectively
- Clear structure, bullet points
- Explain why, not just what

#### 5. **Demonstrate Research Skills**
- Show thought process
- Pivot when stuck (don't just give up)
- Compare to baselines
- Sanity checks everywhere

#### 6. **Use LLMs Effectively**
- He strongly recommends this!
- Use Cursor for coding
- Use Gemini 3 Pro for research (1M context)
- Put relevant docs in context
- Use anti-sycophancy prompts for feedback

#### 7. **Time Management**
- Track time with Toggl
- Spend max 5 hours reading papers
- Focus on 1-2 key insights deeply
- Don't get caught in rabbit holes

---

## Common Mistakes to Avoid

### Critical Errors:
1. ❌ Not acknowledging limitations (or worse, pretending negatives are positives)
2. ❌ Not doing sanity checks
3. ❌ Overcomplicating without trying simple methods first
4. ❌ Working with models too small for the task (e.g., GPT-2)
5. ❌ Not looking at your data - read examples!
6. ❌ Poor writing - if he can't understand, you're rejected
7. ❌ Choosing uninteresting problems (incremental SAE improvements, etc.)
8. ❌ Getting stuck and continuing doomed projects (know when to pivot!)

### What Makes Applications Stand Out:
✅ Clear, well-structured writing with good graphs  
✅ Interesting question with compelling results  
✅ Strong skepticism and truth-seeking  
✅ Technical depth with practical execution  
✅ Simplicity - tried obvious things first  
✅ Deep on 1-2 insights vs. superficial on many  
✅ Showed thought process and pivoted when needed  
✅ Taught him something new

---

## Recommended Project Selection Strategy

### For Maximum Probability:

**Option A: CoT Faithfulness (Highest Probability)**
- **Why**: Critical safety question, active area, aligns with his interests
- **Approach**: 
  1. Study examples of unfaithful CoT (Arcuschin et al, Chen et al)
  2. Design monitors/metrics for faithfulness
  3. Test on reasoning models (Qwen 3, Nemotron 49B)
  4. Find when CoT is causally important
- **Time**: 16-20 hours is feasible
- **Risk**: Medium (well-defined problem)

**Option B: Weird Model Behavior**
- **Why**: Direct model biology focus, practical safety relevance
- **Approach**:
  1. Pick a weird behavior (self-preservation, blackmail, etc.)
  2. Start with smallest model showing it
  3. Read CoT, test with prompts (start simple!)
  4. Understand what's really happening
- **Time**: 16-20 hours is feasible
- **Risk**: Low (he did this, shows it works)

**Option C: User Models Extension**
- **Why**: Surprising finding, understudied, safety-relevant
- **Approach**:
  1. Extend Chen et al findings
  2. What else do models represent about users?
  3. Use probes and steering
  4. Test dynamic user models
- **Time**: 16-20 hours is feasible
- **Risk**: Medium (novel area)

### My Recommendation:
**Start with Option A (CoT Faithfulness)** - it's the most aligned with his current interests, has clear safety relevance, and is a well-defined problem that can show strong research skills in 20 hours.

---

## Resources to Study Before Starting

### Essential Reading:
1. **Neel's blog posts**:
   - [Pragmatic vision for interpretability](http://neelnanda.io/vision)
   - [How to become a mech interp researcher](https://neelnanda.io/getting-started)
   - [Ways interp can help AGI go well](http://neelnanda.io/agenda)

2. **Key Papers** (read abstracts + key sections):
   - Chen et al on user models
   - Arcuschin et al on unfaithful CoT
   - Thought anchors paper
   - Emergent misalignment work

3. **Tools to Learn**:
   - TransformerLens (for <=9B models)
   - nnsight (for larger models)
   - Cursor (for coding)
   - Gemini 3 Pro (for research)

### Time Budget:
- **Before project**: Unlimited general learning (read papers, tutorials)
- **During project**: Max 5 hours reading papers
- **Focus**: Get hands dirty, run experiments, get feedback from reality

---

## Final Checklist Before Submission

- [ ] Executive summary (1-3 pages, max 600 words) with graphs
- [ ] Clear problem statement and why it's interesting
- [ ] High-level takeaways clearly stated
- [ ] Key experiments with graphs and explanations
- [ ] Self-awareness of limitations
- [ ] Sanity checks documented
- [ ] Baselines compared (if applicable)
- [ ] Time tracking screenshot included
- [ ] Google Doc shared with link access
- [ ] Writing is clear and structured
- [ ] You taught Neel something new

---

## Success Metrics

**What "Accept" Looks Like:**
- Clear, well-communicated findings
- Interesting question with compelling results
- Strong research skills demonstrated
- Self-aware about limitations
- Taught Neel something new
- Well-executed in 20 hours

**Remember**: Neel values potential over credentials. A strong application task is enough for acceptance, whatever your background. Focus on demonstrating research ability, not showing off credentials.

---

## Next Steps

1. **Read Neel's vision post** (http://neelnanda.io/vision) - understand his philosophy
2. **Choose your project** - I recommend CoT Faithfulness or Weird Behavior
3. **Sketch a plan** - How will you spend 20 hours?
4. **Start simple** - Don't overcomplicate
5. **Track your time** - Use Toggl
6. **Write clearly** - Spend the extra 2 hours on executive summary
7. **Submit before Dec 23** (or request extension by Jan 2)

**Good luck! This is an incredible opportunity. Focus on doing good research and communicating it clearly.**


