# Analogy-Making Steering: 10-Hour Execution Plan

## Current Status (Baseline Complete)

✅ **Phase 1: Data & Vector Computation** (DONE)
- Generated 100 paired prompts (literal vs analogy) across 5 domains
- Computed analogy steering vector for Gemma-2-9B-IT
- Found layer 41 has highest norm (unexpected - need to investigate)

✅ **Phase 2: Basic Steering** (DONE)
- Implemented steering at multiple layers (20, 30, 41)
- Basic coefficient sweep showing effect
- Quantitative measurement via keyword probabilities

⚠️ **Phase 3: Rigor Checks** (MISSING - Critical Gap)
- No random vector baseline
- No orthogonalization (token entanglement issue)
- No layer stability analysis
- No LLM-as-Judge evaluation
- No simplicity confounder test

---

## 10-Hour Improvement Plan

### Hour 1-2: Code Review & Multi-Model Extension

**Tasks:**
1. **Investigate Layer 41 Finding** (30 min)
   - Check if norm calculation is correct
   - Verify if this is expected or a bug
   - Compare with layer stability analysis

2. **Extend to Two Models** (90 min)
   - Modify `compute_vector.py` to support both Gemma-2-9B-IT and Llama-3.1-70B-Instruct
   - Run vector computation for both models
   - Save vectors with model-specific names
   - Compare vector norms across models

**Deliverables:**
- `compute_vector_multi.py` - Multi-model version
- `analogy_vector_gemma.pt` and `analogy_vector_llama.pt`
- Quick comparison plot of norms across models

---

### Hour 3-4: Layer Stability Analysis (The "Turning Point" Analysis)

**Tasks:**
1. **Implement Cosine Similarity Analysis** (60 min)
   - Calculate cosine similarity between layer L and L+1 for each model
   - Identify where vector direction stabilizes (computation vs propagation)
   - Create visualization showing stability across layers

2. **Layer Ablation Study** (60 min)
   - Test steering effectiveness at every 5 layers (0, 5, 10, ..., 40)
   - Measure analogy score (using keyword proxy initially)
   - Create "Steering Success vs Layer" plot
   - Identify optimal layer range

**Deliverables:**
- `layer_stability_analysis.py` - Cosine similarity between layers
- `layer_ablation.py` - Steering effectiveness across layers
- `layer_stability_plot.png` and `layer_ablation_plot.png`

**Key Insight to Find:** Where does the analogy concept get computed vs. just propagated?

---

### Hour 5-6: Orthogonalization (The "Refusal Paper" Move)

**Tasks:**
1. **Identify Token Entanglement** (30 min)
   - Test if high coefficients cause mode collapse into "Imagine"
   - Identify problematic tokens (likely: "Imagine", "like", "similar")

2. **Implement Orthogonalization** (90 min)
   - Project analogy vector orthogonal to unembedding directions of problematic tokens
   - Formula: `v_clean = v - (v · u) / (u · u) * u` where u is token direction
   - Test with multiple tokens if needed
   - Compare clean vs. dirty vector steering

**Deliverables:**
- `orthogonalize_vector.py` - Remove token entanglement
- `analogy_vector_clean_gemma.pt` and `analogy_vector_clean_llama.pt`
- Comparison: Does mode collapse disappear?

**Key Question:** Can we isolate the "concept" from the "lexical trigger"?

---

### Hour 7-8: LLM-as-a-Judge Evaluation

**Tasks:**
1. **Set Up Judge Model** (30 min)
   - Use Qwen-72B or Llama-70B as judge
   - Create evaluation prompt template
   - Test judge consistency on sample outputs

2. **Implement Evaluation Loop** (90 min)
   - Generate outputs with steering coefficients: 0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0
   - For each output, get judge score (1-5) on analogy usage
   - Run for both models, both clean and dirty vectors
   - Create "Steering Coefficient vs. Analogy Score" plot

**Deliverables:**
- `llm_judge_eval.py` - Automated evaluation system
   - Judge prompt template
   - Batch evaluation function
   - Score aggregation and plotting
- `analogy_score_plot.png` - The "Money Plot"
- `judge_scores.json` - Raw evaluation data

**Key Metric:** Average Analogy Score (1-5) vs. Steering Coefficient

---

### Hour 9: Sanity Checks & Baselines

**Tasks:**
1. **Random Vector Baseline** (30 min)
   - Generate random vector with same norm as analogy vector
   - Test steering with random vector
   - Compare to analogy vector steering

2. **Simplicity Confounder Test** (30 min)
   - Apply vector to poetry generation prompt
   - Does it make poetry simpler or force metaphors?
   - Test on non-technical prompts

3. **Negative Steering Test** (30 min)
   - Test subtracting the vector (coefficient = -1, -2, etc.)
   - Does it remove analogies from analogy prompts?
   - Verify bidirectional effect

**Deliverables:**
- `sanity_checks.py` - All baseline tests
- `sanity_check_results.md` - Summary of findings
- Plots comparing analogy vs. random vs. negative steering

**Key Questions:**
- Is the effect specific to the analogy vector?
- Is it just "simpler language" or actually "analogical"?
- Does it work bidirectionally?

---

### Hour 10: Integration & Documentation

**Tasks:**
1. **Create Unified Evaluation Script** (30 min)
   - Combine all evaluation methods
   - Run full pipeline: compute → orthogonalize → evaluate
   - Generate all plots in one run

2. **Write Executive Summary Draft** (30 min)
   - Key findings (1-2 insights)
   - The "Money Plot" (Steering vs. Analogy Score)
   - Layer stability insight
   - Orthogonalization result
   - Limitations and future work

**Deliverables:**
- `run_full_pipeline.py` - Complete evaluation pipeline
- `EXECUTIVE_SUMMARY.md` - Draft summary with plots
- All final plots and results

---

## Success Criteria

### Minimum Viable (Must Have):
- ✅ Multi-model comparison (Gemma vs. Llama)
- ✅ Layer stability analysis showing where computation happens
- ✅ Orthogonalization removing token entanglement
- ✅ LLM-as-Judge evaluation with clear plot
- ✅ Random vector baseline showing specificity

### Strong Improvements (Should Have):
- ✅ Layer ablation showing optimal steering layer
- ✅ Simplicity confounder test
- ✅ Negative steering test
- ✅ Clear executive summary with key insights

### Nice to Have (If Time Permits):
- Qualitative analysis of best/worst analogies
- Cross-domain generalization test
- Comparison of clean vs. dirty vectors

---

## Key Insights to Document

1. **Layer Analysis:** Where does analogy computation happen? (Early/Mid/Late layers)
2. **Model Comparison:** Do Gemma and Llama use similar mechanisms?
3. **Orthogonalization:** Can we isolate concept from lexical triggers?
4. **Steering Effectiveness:** What coefficient range works best?
5. **Limitations:** When does steering fail? What are edge cases?

---

## Risk Mitigation

**Risk 1: Layer 41 finding is unexpected**
- **Mitigation:** Run layer stability analysis first to understand if this is real or artifact
- **Fallback:** If it's a bug, fix and recompute

**Risk 2: Orthogonalization doesn't help**
- **Mitigation:** Document the attempt - even failures show good research practice
- **Fallback:** Report that token entanglement persists (still valuable finding)

**Risk 3: LLM Judge is inconsistent**
- **Mitigation:** Test judge on known examples first, use temperature=0
- **Fallback:** Use keyword-based proxy as backup metric

**Risk 4: Time runs out**
- **Prioritize:** Orthogonalization > Layer Analysis > LLM Judge > Sanity Checks
- **Minimum:** Must have orthogonalization and layer analysis for competitive application

---

## Files to Create/Modify

### New Files:
- `compute_vector_multi.py` - Multi-model vector computation
- `layer_stability_analysis.py` - Cosine similarity between layers
- `layer_ablation.py` - Steering effectiveness across layers
- `orthogonalize_vector.py` - Remove token entanglement
- `llm_judge_eval.py` - LLM-as-Judge evaluation
- `sanity_checks.py` - Baseline tests
- `run_full_pipeline.py` - Complete pipeline

### Modified Files:
- Update `test_steering.py` to support both models
- Update `sweep_coefficients.py` to use clean vectors

### Output Files:
- `analogy_vector_gemma.pt`, `analogy_vector_llama.pt`
- `analogy_vector_clean_gemma.pt`, `analogy_vector_clean_llama.pt`
- `layer_stability_plot.png`
- `layer_ablation_plot.png`
- `analogy_score_plot.png` (The Money Plot)
- `sanity_check_results.md`

---

## Next Steps (After 10 Hours)

If accepted to interview phase:
1. Deeper mechanistic analysis (which circuits?)
2. Cross-domain generalization study
3. Connection to user models (do models adapt analogies?)
4. Safety implications (can we detect/manipulate analogy usage?)

---

## Notes from Google AI Studio Suggestions

Key priorities:
1. **Layer Analysis** - Find where computation happens (high priority)
2. **Orthogonalization** - Remove token entanglement (highest value move)
3. **LLM Judge** - Better evaluation than keywords (critical for rigor)
4. **Don't scale to 100k prompts** - Current 100 is sufficient

The orthogonalization move is the "accepted" move - even if it fails, the attempt shows good research practice.

