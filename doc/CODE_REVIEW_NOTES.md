# Code Review Notes: Analogy-Making Steering Project

## Issue 1: Layer 41 Finding vs. Expected Middle Layers

### Observation
- Current code finds layer 41 (last layer) has highest norm
- Expected: Middle layers (20-35) should show highest divergence
- Gemma-2-9B-IT has 42 layers (0-41), so layer 41 is the final layer

### Possible Explanations

1. **Accumulation Effect**: The analogy vector might accumulate across layers, with the final layer having the largest magnitude simply because it's the sum of all previous transformations.

2. **Measurement Artifact**: Using `norm()` measures magnitude, not necessarily where the concept is *computed*. The concept might be computed in middle layers but have larger magnitude at the end.

3. **Output Preparation**: The last layer is right before the output head, so it might contain the "final form" of the analogy concept ready for generation.

4. **Actual Finding**: It's possible that analogy-making is a late-stage process that happens near the output, not in middle layers.

### Investigation Plan

The `layer_stability_analysis.py` script will help distinguish:

- **If cosine similarity is high (≈1.0) in late layers**: Vector direction is stable, just propagating
  - This suggests computation happened earlier
  - Steering should work in the stable region (middle-late layers)

- **If cosine similarity drops in late layers**: Vector direction is changing
  - This suggests computation is happening late
  - Steering at layer 41 might actually be correct

- **If norm increases but direction stabilizes**: 
  - Concept computed early, magnitude accumulates
  - Best steering layer is where direction stabilizes (not where norm is max)

### Recommendation

1. Run `layer_stability_analysis.py` first to understand the pattern
2. Use cosine similarity (direction stability) rather than just norm (magnitude)
3. The layer ablation study will empirically test which layers work best for steering

---

## Code Quality Issues Found

### ✅ Fixed Issues

1. **Multi-model support**: Created `compute_vector_multi.py` to handle both Gemma and Llama
2. **Layer analysis**: Added `layer_stability_analysis.py` to find where computation happens
3. **Token entanglement**: Added `orthogonalize_vector.py` to remove lexical triggers
4. **Evaluation**: Added `llm_judge_eval.py` for better evaluation than keywords
5. **Sanity checks**: Added `sanity_checks.py` for baselines and confounders

### ⚠️ Potential Issues

1. **Hardcoded paths**: Some scripts use hardcoded paths. Consider making configurable.
2. **Memory management**: Large models might need explicit cleanup between runs.
3. **Error handling**: Some scripts could benefit from better error handling.
4. **Model loading**: Each script loads the model separately - could be optimized.

### 🔍 Things to Verify

1. **Tokenization**: Ensure chat templates are applied correctly for both models
2. **Layer indexing**: Verify layer numbers match between different scripts
3. **Vector shapes**: Ensure vectors from different models have compatible shapes
4. **Device placement**: All tensors should be on the correct device

---

## Expected Findings from Improvements

### Layer Stability Analysis
- **Expected**: Similarity drops in layers 15-25 (computation), then stabilizes
- **If different**: Document the actual pattern - this is still valuable!

### Orthogonalization
- **Expected**: Removes 10-30% of vector magnitude
- **Expected**: "Imagine" token contributes most to removal
- **Test**: Does clean vector still work? Does it reduce mode collapse?

### LLM Judge
- **Expected**: Analogy score increases with coefficient (up to a point)
- **Expected**: Score plateaus or decreases at very high coefficients (mode collapse)
- **Key metric**: The "money plot" showing coefficient vs. score

### Sanity Checks
- **Random baseline**: Should show analogy vector >> random vector
- **Simplicity test**: Should show keyword increase (analogy), not just word decrease (simplification)
- **Negative steering**: Should reduce keywords when subtracting vector

---

## Next Steps After Running Pipeline

1. **Review layer stability plot**: Where does direction stabilize?
2. **Update TARGET_LAYER**: Use the stable region, not just max norm layer
3. **Compare models**: Do Gemma and Llama show similar patterns?
4. **Document findings**: Update executive summary with actual results
5. **Address limitations**: Be honest about what doesn't work

---

## Key Insights to Extract

1. **Where computation happens**: Early/Mid/Late layers?
2. **Model differences**: Do Gemma and Llama use similar mechanisms?
3. **Orthogonalization success**: Can we isolate concept from tokens?
4. **Optimal coefficient**: What range works best?
5. **Failure modes**: When does steering break?

