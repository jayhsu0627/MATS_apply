# Sanity Checks Improvements: Updated Test Values

## Changes Made Based on Findings

### 1. More Granular Coefficient Testing

**Before**: `[0.0, 0.5, 1.0, 2.0]`  
**After**: `[0.0, 0.5, 0.75, 1.0, 1.5, 2.0]`

**Why**: Need finer resolution around validated range (0.5-1.0) to see where effect starts and where mode collapse begins.

### 2. Better Negative Steering Test

**Before**: Used `ANALOGY_PROMPT` ("Explain DNS using an analogy") - already requests analogy  
**After**: Uses `LITERAL_PROMPT` ("Explain DNS technically") - doesn't request analogy

**Why**: Previous test was confounded - baseline already had 2 keywords because prompt requested analogy. New test uses literal prompt to better test bidirectional effect.

### 3. Enhanced Clean vs. Dirty Test

**Before**: Tested limited coefficients  
**After**: Tests `VALIDATED_COEFFICIENTS + MODE_COLLAPSE_TEST_COEFFS` = `[0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`

**Why**: Need to see:
- Validated range (0.5-1.0): Both vectors work
- Transition (1.5-2.0): Where mode collapse starts
- High coefficients (3.0-6.0): Extreme mode collapse difference

### 4. Improved Test Coverage

**Random Baseline**: Now tests 6 coefficients (0.0, 0.5, 0.75, 1.0, 1.5, 2.0)  
**Simplicity Test**: Now tests 6 coefficients (same range)  
**Negative Steering**: Tests 8 coefficients including negative range  
**Clean vs. Dirty**: Tests 9 coefficients covering full range

---

## Expected Improvements

1. **Better negative steering test**: Using literal prompt should show clearer bidirectional effect
2. **More granular analysis**: Finer coefficient resolution will show where effect starts/ends
3. **Better mode collapse detection**: More coefficients in collapse range will show transition point
4. **Clearer clean vs. dirty comparison**: Full range testing will show where clean vector helps most

---

## Key Test Values Summary

```python
# Validated range (no mode collapse)
VALIDATED_COEFFICIENTS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]

# Mode collapse range (where dirty vector fails)
MODE_COLLAPSE_TEST_COEFFS = [2.0, 3.0, 4.0, 6.0]

# Negative steering test
NEGATIVE_TEST_COEFFS = [-1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
```

---

## What to Look For in Results

1. **Random Baseline**: Analogy vector should show increasing keywords (0→3), random should stay at 0
2. **Simplicity**: Keywords should increase (0→3) while word count may increase or stay same
3. **Negative Steering**: 
   - Positive coeffs on literal prompt: Should ADD keywords (0→2-3)
   - Negative coeffs on literal prompt: Should REMOVE keywords (if any present)
4. **Clean vs. Dirty**:
   - At 0.5-1.0: Both should work similarly
   - At 2.0: Dirty should collapse (41+ "imagine"), clean should not (0-3 "imagine")
   - At 4.0+: Dirty should collapse heavily (80 "imagine"), clean should not collapse into "imagine"

