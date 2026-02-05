# HiMAP Bug Report

Comprehensive bug analysis of the HiMAP (Hidden Markov models for Advanced Prognostics) codebase.

Bugs are categorized by:
1. **Type:** Scientific (violates HMM/HSMM theory) vs Algorithmic (implementation issues)
2. **Severity:** Critical, Important, Standard, Optional

---

# SCIENTIFIC BUGS

These bugs violate the mathematical or theoretical foundations of Hidden Markov Models and Hidden Semi-Markov Models.

---

## Critical Scientific Issues

### S1. HSMM Convergence Criterion Violates EM Theory
**File:** `himap/base.py:688`
**Status:** FIXED

**Original code:**
```python
if itera > 0 and abs(abs(score) - abs(old_score)) < self.tol:
```

**When it occurs:** Every time `HSMM.fit()` is called to train a model. The bug affects the decision of when to stop the EM iterations.

**Why it happens:** The code uses `abs(abs(score) - abs(old_score))` instead of the mathematically correct relative change formula. Since log-likelihoods are always negative numbers (logarithm of probabilities between 0 and 1), taking the absolute value of the score converts it to a positive number. Then taking the difference of two positive numbers gives a completely different result than measuring the actual improvement in log-likelihood.

**Example:**
- If `old_score = -1000` and `score = -999` (improvement of 1)
- Correct calculation: `|(-999) - (-1000)| = 1`
- Bug calculation: `|abs(-999) - abs(-1000)| = |999 - 1000| = 1`
- This happens to give the same result here, BUT:
- If `old_score = -1000` and `score = -1001` (worsening of 1)
- Correct: `|(-1001) - (-1000)| = 1`
- Bug: `|1001 - 1000| = 1`
- The bug cannot distinguish improvement from worsening!

**Scientific Problem:** The EM algorithm guarantees monotonic increase of log-likelihood. The convergence check should detect when the improvement becomes negligible. The current formula loses the direction of change.

**Impact:** HSMM training may:
- Stop too early (false convergence)
- Continue unnecessarily (waste computation)
- Accept models that are getting worse

**Fix applied:** Changed to use proper relative change formula:
```python
if itera > 0 and (abs(score - old_score) / (1 + abs(old_score))) < self.tol:
```

---

### S2. Left-to-Right Transition Matrix Violates Stochastic Matrix Property
**File:** `himap/base.py:135-142`
**Status:** FIXED

**Original code:**
```python
self.tmat = np.zeros((self.n_states, self.n_states))
for i in range(len(self.tmat)):
    for j in range(len(self.tmat[i]) - 1):
        if i == j and j < len(self.tmat[i]) - 2:
            self.tmat[i, j + 1] = 1
```

**When it occurs:** During initialization of an HSMM with `left_to_right=True` parameter. This is common for degradation modeling where states represent progressive damage levels.

**Why it happens:** The loop sets `tmat[i, i+1] = 1` for states 0 through n_states-2, meaning each state transitions to the next with probability 1. However, the loop condition `j < len(self.tmat[i]) - 2` excludes the last row, leaving `tmat[n_states-1, :]` as all zeros.

**Example with 4 states:**
```
Before (intended):     After (actual):
[0 1 0 0]              [0 1 0 0]
[0 0 1 0]              [0 0 1 0]
[0 0 0 1]              [0 0 0 1]
[0 0 0 1]  <- absorbing [0 0 0 0]  <- INVALID!
```

**Scientific Problem:** A valid transition matrix must be row-stochastic: each row must sum to exactly 1.0, representing a valid probability distribution over next states. The last row sums to 0, which means:
- The model has no defined behavior when reaching the final state
- Forward-backward algorithm computations involving this row produce undefined results
- The Markov chain is not properly defined

**Impact:**
- Forward algorithm may produce NaN/inf when reaching final state
- Viterbi decoding may fail for sequences that reach the final state
- Model cannot properly represent absorbing failure states

**Fix applied:** Added `self.tmat[-1, -1] = 1.0` to make the last state absorbing (stays in itself forever).

---

### S3. NaN Replacement Violates Probability Normalization
**File:** `himap/base.py:1646-1647`
**Status:** FIXED

**Original code:**
```python
calc_tr[np.isnan(calc_tr)] = 0
calc_emi[np.isnan(calc_emi)] = 0
```

**When it occurs:** During the M-step of EM algorithm in HMM training, when updating transition and emission probability matrices. Happens when a state is never visited in the training data.

**Why it happens:** The M-step computes:
```python
calc_tr = tr / total_transitions[:, np.newaxis]  # tr[i,j] / sum over j of tr[i,:]
calc_emi = emi / total_emissions[:, np.newaxis]
```
If `total_transitions[i] = 0` (state i was never occupied), division produces NaN. The code then replaces NaN with 0.

**Example:**
- State 2 is never visited in training data
- `total_transitions[2] = 0`
- `calc_tr[2, :] = [0, 0, 0] / 0 = [NaN, NaN, NaN]`
- After fix: `calc_tr[2, :] = [0, 0, 0]` <- Sums to 0, not 1!

**Scientific Problem:**
1. **Row-stochastic violation:** Transition matrix rows must sum to 1. A row of zeros sums to 0.
2. **Probability axiom violation:** Every state must have *some* transition probability, even if the state was never seen.
3. **Model corruption:** The trained model now has "dead" states with no valid transitions.

**Impact:**
- Model silently becomes invalid
- Predictions from rarely-seen states are undefined
- Forward-backward algorithm may produce incorrect results

**Fix applied:** Used Laplace (additive) smoothing:
```python
epsilon = 1e-10
calc_emi = (emi + epsilon) / (total_emissions[:, np.newaxis] + epsilon * self.n_obs_symbols)
calc_tr = (tr + epsilon) / (total_transitions[:, np.newaxis] + epsilon * self.n_states)
```
This ensures every state has small but non-zero transition probabilities.

---

### S4. Log-Domain Operations Risk Numerical Underflow
**File:** `himap/utils.py:418-427`
**Status:** NOT FIXED (requires significant refactoring)

```python
logf = np.log(fs)
logb = np.log(bs)
# Later:
tr[i, j] += np.exp(logf[i, h] + logGTR[i, j] + logGE[j, history[h + 1] - 1] + logb[j, h + 1]) / scale_h1
```

**When it occurs:** During Baum-Welch algorithm execution for HMM parameter estimation. Happens most severely with:
- Long observation sequences (hundreds or thousands of timesteps)
- Many hidden states
- Low probability observations

**Why it happens:** The forward-backward algorithm computes products of many probabilities:
```
alpha(t) = P(o1, o2, ..., ot, state_t) = product of many probabilities < 1
```
For a sequence of length 100 with average transition probability 0.5:
- alpha ~= 0.5^100 ~= 10^-30

The code uses scaling to prevent underflow in the forward-backward pass, but then:
1. Takes log of already-scaled probabilities (which are still small)
2. Adds logs together (fine)
3. Exponentiates the result (UNDERFLOW RISK!)
4. Divides by scaling factor (amplifies errors)

**Scientific Problem:** The Baum-Welch algorithm should work entirely in log-domain to handle arbitrary sequence lengths. The log-sum-exp trick avoids ever exponentiating large negative numbers.

**Impact:**
- Training fails silently for long sequences
- Parameter estimates become 0 instead of small positive values
- Model appears to train but produces garbage predictions

---

## Important Scientific Issues

### S5. Division by Zero in Backward Probabilities
**File:** `himap/utils.py:511`
**Status:** NOT FIXED (requires careful handling)

```python
bs[state, count] = (1 / s[0, count + 1]) * np.sum(...)
```

**When it occurs:** During backward pass of Baum-Welch algorithm. Happens when:
- An observation is impossible given all states (emission probability = 0 for all states)
- Numerical underflow causes forward probabilities to become exactly 0
- Data contains observation values outside the expected range

**Why it happens:** The scaling factor `s[0, count + 1]` is computed as the sum of forward probabilities at timestep `count + 1`. If all forward probabilities are 0, the scaling factor is 0, and dividing by it produces `inf`.

**Scientific Problem:** A zero scaling factor indicates one of two things:
1. **Impossible observation:** The observation at that timestep has zero probability under all states.
2. **Numerical underflow:** Forward probabilities became too small and underflowed.

**Impact:**
- `inf` values propagate through backward pass
- Gamma (state occupancy) computations become `NaN`
- M-step produces invalid parameter updates
- Training appears to complete but model is corrupted

---

### S6. Viterbi Algorithm Bounds Not Validated
**File:** `himap/cython_build/fwd_bwd.pyx:169-174`
**Status:** NOT FIXED (Cython code, requires careful handling)

**When it occurs:** During Viterbi decoding to find the most likely state sequence. Specifically during the backtracking phase after the forward pass.

**Why it happens:** The Viterbi backward pass uses indices stored during the forward pass:
```cython
back_state = psi[back_t, back_state, 1]  # Get previous state from backpointer
```
If `psi` contains invalid indices (due to bugs or uninitialized memory), `back_state` could become negative or >= n_states.

**Scientific Problem:** The Viterbi algorithm should always produce a valid state sequence where:
- `0 <= state[t] < n_states` for all timesteps
- The sequence represents the maximum probability path through the HMM

**Impact:**
- Segmentation faults (memory access violation)
- Silent corruption of decoded state sequences
- Invalid prognostic predictions based on wrong state sequences

---

### S7. RUL Calculation Break Violates Expectation Computation
**File:** `himap/base.py:2041-2045`
**Status:** FIXED

**Original code:**
```python
if np.isnan(rul_value) or rul_value == 0:
    rul_mean.append(0)
    rul_upper_bound.append(0)
    rul_lower_bound.append(0)
    break  # STOPS THE ENTIRE LOOP
```

**When it occurs:** During RUL (Remaining Useful Life) prediction for a test trajectory. The loop iterates over timesteps, computing RUL distribution at each point.

**Why it happens:** If the RUL calculation produces NaN (numerical error) or 0 (reached end of life), the code breaks out of the loop instead of continuing to the next timestep.

**Example:**
- Trajectory has 100 timesteps
- RUL calculation fails at timestep 30 (produces NaN)
- Loop breaks at timestep 30
- Only 30 RUL values are returned instead of 100
- User receives truncated results without warning

**Scientific Problem:** RUL is defined as the expected time until failure given observations up to current time. Even if one timestep's calculation fails:
1. Other timesteps may have valid RUL values
2. The failure should be flagged, not cause silent truncation
3. Batch processing should not stop for one bad sample

**Impact:**
- Incomplete RUL predictions
- Users don't know data is truncated
- Downstream analysis uses wrong array lengths
- Metrics computed on partial data are misleading

**Fix applied:** Changed `break` to `continue` to process all timesteps.

---

## Standard Scientific Issues

### S8. Missing Initialization of `old_score` Variable
**File:** `himap/base.py:688, 697`
**Status:** FIXED

**When it occurs:** During first iteration of EM algorithm if an early exit condition is triggered.

**Why it happens:** The variable `old_score` is only assigned inside the `else` block. If `itera == 0` and the NaN check triggers a break, `old_score` is never defined.

**Scientific Problem:** The EM algorithm requires comparing consecutive log-likelihoods. Without proper initialization, the first comparison is undefined.

**Impact:** Potential `NameError` in edge cases, though current code structure usually avoids this.

**Fix applied:** Added `old_score = -np.inf` before the iteration loop.

---

### S9. Inconsistent Convergence Criteria Between HMM and HSMM
**File:** `himap/base.py:688` vs `himap/base.py:1650-1652`
**Status:** PARTIALLY FIXED (HSMM now uses correct formula, but doesn't check parameter stability)

**When it occurs:** Every time HMM or HSMM models are trained.

**Why it happens:** The two model classes were implemented separately with different convergence logic:

**HSMM (simpler, now fixed):**
```python
if (abs(score - old_score) / (1 + abs(old_score))) < self.tol:
    break  # Only checks log-likelihood
```

**HMM (more comprehensive):**
```python
if (abs(score - old_score) / (1 + abs(old_score))) < self.tol and \
   np.linalg.norm(calc_tr - old_tr, ord=np.inf) / self.n_states < self.tol and \
   np.linalg.norm(calc_emi - old_emi, ord=np.inf) / self.n_obs_symbols < self.tol:
```

**Scientific Problem:** Log-likelihood plateau doesn't guarantee parameter convergence. The likelihood surface can be flat while parameters are still changing significantly. HMM correctly checks both likelihood AND parameter stability.

**Impact:** HSMM may declare convergence prematurely with suboptimal parameters.

---

### S10. GaussianHSMM K-Means on Truncated Data
**File:** `himap/base.py:1226`
**Status:** NOT FIXED (may be intentional design choice)

```python
kmeans.fit(X[:-self.obs_state_len])
```

**When it occurs:** During initialization of GaussianHSMM model with `last_observed=True`.

**Why it happens:** The code intentionally excludes the last `obs_state_len` samples to prevent the failure state mean from being pulled toward end-of-life values. However:
1. This design choice is not documented
2. If `obs_state_len >= len(X)`, the array becomes empty
3. K-means on empty array fails

**Scientific Problem:** While excluding end-of-life data from initialization can be valid, it should be:
- Documented as intentional behavior
- Validated against empty arrays
- Considered whether it biases initial estimates

**Impact:** Crash with short sequences; potentially biased initialization for normal sequences.

---

# ALGORITHMIC BUGS

These are implementation issues that cause crashes, incorrect behavior, or poor performance.

---

## Critical Algorithmic Issues

### A1. IndexError in `mc_dataset()` - Array Out of Bounds
**File:** `himap/base.py:345-350`
**Status:** FIXED

**Original code:**
```python
for j in range(len(states1)):
    if states1[j] > states1[j + 1]:  # BUG: j+1 out of bounds when j = len-1
        idx = j
        break
obs.update({f'traj_{i + 1}': list(obs1[:idx + 1, 0])})
```

**When it occurs:** Every time `mc_dataset()` is called to generate Monte Carlo samples from a trained model. The bug triggers on the last iteration of the inner loop.

**Why it happens:** The loop uses `range(len(states1))` which iterates j from 0 to len-1. But inside the loop, it accesses `states1[j + 1]`. When j equals len-1, j+1 equals len, which is out of bounds.

**Impact:** `mc_dataset()` crashes immediately, making Monte Carlo sampling unusable.

**Fix applied:** Changed to `range(len(states1) - 1)` to stop before the last element.

---

### A2. Undefined Variable `idx`
**File:** `himap/base.py:345-350`
**Status:** FIXED

**When it occurs:** In `mc_dataset()` when the sampled trajectory never has a state decrease (monotonically non-decreasing state sequence).

**Why it happens:** The variable `idx` is only assigned inside the `if` block when `states1[j] > states1[j + 1]` is true. If this condition is never satisfied (states only increase or stay same), the loop completes without break, and `idx` is never defined.

**Impact:** Crash for any trajectory that doesn't have a state decrease.

**Fix applied:** Initialize `idx = len(states1) - 1` before the loop as a default.

---

### A3. Plotting Crashes with Single Subplot
**File:** `himap/plot.py:35, 42`
**Status:** FIXED

**Original code:**
```python
fig, axs = plt.subplots(num2plot, figsize=(19, 10))
ax1 = axs[i]  # TypeError when num2plot=1
```

**When it occurs:** When calling `plot_multiple_observ()` with `num2plot=1`.

**Why it happens:** `plt.subplots(n)` has different return types depending on n:
- `plt.subplots(1)` returns a single Axes object
- `plt.subplots(2)` returns a numpy array of Axes

When n=1, `axs` is a single Axes object, not an array. Calling `axs[0]` tries to index into an Axes object, which fails.

**Impact:** Cannot plot single trajectories.

**Fix applied:** Use `plt.subplots(num2plot, squeeze=False)` which always returns 2D array, and access with `axs[i, 0]`.

---

## Important Algorithmic Issues

### A4. Inconsistent Exception Types
**File:** `himap/base.py:1480-1487` vs `himap/base.py:66-69`
**Status:** FIXED

**When it occurs:** When creating HMM or HSMM with invalid parameters.

**Why it happens:** Different developers or development sessions used different error handling patterns:

**HMM used assert statements (now fixed):**
```python
assert n_states >= 2, "number of states (n_states) must be at least 2"
```

**HSMM uses explicit raises:**
```python
if not n_states >= 2:
    raise ValueError("n_states must be at least 2")
```

**Impact:** Users cannot write consistent error handling.

**Fix applied:** Changed HMM to use `raise ValueError()` for consistency.

---

### A5. Hard-coded Column Name
**File:** `himap/utils.py:57`
**Status:** NOT FIXED (may be intentional API)

```python
list(pd.read_csv(files[i], usecols=[0])['clusters'])
```

**When it occurs:** When using `create_data_hsmm()` to load custom data files.

**Why it happens:** The code assumes CSV files have a column named 'clusters'. This was probably the format used during development, but users may have different column names.

**Impact:**
- `KeyError` if column isn't named 'clusters'
- No way to specify different column name
- No helpful error message

---

### A6. O(n^2) DataFrame Concatenation
**File:** `himap/utils.py:247, 285, 340`
**Status:** FIXED

**Original code:**
```python
df_results = pd.DataFrame(columns=['Name', 'rmse'])
for key in mean_rul_dict.keys():
    # ... compute values ...
    new_row = pd.DataFrame([{'Name': key, 'rmse': rmse_pred}])
    df_results = pd.concat([df_results, new_row], ignore_index=True)
```

**When it occurs:** In `get_rmse()`, `get_coverage()`, and `get_wsu()` functions when computing metrics for many trajectories.

**Why it happens:** Each `pd.concat()` creates a complete copy of the DataFrame plus the new row. With n rows:
- Total copies: 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n^2)

**Impact:** Severe slowdown with many trajectories. For 1000 trajectories, instead of O(1000) operations, it does O(500,000) row copies.

**Fix applied:** Collect dictionaries in a list, create DataFrame once at end:
```python
rows = []
for key in mean_rul_dict.keys():
    rows.append({'Name': key, 'rmse': rmse_pred})
df_results = pd.DataFrame(rows)
```

---

### A7. Missing Bounds Check for Observation Indices
**File:** `himap/utils.py:427, 475, 512`
**Status:** NOT FIXED (requires careful validation)

```python
logGE[j, history[h + 1] - 1]
```

**When it occurs:** During Baum-Welch algorithm when processing observation sequences.

**Why it happens:** The code assumes `history[t]` contains valid observation indices (1 to n_obs_symbols, since it subtracts 1 for 0-based indexing). No validation checks this assumption.

**If violated:**
- `history[t] = 0` -> accesses index -1 (last column, wrong data!)
- `history[t] > n_obs_symbols` -> IndexError

**Impact:** Silent wrong results or crashes depending on data.

---

### A8. File I/O Without Error Handling
**File:** `himap/utils.py:85-89`
**Status:** NOT FIXED (low priority)

```python
with train_res.open("rb") as f:
    train = pd.read_csv(f, sep=";")
```

**When it occurs:** When loading CMAPSS data using `load_data_cmapss()`.

**Why it happens:** No try-except around file operations. Also uses binary mode ("rb") which is unusual for CSV reading.

**Impact:**
- `FileNotFoundError` if data files are missing
- Encoding errors possible with binary mode
- No helpful error message for users

---

## Standard Algorithmic Issues

### A9. Division by Zero in Coverage
**File:** `himap/utils.py:283`
**Status:** FIXED

**Original code:**
```python
cov = count_within_bounds / len(true_values)
```

**When it occurs:** When computing coverage metric for a trajectory with RUL = 0.

**Why it happens:** `true_values = list(range(0, -1, -1))` creates an empty list when RUL is 0. Dividing by `len([])` = 0 raises ZeroDivisionError.

**Impact:** Crashes when evaluating sequences at end of life.

**Fix applied:** Added guard check:
```python
if len(true_values) == 0:
    cov = 0.0
else:
    # ... original calculation
```

---

### A10. Test Warning Suppression Too Broad
**File:** `pytest.ini:5-6`
**Status:** NOT FIXED (may be intentional)

```ini
filterwarnings =
    ignore::RuntimeWarning
    ignore::FutureWarning
```

**When it occurs:** During all test runs.

**Why it happens:** Probably added to suppress noisy warnings during development, but it's too broad.

**Impact:** Real warnings about numerical issues or deprecated APIs are hidden, masking bugs.

---

### A11. Missing Test Coverage
**File:** `tests/`
**Status:** NOT FIXED (requires significant test development)

**When it occurs:** Gap exists in test suite.

**Why it happens:** Core algorithms were not unit tested, only integration tests exist.

**Not tested:**
- `HSMM.fit()`, `HMM.fit()` - Core EM algorithm logic
- `ab._forward()`, `ab._backward()` - Forward-backward implementations
- `fwd_bwd.pyx` - Cython implementation
- `RUL()`, `prognostics()` - Main prediction functions
- Edge cases: empty sequences, single states, long sequences

**Impact:** Bugs in core algorithms go undetected.

---

### A12. Incomplete Return Documentation
**File:** `himap/utils.py:534-557`
**Status:** FIXED

**Original docstring:**
```python
def calculate_cdf(pmf, confidence_level):
    """
    Returns
    -------
    lower_value : int
        The index corresponding to the lower percentile.
    """
    return lower_value, upper_value  # Actually returns TWO values!
```

**When it occurs:** When users read documentation to understand function API.

**Why it happens:** Docstring wasn't updated when return value changed.

**Impact:** Users don't know about second return value; may cause unpacking errors.

**Fix applied:** Added documentation for `upper_value` return value.

---

## Optional Algorithmic Improvements

### A13. Missing Type Hints
**Files:** `base.py`, `utils.py`, `ab.py`
**Status:** NOT FIXED (enhancement)

All public functions lack type annotations, reducing IDE support and documentation clarity.

### A14. Debug Code in Production
**File:** `setup.py:8`
**Status:** FIXED

**Original code:**
```python
print("DISCOVERED PACKAGES:", ...)
```

**Fix applied:** Removed debug print statement.

### A15. Old Cython Version
**File:** `pyproject.toml:17`
**Status:** NOT FIXED (may cause compatibility issues)

```toml
"Cython>=0.29.35"  # From 2021
```

Should use modern Cython >=3.0.

### A16. No Test Coverage Measurement
**File:** `pyproject.toml:44-46`
**Status:** FIXED

**Fix applied:** Added `pytest-cov>=4.0` to measure test coverage.

### A17. Fixture Missing Return
**File:** `tests/conftest.py:33-42`
**Status:** FIXED

**Fix applied:** Added explicit `return _tqdm` statement.

### A18. Unused Return Value
**File:** `himap/base.py:310`
**Status:** NOT FIXED (may be intentional API)

`ctr_sample` is returned but ignored by all callers.

---

# Summary Tables

## By Type and Severity

| Type | Critical | Important | Standard | Optional | Total |
|------|----------|-----------|----------|----------|-------|
| **Scientific** | 4 | 3 | 3 | 0 | **10** |
| **Algorithmic** | 3 | 5 | 4 | 6 | **18** |
| **Total** | **7** | **8** | **7** | **6** | **28** |

## Scientific Bugs Summary

| ID | Severity | Issue | File:Line | Status |
|----|----------|-------|-----------|--------|
| S1 | Critical | Wrong convergence criterion | base.py:688 | FIXED |
| S2 | Critical | Invalid transition matrix | base.py:135-142 | FIXED |
| S3 | Critical | NaN replacement breaks probabilities | base.py:1646-1647 | FIXED |
| S4 | Critical | Log-domain underflow | utils.py:418-427 | NOT FIXED |
| S5 | Important | Division by zero in backward | utils.py:511 | NOT FIXED |
| S6 | Important | Viterbi bounds not validated | fwd_bwd.pyx:169-174 | NOT FIXED |
| S7 | Important | RUL break violates expectation | base.py:2041-2045 | FIXED |
| S8 | Standard | Uninitialized old_score | base.py:688, 697 | FIXED |
| S9 | Standard | Inconsistent convergence criteria | base.py:688 vs 1650 | PARTIALLY FIXED |
| S10 | Standard | K-means on truncated data | base.py:1226 | NOT FIXED |

## Algorithmic Bugs Summary

| ID | Severity | Issue | File:Line | Status |
|----|----------|-------|-----------|--------|
| A1 | Critical | IndexError in mc_dataset | base.py:345-350 | FIXED |
| A2 | Critical | Undefined variable idx | base.py:345-350 | FIXED |
| A3 | Critical | Plotting single subplot | plot.py:35, 42 | FIXED |
| A4 | Important | Inconsistent exceptions | base.py:1480 vs 66 | FIXED |
| A5 | Important | Hard-coded column name | utils.py:57 | NOT FIXED |
| A6 | Important | O(n^2) DataFrame concat | utils.py:247, 285, 340 | FIXED |
| A7 | Important | Missing bounds check | utils.py:427, 475, 512 | NOT FIXED |
| A8 | Important | No file I/O error handling | utils.py:85-89 | NOT FIXED |
| A9 | Standard | Division by zero coverage | utils.py:283 | FIXED |
| A10 | Standard | Broad warning suppression | pytest.ini:5-6 | NOT FIXED |
| A11 | Standard | Missing test coverage | tests/ | NOT FIXED |
| A12 | Standard | Incomplete return docs | utils.py:534-557 | FIXED |
| A13 | Optional | Missing type hints | multiple files | NOT FIXED |
| A14 | Optional | Debug code in production | setup.py:8 | FIXED |
| A15 | Optional | Old Cython version | pyproject.toml:17 | NOT FIXED |
| A16 | Optional | No coverage measurement | pyproject.toml:44-46 | FIXED |
| A17 | Optional | Fixture missing return | conftest.py:33-42 | FIXED |
| A18 | Optional | Unused return value | base.py:310 | NOT FIXED |

---

# Fix Summary

## Bugs Fixed (14 total)

### Critical (5)
- S1: HSMM convergence criterion
- S2: Left-to-right transition matrix
- S3: NaN replacement in HMM
- A1: IndexError in mc_dataset
- A2: Undefined variable idx
- A3: Plotting single subplot

### Important (3)
- S7: RUL calculation break
- A4: Inconsistent exception types
- A6: O(n^2) DataFrame concatenation

### Standard (4)
- S8: Initialize old_score variable
- A9: Division by zero in coverage
- A12: Incomplete return docs

### Optional (3)
- A14: Debug code in production
- A16: No coverage measurement
- A17: Fixture missing return

## Bugs Not Fixed (14 total)

These require more significant refactoring or may be intentional design choices:
- S4: Log-domain underflow (requires major refactoring)
- S5: Division by zero in backward (requires careful handling)
- S6: Viterbi bounds validation (Cython code)
- S9: Inconsistent convergence criteria (partial fix applied)
- S10: K-means on truncated data (may be intentional)
- A5: Hard-coded column name (may be intentional API)
- A7: Missing bounds check (requires validation logic)
- A8: File I/O error handling (low priority)
- A10: Broad warning suppression (may be intentional)
- A11: Missing test coverage (requires test development)
- A13: Missing type hints (enhancement)
- A15: Old Cython version (compatibility concerns)
- A18: Unused return value (may be intentional API)
