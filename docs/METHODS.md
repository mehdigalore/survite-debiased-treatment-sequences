# Detailed Methods & Pseudocode

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [SurvITE Baseline](#survite-baseline)
3. [Time-Dependent Confounding](#time-dependent-confounding)
4. [Debiasing Approach](#debiasing-approach)
5. [Algorithm Pseudocode](#algorithm-pseudocode)
6. [Implementation Details](#implementation-details)

---

## Problem Formulation

### Notation

- **T**: Continuous survival time (follow-up duration until event)
- **E**: Event indicator (1 = death/censoring event, 0 = censored)
- **A_t**: Treatment received at time step t (e.g., chemo=1, radio=2, immuno=3)
- **X_t**: Patient covariates at time t (age, disease markers, etc.)
- **H_t**: History up to time t: {X_0:t, A_0:t-1}
- **S(t|A_{0:T})**: Survival probability at time t given full treatment sequence A_{0:T}

### Objective

Estimate the **counterfactual survival curve** for a given treatment sequence **a = (a_0, a_1, ..., a_K)**:

```
S(t | do(A_0:K = a)) = E[S(t) | hypothetically following sequence a]
```

Without bias from treatment assignment confounding at each step.

---

## SurvITE Baseline

### Architecture (Curth et al., 2021)

SurvITE combines representation learning + survival modeling:

```
Input: (X_0:T, A_0:T, T, E)
  ↓
[1] Representation Layer
    - LSTM/GRU processes temporal covariates + treatments
    - Learns time-invariant representation φ(H_T)
  ↓
[2] Treatment Effect Layer
    - For each treatment sequence, predicts heterogeneous effect
    - Output: layer-specific treatment coefficients β_a(φ)
  ↓
[3] Survival Head
    - Parametric survival model (e.g., Weibull, log-normal)
    - P(T > t | φ, a) = S(t | φ, a)
  ↓
Output: S(t | A_0:K = a)
```

### Loss Function (without debiasing)

```
L_SurvITE = E[-(E·log f(T|φ,a) + (1-E)·log S(T|φ,a))]
```

Where:
- f(T|φ,a) = density of survival distribution
- S(T|φ,a) = survival function

**Issue**: This assumes treatments are randomly assigned, but they're not—clinicians adapt to patient response.

---

## Time-Dependent Confounding

### The Confounding Structure

```
Past Response L_{t-1}  ← affects disease progression
        ↓
        → influences treatment choice A_t (clinician decision)
        ↓
L_{t-1} also affects future survival independently
        ↓
Result: A_t is correlated with future outcomes through L_{t-1}
```

### Example: Cancer Treatment Sequence

```
Time 0:     Patient presents with stage IV cancer
            Clinician chooses: Chemotherapy

Time 6 mo:  Patient's tumor shrinks (good response to chemo)
            Disease markers improve → L_1 = "good"
            Clinician decides: Add immunotherapy
            (Better prognosis patients get combo therapy)

Time 12 mo: Chemo+Immuno patients survive longer
            BUT: This is partially due to "good responder" status
            NOT purely due to immunotherapy sequence
```

### Mathematical Formulation

Without debiasing, we estimate:
```
Estimated Effect = Direct Effect + Confounding Bias
```

The bias arises from:
```
Bias = Cov(A_t, L_t | H_{t-1})
```

Where L_t (disease state/response) is unobserved or incompletely observed.

---

## Debiasing Approach

### Strategy: Inverse Probability Weighting (IPW) + Doubly Robust Estimation

Based on Bica et al. (2019), we use:

#### Step 1: Propensity Score Estimation

At each time step t, estimate the probability of receiving treatment a_t given history:

```
e_t(a_t | H_t) = P(A_t = a_t | H_t)
```

Implementation:
- Train a classifier (logistic regression or neural network)
- Input: H_t = {X_0:t, A_0:t-1, L_0:t-1}
- Output: propensity scores for each treatment option

#### Step 2: Inverse Probability of Treatment Weighting (IPTW)

Weight each patient's contribution by inverse probability:

```
w_i = ∏_{t=0}^{K} 1 / e_t(a_{i,t} | H_{i,t})
```

This creates a pseudo-population where treatment is independent of history.

#### Step 3: Doubly Robust Estimation

Combine:
- **Outcome Regression**: μ(t, a, φ) = E[S(t) | a, φ]
- **Propensity Scoring**: e_t(a_t | H_t)

Doubly robust estimator:

```
DR(t, a) = μ(t, a, φ*) 
           + ∑_{t=0}^{K} w_t · [Y_t - μ(t, A_t, φ*)]
```

Benefits:
- Consistent if EITHER outcome model OR propensity model is correct
- Reduces variance vs. pure IPTW
- Robust to model misspecification

---

## Algorithm Pseudocode

### Main Training Loop

```python
Algorithm: DebiasedSurvITE Training

Input:
  - Data: {(X_i, A_i, T_i, E_i, L_i)}_{i=1}^{N}
  - Treatment sequence dimension K
  - Representation dimension d_rep
  - Number of epochs N_epochs

Output:
  - Trained model θ = (φ_θ, μ_θ, e_θ)

1. Split data: train_rep, train_debiasing, test

2. FOR epoch = 1 to N_epochs:

     [Phase 1: Train Representation]
     FOR each batch in train_rep:
       - Encode trajectory: φ = LSTM(X_0:T, A_0:T)
       - Predict survival: Ŝ(t|φ) using baseline SurvITE
       - Compute loss: L_surv
       - Update φ_θ
     END

     [Phase 2: Train Propensity Scores]
     FOR each batch in train_debiasing:
       FOR t = 0 to K-1:
         - Input: H_t = {X_0:t, A_0:t-1, L_0:t-1}
         - Predict: ê_t(a_t | H_t) using neural classifier
         - Compute loss: cross-entropy
         - Update e_θ
       END
     END

     [Phase 3: Train Outcome Regression + Debiasing]
     FOR each batch in train_debiasing:
       - Compute inverse weights: w_i = ∏_t 1/e_t(a_{i,t}|H_{i,t})
       - Predict outcomes: μ̂ = outcome_model(φ, a)
       - Compute doubly-robust loss:
         L_DR = w·[E_i·log f(T_i) + (1-E_i)·log S(T_i)] + regularization
       - Update μ_θ
     END

3. RETURN θ = (φ_θ, μ_θ, e_θ)
```

### Inference

```python
Algorithm: DebiasedSurvITE Prediction

Input:
  - Test patient: (X_test, A_test, H_test)
  - Trained model θ
  - Target treatment sequence: a*

Output:
  - Predicted survival curve: Ŝ(t | a*)

1. Encode patient history: φ* = LSTM_θ(X_test, A_test)

2. IF predicting observed sequence A_test:
   - Apply IPTW correction
   - w = ∏_t 1/e_t(a_{test,t} | H_{test,t})
   - Ŝ(t) = outcome_regression_θ(t, φ*, A_test, w)

3. ELSE (counterfactual sequence a*):
   - Assume sequential ignorability (unconfoundedness given H_t)
   - Predict: Ŝ(t) = outcome_regression_θ(t, φ*, a*)
   - (No reweighting needed for counterfactual)

4. RETURN Ŝ(t)
```

---

## Implementation Details

### Representation Layer (φ)

**Architecture**:
```
Input: (X_0:T, A_0:T)
  ↓
[Embedding]: A_0:T → d_emb dimensional embeddings
  ↓
[LSTM]: Processes (X_t, emb(A_t)) sequence
  ↓
[Attention]: Optional attention over time steps
  ↓
[Output]: φ ∈ R^d_rep (final hidden state or attention-weighted sum)
```

**Hyperparameters**:
- `lstm_hidden_dim`: 64-128
- `representation_dim`: 32-128
- `embedding_dim`: 8-16
- `dropout`: 0.2-0.5

### Propensity Score Estimation (e_t)

**Architecture**:
```
Input: H_t = [X_0:t, A_0:t-1, L_0:t-1]
  ↓
[Dense]: 2-3 hidden layers (64-128 units each)
  ↓
[Softmax]: Output probabilities for each treatment option
  ↓
Output: ê_t(a | H_t) ∈ [0, 1]^|Actions|
```

**Training**:
- Loss: Cross-entropy with label A_t
- Regularization: L2 to prevent overconfident predictions
- Separate model per time step OR shared architecture with time embedding

### Outcome Regression (μ)

**Architecture**:
```
Input: (φ, a) or (φ, a, w) with weights
  ↓
[Dense]: 2 hidden layers (128 units each)
  ↓
[Treatment Branch]: Learns heterogeneous coefficients for sequence a
  ↓
[Survival Head]: Parametric model (Weibull, Lognormal)
  ↓
Output: S(t | φ, a) and f(t | φ, a)
```

**Loss with IPTW**:
```
L = E[w · (E·log f(T|φ,a) + (1-E)·log S(T|φ,a))]
```

---

## Key Assumptions

1. **Sequential Ignorability (Unconfoundedness)**: After observing H_t, future treatment is independent of unobserved confounders
   ```
   A_t ⊥⊥ U | H_t  for all t
   ```

2. **Overlap/Positivity**: For every patient history, there's nonzero probability of each treatment
   ```
   0 < e_t(a_t | H_t) < 1  for all a_t, H_t
   ```

3. **Consistency**: Treatment received affects outcomes only through its definition
   ```
   T(A=a) = T_obs  if A=a actually observed
   ```

4. **No Interference**: One patient's treatment doesn't affect another's

---

## References

- Curth, A., Lee, S., & van der Schaar, M. (2021). SurvITE. NeurIPS 2021.
- Bica, I., Jordon, J., & van der Schaar, M. (2019). Estimating Counterfactual Treatment Outcomes over Time. NeurIPS 2019.
- Robins, J. M. (1986). A new approach to causal inference in mortality studies. Mathematical Modelling, 7(5-8), 1393-1512.
- Kennedy, E. H. (2016). Semiparametric theory and empirical processes via stratified sampling. arXiv preprint arXiv:1601.04635.
