# SurvITE with Debiased Sequential Treatment Effects

## Overview

This repository investigates how to predict the effect of sequential cancer treatment sequences (chemotherapy, radiotherapy, immunotherapy, targeted therapy, etc.) on patient survival outcomes while correcting for **time-dependent confounding**.

### The Core Challenge

Clinicians choose each subsequent treatment based on:
- How the patient responded to previous treatments
- Current disease status
- Patient's evolving health condition

This introduces **time-dependent confounding**: treatment decisions are not independent of prior outcomes, biasing naive survival estimates.

### Our Approach

We extend **SurvITE** (Curth, Lee & van der Schaar, 2021) by integrating debiasing techniques from **Bica et al. (2019)** to:

1. **Estimate propensity scores** at each treatment decision point
2. **Apply doubly-robust estimation** to correct for confounding
3. **Predict unbiased survival outcomes** for sequential treatment regimens
4. **Validate** that the model correctly handles time-dependent bias

---

## Key References

- **Curth, A., Lee, S., & van der Schaar, M.** (2021). "SurvITE: Learning Heterogeneous Treatment Effects from Time-Event Data." *NeurIPS 2021*.
- **Bica, I., Jordon, J., & van der Schaar, M.** (2019). "Estimating Counterfactual Treatment Outcomes over Time through Adapting Weight-Adjustment for Confounding." *NeurIPS 2019*.
- **Kunzel, S. R., Sekhon, J. S., Tejada, J. F., & Zhao, Q.** (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning." *PNAS*.

---

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── LICENSE                           # Project license
│
├── docs/
│   ├── METHODS.md                   # Detailed methods & pseudocode
│   ├── LITERATURE_REVIEW.md         # Background & related work
│   └── EXPERIMENT_PROTOCOL.md       # Experimental design
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py               # Data loading utilities
│   │   ├── processors.py            # Data preprocessing & feature engineering
│   │   └── generators.py            # Synthetic data generation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── survite_base.py          # Original SurvITE implementation
│   │   ├── survite_debiased.py      # Debiased SurvITE variant
│   │   ├── propensity_scorer.py     # Propensity score estimation
│   │   └── doubly_robust.py         # Doubly-robust estimation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Survival metrics (C-index, calibration)
│   │   ├── bias_assessment.py       # Bias detection & quantification
│   │   └── visualization.py         # Plotting utilities
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       └── logging.py               # Logging utilities
│
├── experiments/
│   ├── synthetic_data_experiment.py    # Controlled bias experiments
│   ├── real_data_experiment.py         # SEER/clinical data experiments
│   └── ablation_study.py               # Component ablation studies
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA & dataset analysis
│   ├── 02_baseline_survite.ipynb       # SurvITE baseline implementation
│   ├── 03_debiasing_integration.ipynb  # Integrate debiasing methods
│   └── 04_results_analysis.ipynb       # Results interpretation
│
└── tests/
    ├── test_data_loaders.py
    ├── test_models.py
    └── test_evaluation.py
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- scikit-learn, pandas, numpy

### Installation

```bash
git clone https://github.com/mehdigalore/survite-debiased-treatment-sequences.git
cd survite-debiased-treatment-sequences

pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### 1. Load and Explore Data

```python
from src.data.loaders import load_cancer_dataset
from src.data.processors import preprocess_treatment_sequences

# Load dataset (SEER or synthetic)
data = load_cancer_dataset(source='synthetic', n_patients=1000)

# Preprocess: create time-indexed treatment sequences
processed_data = preprocess_treatment_sequences(data)
```

### 2. Fit Debiased SurvITE Model

```python
from src.models.survite_debiased import DebiasedSurvITE

model = DebiasedSurvITE(
    representation_dim=64,
    debiasing_method='doubly_robust'
)

model.fit(processed_data['train'])
predictions = model.predict(processed_data['test'])
```

### 3. Evaluate Predictions & Bias

```python
from src.evaluation.metrics import c_index, calibration_error
from src.evaluation.bias_assessment import estimate_treatment_bias

# Prediction accuracy
c_idx = c_index(predictions, processed_data['test']['time'], processed_data['test']['event'])

# Time-dependent confounding assessment
bias = estimate_treatment_bias(model, processed_data['test'])
```

---

## Experiments

### Synthetic Data Experiments

Validate unbiasedness with known ground truth:

```bash
python experiments/synthetic_data_experiment.py --n_samples 5000 --confounding_strength high
```

### Real Data Experiments

Test on SEER cancer registry data:

```bash
python experiments/real_data_experiment.py --dataset seer --cohort lung_cancer
```

### Ablation Studies

Test individual components:

```bash
python experiments/ablation_study.py --component propensity_scorer
```

---

## Results

*To be populated with experimental findings*

- Prediction accuracy (C-index comparisons)
- Bias reduction quantification
- Sensitivity analyses
- Subgroup performance

---

## Contributing

Contributions welcome! Please:
1. Create a feature branch (`git checkout -b feature/your-feature`)
2. Commit changes (`git commit -am 'Add feature'`)
3. Push to branch (`git push origin feature/your-feature`)
4. Submit a pull request

---

## License

MIT License - see LICENSE file

---

## Contact

For questions or collaboration, please contact: mehdigalore

---

## Citation

If you use this code, please cite:

```bibtex
@software{galore2025survite_debiased,
  author = {Galore, Mehdi},
  title = {SurvITE with Debiased Sequential Treatment Effects},
  year = {2025},
  url = {https://github.com/mehdigalore/survite-debiased-treatment-sequences}
}
```
