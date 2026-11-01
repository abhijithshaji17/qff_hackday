# Quantum Machine Learning for Alzheimer's Disease Classification

**Challenge 02 · Quantum Machine Learning | Healthcare**

A hybrid quantum-classical pipeline that uses a **Variational Quantum Classifier (VQC)** to distinguish **Alzheimer's Disease (AD)** patients from **Healthy Controls (HC)** using EEG spectral biomarkers. The quantum model is benchmarked against classical baselines (RBF SVM and Random Forest) on the same train/test split.

---

## Table of Contents

- [Background](#background)
- [Approach](#approach)
- [Pipeline Overview](#pipeline-overview)
- [Quantum Circuit Design](#quantum-circuit-design)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Limitations & Notes](#limitations--notes)
- [Project Structure](#project-structure)
- [References](#references)

---

## Background

Alzheimer's disease causes progressive synaptic dysfunction and cortical disconnection, which appears in EEG as **"EEG slowing"**:

| Signature | Description |
|---|---|
| **Elevated low-frequency power** | Increased Delta (0.5–4 Hz) and Theta (4–8 Hz) activity |
| **Suppressed high-frequency power** | Decreased Alpha (8–13 Hz) and Beta (13–30 Hz) activity, especially over temporal and parietal regions |
| **Loss of complexity** | Lower spectral entropy and reduced cross-channel phase coherence |

This project asks whether a small quantum classifier can pick up these spectral patterns, and how it compares to classical models.

## Approach

1. Compress multi-channel EEG spectral features to **N = 4 qubits** with PCA.
2. Encode them with an entangling **ZZ-Feature Map**, mapping feature interactions into a 2⁴ = 16-dimensional quantum state space.
3. Train a parameterized **EfficientSU2** ansatz with a classical optimizer (COBYLA by default; SPSA is also imported) to minimize cross-entropy loss.
4. Compare against **RBF SVM** and **Random Forest** using accuracy, sensitivity (recall), precision, F1, and training time.

Sensitivity is emphasized because in a clinical setting a false negative (a missed AD patient) is the costliest error.

## Pipeline Overview

| Step | Section | What happens |
|---|---|---|
| 1 | Imports & setup | Loads libraries, fixes random seeds (42) |
| 2 | Dataset loading | Loads a local CSV or generates a synthetic EEG dataset |
| 3 | EDA | Bar chart verifying EEG slowing (band power, AD vs. HC) |
| 4 | Preprocessing | `StandardScaler` → PCA (4 components) → `MinMaxScaler` to [0, π] → stratified 80/20 split |
| 5 | Classical baselines | Trains and evaluates SVC (RBF) and Random Forest |
| 6 | Circuit architecture | Builds and draws the feature map and ansatz |
| 7 | VQC training | Trains the VQC with COBYLA (60 iterations) and logs loss |
| 8 | Convergence | Plots the loss curve |
| 9 | Evaluation | Classification report and side-by-side confusion matrices (SVC vs. VQC) |
| 10 | Benchmarking | Summary table and bar chart of all three models |
| 11 | Discussion | Key takeaways and NISQ considerations |

## Quantum Circuit Design

**Feature map: `ZZFeatureMap`**
- 4 qubits, `reps=1`, linear entanglement
- Uses Hadamard gates, single-qubit R_z rotations, and two-qubit ZZ interactions
- Inputs are scaled to [0, π] so they act directly as rotation angles

**Ansatz: `EfficientSU2`**
- 4 qubits, `reps=2`, linear entanglement
- Parameterized R_y and R_z rotations with CNOT entanglement layers
- 24 trainable weights (2 rotations × 4 qubits × 3 layers)

**Training**
- Sampler: `StatevectorSampler` (exact, noiseless probabilities)
- Optimizer: `COBYLA(maxiter=60)`
- Loss: cross-entropy, tracked per iteration through a callback

## Dataset

The notebook supports two modes, toggled with `USE_LOCAL_CSV`.

### Option A: Synthetic data (default)

`USE_LOCAL_CSV = False` generates a simulated dataset:
- 200 subjects (100 HC, 100 AD)
- 16 features: 4 regions (Frontal, Temporal, Parietal, Occipital) × 4 bands (Delta, Theta, Alpha, Beta)
- Gaussian features (σ = 0.45) with class means chosen to reflect EEG slowing:

| Band | HC mean | AD mean |
|---|---|---|
| Delta | 1.1 | 2.9 |
| Theta | 1.4 | 3.1 |
| Alpha | 3.8 | 1.7 |
| Beta | 2.6 | 1.3 |

### Option B: Real EEG data

Set `USE_LOCAL_CSV = True` and point `CSV_PATH` to your file. Suggested sources:
- Kaggle EEG Alzheimer's dataset
- OpenNeuro [`ds004504`](https://openneuro.org/datasets/ds004504)

The loader looks for a target column named `class`, `diagnosis`, `label`, or `Group` (values `HC`/`AD` or `0`/`1`) and drops `subject_id`/`id` columns if present. Remaining columns are treated as numeric features. **Note:** the EDA cell (Section 3) expects the synthetic column names (`Frontal_Delta`, etc.) and a `Diagnosis` column, so it needs adapting for a custom CSV.

## Installation

Python 3.9+ is recommended.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn \
            qiskit qiskit-aer qiskit-algorithms qiskit-machine-learning \
            jupyter
```

The notebook uses `StatevectorSampler` (Qiskit primitives V2), so use a recent Qiskit (1.x) together with a compatible `qiskit-machine-learning` release (0.8 or later). If you hit import or primitive-version errors, check version compatibility first.

## Usage

```bash
jupyter notebook
```

Open the notebook and run all cells top to bottom. With the default settings (synthetic data, 4 qubits, 60 COBYLA iterations) it runs on a laptop CPU.

## Outputs

- Printed PCA explained variance per component and cumulative total
- Classical baseline metrics (accuracy, recall, F1, training time)
- Circuit diagrams for the feature map and ansatz
- VQC loss curve over optimization iterations
- Classification report and confusion matrices (SVC vs. VQC)
- Benchmark table and grouped bar chart comparing SVC, Random Forest, and VQC

## Configuration

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `USE_LOCAL_CSV` | Section 2 | `False` | Switch between synthetic and real data |
| `CSV_PATH` | Section 2 | `eeg_alzheimers_dataset.csv` | Path to local data |
| `n_qubits` | Section 4 | `4` | Also the number of PCA components |
| `reps` (feature map) | Section 6 | `1` | Depth of the ZZ encoding |
| `reps` (ansatz) | Section 6 | `2` | Depth of the variational circuit |
| `max_iterations` | Section 7 | `60` | COBYLA iterations; increase for better convergence |
| `test_size` | Section 4 | `0.20` | Stratified split |
| Random seed | Section 1 | `42` | Set via `algorithm_globals` and NumPy |

Possible experiments: swap COBYLA for SPSA, try `RealAmplitudes` or `ZFeatureMap` (already imported), vary circuit depth, or use `AerSimulator` with a noise model.

## Limitations & Notes

- **Synthetic data is easy to separate.** The default dataset has well-separated class means, so high accuracy from all models is expected and should not be read as clinical performance or evidence of quantum advantage.
- **Possible data leakage.** The scaler, PCA, and MinMax scaler are fit on the full dataset *before* the train/test split. For rigorous evaluation, fit these on the training set only (e.g., with a scikit-learn `Pipeline`) and apply them to the test set.
- **Small test set.** With 200 subjects and an 80/20 split, the test set has 40 samples, so metrics are noisy. Consider repeated runs or cross-validation.
- **Simulation only.** Training uses an ideal statevector simulator with no hardware noise. Classical models train in milliseconds, while the VQC is much slower on a simulator.
- **Not a medical device.** This is a research and educational project, not a diagnostic tool.

## Project Structure

```
.
├── qml_alzheimers_vqc.ipynb       # Main notebook (this pipeline)
├── eeg_alzheimers_dataset.csv     # Optional: local dataset (not included)
└── README.md
```

## References

- Jeong, J. (2004). EEG dynamics in patients with Alzheimer's disease. *Clinical Neurophysiology*.
- Dauwels, J., Vialatte, F., & Cichocki, A. (2010). Diagnosis of Alzheimer's disease from EEG signals: where are we standing? *Current Alzheimer Research*.
- Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*.
- [Qiskit Machine Learning documentation](https://qiskit-community.github.io/qiskit-machine-learning/)
