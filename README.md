# Q-DisC: Multi-Stage Quantum Machine Learning for Alzheimer's & MCI Classification
### Challenge 02 · Quantum Machine Learning | Healthcare · Team DisC

A hybrid quantum-classical machine learning pipeline and interactive clinical dashboard that stratifies subjects across the electrophysiological continuum of neurodegeneration:
1. **Healthy Controls (HC)**: Normal cognitive functioning.
2. **Mild Cognitive Impairment (MCI)**: Early transitional stage with emerging electrophysiological slowing; the critical therapeutic window.
3. **Alzheimer's Disease (AD)**: Clinically diagnosed neurodegeneration with hallmark cortical disconnection.

The quantum models (**Quantum Kernel Support Vector Classifier / QSVC** and **Variational Quantum Classifier / VQC**) are benchmarked against classical baselines (RBF SVM and Random Forest) on the exact same train/test split.

---

## Table of Contents

- [Key Features](#key-features)
- [Background & Clinical Context](#background--clinical-context)
- [Multi-Stage Classification System](#multi-stage-classification-system)
- [Quantum Pipeline Architecture](#quantum-pipeline-architecture)
- [Installation](#installation)
- [Running the Project](#running-the-project)
  - [1. Interactive Web Application](#1-interactive-web-application)
  - [2. Jupyter Research Notebook](#2-jupyter-research-notebook)
- [Model Benchmarking](#model-benchmarking)
- [Project Structure](#project-structure)
- [References](#references)

---

## Key Features

- **3-Stage Clinical Stratification**: HC vs. MCI vs. AD detection targeting the critical prodromal intervention window.
- **Dual Quantum Modeling**:
  - **Quantum Kernel Support Vector Classifier (QSVC)** with `FidelityQuantumKernel` using the 4-qubit `ZZFeatureMap` (73–84% 3-class test accuracy).
  - **Variational Quantum Classifier (VQC)** with parameterized `EfficientSU2` ansatz trained via classical `COBYLA` optimization.
- **Interactive Clinical Decision Support Dashboard ([app.py](file:///c:/Users/Razi/Desktop/qff%20hack/qff_hackday/app.py))**:
  - Clean, professional white clinical medical aesthetic with custom **Team DisC** SVG branding.
  - Live patient EEG spectral power sliders with clinical case presets (Healthy, MCI, Severe AD, Custom).
  - Real-time quantum vs. classical prediction cards with multi-class confidence probability distributions.
  - Live bound quantum phase rotation angles table $(\theta_0, \theta_1, \theta_2, \theta_3 \in [0, \pi])$.
  - Dynamic electrophysiological slowing ratio $((\theta + \delta)/(\alpha + \beta))$ calculator.
- **Fully Executed Research Notebook ([qml_alzheimers_vqc.ipynb](file:///c:/Users/Razi/Desktop/qff%20hack/qff_hackday/qml_alzheimers_vqc.ipynb))**:
  - Complete 22-cell end-to-end execution with $3 \times 3$ confusion matrices, loss convergence curve, and comparative metric charts.

---

## Background & Clinical Context

Alzheimer's disease causes progressive synaptic dysfunction and cortical disconnection, appearing in EEG as **"EEG slowing"**:

| Signature | Description | Clinical Significance |
|---|---|---|
| **Elevated Low-Frequency Power** | Surge in **Delta (0.5–4 Hz)** and **Theta (4–8 Hz)** activity | Sign of cortical de-afferentation and synaptic failure |
| **Suppressed High-Frequency Power** | Decrease in **Alpha (8–13 Hz)** and **Beta (13–30 Hz)** activity | Loss of synchronized pyramidal neuron firing |
| **Loss of Signal Complexity** | Reduced spectral entropy and phase coherence | Loss of functional cortical connectivity |

---

## Multi-Stage Classification System

| Stage | Clinical Condition | Electrophysiological Hallmark | Clinical Action |
|---|---|---|---|
| **Stage 0: HC** | Healthy Control | High Alpha & Beta, minimal Delta/Theta | Routine annual monitoring |
| **Stage 1: MCI** | Mild Cognitive Impairment | Moderate Alpha drop, emerging Theta surge | **Active Triage & Early Therapy Window** |
| **Stage 2: AD** | Alzheimer's Disease | Severe Delta/Theta surge, depleted Alpha/Beta | Full clinical intervention |

---

## Quantum Pipeline Architecture

1. **Dimensionality Reduction & Angle Encoding:**
   - 16-channel EEG spectral powers (4 regions $\times$ 4 bands) are standardized and projected to $N = 4$ principal components via PCA (capturing $>86\%$ cumulative variance).
   - Features are rescaled to $[0, \pi]$ using `MinMaxScaler` to serve directly as quantum gate rotation angles.
2. **Entangling Feature Map (`ZZFeatureMap`):**
   - 4 qubits with linear entanglement.
   - Encodes non-linear pairwise spectral interactions into a $2^4 = 16$-dimensional Hilbert state space via $e^{i(\pi - x_i)(\pi - x_j)Z_i Z_j}$ gates.
3. **Quantum Classifiers:**
   - **QSVC**: Computes quantum state fidelity kernels $K(x, z) = |\langle 0 | U^\dagger(z)U(x) | 0 \rangle|^2$ and constructs optimal hyperplanes in Hilbert space.
   - **VQC**: Parameterized `EfficientSU2` ansatz optimized by COBYLA to minimize cross-entropy loss.

---

## Installation

Recommended Python version: **3.10 – 3.12**.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn \
            qiskit qiskit-aer qiskit-algorithms qiskit-machine-learning \
            streamlit jupyter
```

---

## Running the Project

### 1. Interactive Web Application

Launch the Streamlit clinical dashboard:

```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your browser.

### 2. Jupyter Research Notebook

To run or inspect the end-to-end research experiments:

```bash
jupyter notebook qml_alzheimers_vqc.ipynb
```

---

## Model Benchmarking

Benchmarked on the stratified 3-stage validation dataset (HC vs. MCI vs. AD):

| Model | Classification Paradigm | 3-Class Test Accuracy | Macro Recall | Macro F1-Score | Training Latency |
|---|---|---|---|---|---|
| **Classical SVM (RBF)** | Convex Margin (RBF Kernel) | 100.00% | 100.00% | 100.00% | 0.006s |
| **Classical Random Forest** | Ensemble Trees | 100.00% | 100.00% | 100.00% | 0.114s |
| **Quantum Kernel (QSVC)** | Hilbert Space Fidelity Kernel | **73.33% – 83.33%** | **73.33% – 83.33%** | **73.47% – 83.47%** | 16.47s |
| **Variational Quantum (VQC)** | Parameterized Ansatz (COBYLA) | 53.33% | 53.33% | 51.80% | 18.94s |

---

## Project Structure

```
.
├── app.py                      # Interactive Streamlit Web App (Q-DisC)
├── qml_alzheimers_vqc.ipynb    # Complete research notebook with 3-class VQC & QSVC
├── .streamlit/
│   └── config.toml             # Streamlit light theme styling configuration
├── README.md                   # Project documentation and clinical overview
└── .gitignore                  # Git ignore rules
```

---

## References

- Jeong, J. (2004). EEG dynamics in patients with Alzheimer's disease. *Clinical Neurophysiology*.
- Dauwels, J., Vialatte, F., & Cichocki, A. (2010). Diagnosis of Alzheimer's disease from EEG signals: where are we standing? *Current Alzheimer Research*.
- Babiloni, C. et al. (2006). Sources of cortical rhythms in mild cognitive impairment and Alzheimer's disease. *NeuroImage*.
- Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*.
- [Qiskit Machine Learning Documentation](https://qiskit-community.github.io/qiskit-machine-learning/)
