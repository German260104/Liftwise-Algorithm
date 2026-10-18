# Liftwise — Automated Workout Interval Detection

**Python · Time-Series Analysis · Signal Processing · Ensemble Methods**

University team software project that detects workout intervals in multivariate time-series data collected from wearable sensors. It combines three complementary detection approaches within a modular framework for running methods, aggregating predictions, and evaluating results against manually annotated data.

## Project overview

Identifying the beginning and end of exercise sets in continuous sensor recordings is challenging: movements vary between exercises, and transitions between activity and rest are not always clear. Liftwise explores multiple ways to detect these intervals automatically and uses an ensemble to combine their predictions.

The project focuses on **workout interval detection**, not on identifying the name of every exercise.

## Detection methods

| Method | Main idea |
| --- | --- |
| **Greedy Gaussian Segmentation (GGS)** | Detects changes in the statistical structure of multivariate sensor signals to propose segment boundaries. |
| **FLUSS with autocorrelation** | Uses time-series segmentation and recurring signal patterns to identify candidate activity intervals. |
| **FFT with autocorrelation** | Examines frequency content and periodicity to detect repeated movement patterns. |

The three methods produce predictions that can be evaluated individually and combined by the framework.

## Framework and ensemble voting

The benchmark framework provides a common workflow for comparing the methods:

1. **Run methods** — `run_methods.py` executes configured detection methods on sensor CSV files.
2. **Combine predictions** — `run_voting.py` aggregates outputs using configurable percentage-based or weighted voting.
3. **Evaluate** — `run_benchmark.py` compares predicted labels with manually annotated ground truth and generates a results report.
4. **Inspect** — `run_upload.py` supports uploading predictions to Label Studio for visual review.

```text
Wearable sensor CSV files
          |
          v
   Detection methods
    /      |      \
   GGS   FLUSS   FFT + AC
    \      |      /
          v
     Ensemble voting
          |
          v
   Predicted intervals
          |
          v
 Ground-truth evaluation
```

## Evaluation result

**91.4% recall — ensemble voting system**

In the project's evaluation, the **combined voting system** achieved **91.4% recall** for workout interval detection. This figure describes the ensemble result, **not** the performance of any individual detection method. Recall measures how much of the annotated exercise activity was successfully detected; it does not, by itself, describe false-positive performance.

## Repository structure

```text
Liftwise-Algorithm/
├── README.md                         # Project overview (this file)
├── .gitignore
├── Framework/                       # Execution, voting, benchmarking and upload scripts
│   ├── README.md                     # Detailed installation and usage guide
│   ├── requirements.txt
│   ├── run_methods.py
│   ├── run_voting.py
│   ├── run_benchmark.py
│   └── run_upload.py
└── FertigeAlgorithmenFürFramework/   # Detection methods and GGS dependencies
    ├── GreedyGaussianSegmentation.py
    ├── fluss.py
    └── method_fft_ac_hybrid.py
```

The tree highlights key files rather than listing every file in the repository.

## Getting started

See the **[Framework installation and usage guide](Framework/README.md)** for setup, input/output conventions, execution steps, benchmarking, voting configuration, and Label Studio integration.

**Important:** The public repository excludes original sensor datasets, manually labeled datasets, local API credentials, and the trained GGS `.joblib` artifact. The GGS method requires its trained model to be supplied locally before it can run in the documented configuration. Only use datasets and model files that you are authorized to access and share.

## Project context

Developed as a **university team software project**. The repository brings together the detection methods and the shared framework used to evaluate and combine them.
