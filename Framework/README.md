# Exercise Detection Benchmark Framework

Installation, usage, benchmarking, and integration guide

This framework runs and benchmarks different algorithms for automatically detecting and segmenting workout activity from sensor data. It can also upload predictions to Label Studio for visual inspection and combine the outputs of multiple algorithms through an ensemble voting system.

## Contents

1. [Framework overview](#1-framework-overview)
2. [Directory structure](#2-directory-structure)
3. [Installation](#3-installation)
4. [Typical workflow](#4-typical-workflow)
5. [Evaluation logic and metrics](#5-evaluation-logic-and-metrics)
6. [Visual inspection in Label Studio](#6-visual-inspection-in-label-studio)
7. [Ensemble voting](#7-ensemble-voting)
8. [Integrating an existing algorithm](#8-integrating-an-existing-algorithm)
9. [Common issues and best practices](#9-common-issues-and-best-practices)

## 1. Framework overview

The framework is designed to compare any number of detection methods in a modular, automated workflow:

1. **Run methods (`run_methods.py`):** Each configured algorithm reads raw sensor CSV files from `InputCSV/` and writes predictions (`exercising` or `pause`) to its own output directory.
2. **Combine predictions (`run_voting.py`):** Aggregate multiple methods using a configurable quorum or weighted voting.
3. **Benchmark (`run_benchmark.py`):** Automatically discover method output directories, compare each row with manually annotated ground truth from `manualCSV/`, compute evaluation metrics, and save the results to `results.txt`.
4. **Upload results (`run_upload.py`):** Upload predictions to Label Studio for visual inspection.

## 2. Directory structure

| File or directory | Purpose |
| --- | --- |
| `InputCSV/` | Raw sensor data; the scripts do not modify these files. |
| `manualCSV/` | Manually annotated ground-truth data for validation. |
| `run_methods.py` | Runs the configured detection methods. |
| `run_voting.py` | Combines method predictions into an ensemble. |
| `run_benchmark.py` | Evaluates predictions and generates `results.txt`. |
| `run_upload.py` | Starts Label Studio and uploads predictions for visual inspection. |
| `results.txt` | Automatically generated evaluation report and overall ranking. |
| `method_template.py.txt` | Template for implementing new detection methods.
| `requirements.txt` | Required Python packages. |

**Public repository note:** The original sensor data, manual labels, and `KEY.txt` are intentionally excluded from the public repository. Provide your own authorized CSV files and Label Studio credentials when using the framework.

## 3. Installation

From the `Framework/` directory, install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

On Windows, you can alternatively use:

```bash
py -m pip install -r requirements.txt
```

## 4. Typical workflow

### Step 1 — Provide sensor data

Place the original, unmodified sensor CSV files in `InputCSV/`.

### Step 2 — Run detection methods

Place the desired method scripts in the `Framework/` directory and register them in `run_methods.py`.

**GGS dependency:** Greedy Gaussian Segmentation also requires a `ggs_dependencies/` directory containing `ggs_exercise_classifier_metadata.json` and `ggs_exercise_classifier.joblib`. The trained `.joblib` model is excluded from the public repository; GGS cannot run in the documented configuration until that dependency is supplied locally.

Run:

```bash
python run_methods.py
```

Each method creates its own output directory with a name ending in `_output`, provided it follows the framework's method interface.

### Step 3 — Optionally combine predictions

To create an ensemble prediction from multiple methods, run:

```bash
python run_voting.py
```

This creates `voting_output/`.

### Step 4 — Benchmark predictions

Place the manually labeled reference CSV files in `manualCSV/`. Their filenames must exactly match the corresponding files in `InputCSV/`.

Run:

```bash
python run_benchmark.py
```

### Step 5 — Review results

Open the generated `results.txt` file to inspect the overall ranking and detailed metrics for each dataset.

## 5. Evaluation logic and metrics

- **Positive class:** `exercising` (active workout activity).
- **Negative class:** `pause` (rest between sets or inactivity).
- **Pooling:** True positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) are summed across all rows in all files. Longer recordings therefore have proportionally more influence than shorter ones.
- **Primary ranking metric:** F2-score, followed by recall. The F2-score places greater emphasis on recall than precision:

  \[
  F_2 = \frac{5TP}{5TP + FP + 4FN}
  \]

- **Additional metrics:** Precision, recall, F1-score, F0.5-score, and accuracy are reported in `results.txt`.
- **Consistency checks:** If the number of rows or the timestamps in an output file do not exactly match the ground truth, the method is flagged as invalid and excluded from the ranking.

## 6. Visual inspection in Label Studio

1. Place your Label Studio API token in a **local** `KEY.txt` file. Never upload or commit it to GitHub.
2. Select the desired output directory in `run_upload.py`.
3. Run:

   ```bash
   python run_upload.py
   ```

The script opens Label Studio in the browser and displays the predicted segments over the sensor signals.

## 7. Ensemble voting

When outputs from multiple methods are available, `run_voting.py` can aggregate them row by row:

- **Percentage-based quorum:** Configure `MIN_ALGORITHMS_PERCENT` (for example, `50.0` for a 50% threshold or `100.0` for consensus).
- **Weighted voting:** Configure `METHOD_WEIGHTS` to assign different weights to individual methods according to their reliability.
- **Smoothing:** Configure `MIN_DURATION_SECONDS` to suppress brief label fluctuations.

The generated `voting_output/` directory is evaluated by `run_benchmark.py` like any other detection method.

## 8. Integrating an existing algorithm

An existing algorithm—whether based on signal processing, heuristics, or machine learning—can be integrated by following the framework's input/output conventions.

### Required interface conventions

1. **Filename:** Conventionally `method_<name>.py`.
2. **Output directory:** Must end in `_output` (for example, `my_detector_output/`).
3. **Matching filenames:** Each input CSV must be saved under exactly the same filename in the method's output directory.
4. **Matching rows and timestamps:** Preserve the exact number and order of rows, including matching values in `timestamp_s`.
5. **Additional label column:** Preserve all original sensor columns and append exactly one column named `label`.
6. **Allowed labels:** Every `label` value must be either `exercising` or `pause`.

### Step-by-step integration

**Step 1 — Copy the template**

Copy `method_template.py.txt` and rename the copy, for example, to `method_energy.py`.

**Step 2 — Define the output directory**

At the top of the method script, define a unique output folder:

```python
OUTPUT_FOLDER_NAME = "energy_output"
OUTPUT_FOLDER = HERE / OUTPUT_FOLDER_NAME
```

**Step 3 — Implement `predict()`**

Replace the implementation of `predict(rows)`:

- `rows` is a list of dictionaries representing CSV rows.
- Extract the required sensor columns (for example, `acc_x` or `gyro_z`).
- Apply your detection logic.
- Return a list or array of booleans with exactly as many entries as `rows`: `True` for `exercising`, `False` for `pause`.

**Step 4 — Preserve atomic output writing**

The template first writes output to a temporary file with the `.partial` extension and renames it to `.csv` only when writing is complete. This prevents incomplete CSV files if execution is interrupted.

**Step 5 — Register the method**

In `run_methods.py`, import and register the method:

```python
from method_energy import energy_method

METHODS_TO_RUN = [energy_method]
```

After running `python run_methods.py` and `python run_benchmark.py`, the method will appear in the benchmark ranking.

## 9. Common issues and best practices

- **Do not drop or resample rows.** Shortening a time series or omitting rows causes benchmark errors such as `Row-count mismatch` or `Timestamp mismatch`. Map all computations back to the exact input timestamps.
- **Check optional sensor columns.** Fields such as `acc_earth_x` or `altitude_m` are not present in every dataset. Check before accessing them, for example, with `'acc_earth_x' in rows[0]` (when `rows` is nonempty).
- **Reduce noisy label fluctuations.** Pointwise classifiers may produce rapid switching between classes. Consider closing short gaps within exercise intervals and removing isolated spikes, for example, using morphological closing and opening.
- **Remove stale output directories before rerunning.** After changing a method, delete its existing `<name>_output/` directory so that all output files are regenerated.
