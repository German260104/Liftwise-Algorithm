"""
GREEDY GAUSSIAN SEGMENTATION (GGS) & RANDOM FOREST KLASSIFIKATOR
================================================================
Dieses Modul implementiert die Greedy Gaussian Segmentation (GGS) zur
unueberwachten Zeitreihen-Segmentierung von Sensordaten (Beschleunigung & Gyroskop)
kombiniert mit einem Random Forest Klassifikator zur Uebungserkennung (exercising / pause).

Ablauf der Pipeline:
1. Signalvorverarbeitung:
   - Normalisierung der Spaltennamen und Pruefung auf fehlende Werte
   - Median-Filterung zur Spike-Entfernung
   - Butterworth-Tiefpassfilterung (Glaettung von hochfrequentem Rauschen)
   - Berechnung euklidischer Normen (acc_norm, gyro_norm) und Ableitungen
2. GGS Segmentierung:
   - Standardisierung der Sensorsignale
   - Berechnung von Praefix-Summen fuer schnelle Kovarianz-/Likelihood-Berechnung
   - Greedy-Suche nach optimalen Breakpoints unter Kovarianz-Regularisierung
   - Automatische Bestimmung der optimalen Segmentanzahl K via Elbow-Methode
3. Merkmalsextraktion pro Segment:
   - Dauer, Standardabweichung, Signalenergie, Autokorrelations-Periodizitaet
4. Klassifikation / Training:
   - Random Forest Modell auf den Segmentmerkmalen
5. Inferenz & Framework-Integration:
   - 'ggs_method()' und 'predict()' fuer nahtlose Integration in das Benchmark-Framework
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import quote, unquote

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# ==============================================================================
# 1. KONFIGURATION & HYPERPARAMETER
# ==============================================================================

# Standard-Abtastrate und Filterparameter fuer die Vorverarbeitung
DEFAULT_FS = 100                    # Abtastrate in Hertz (Hz)
DEFAULT_BUTTERWORTH_CUTOFF = 20     # Grenzfrequenz des Tiefpassfilters in Hz
DEFAULT_BUTTERWORTH_ORDER = 3       # Filterordnung des Butterworth-Filters

# Basis-Pfade
PROJECT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_CSV_DIR = PROJECT_DIR / "ground_truth_csvs"
OUTPUT_DIR = PROJECT_DIR / "ggs_dependencies"

# Ground Truth Spalten- und Label-Definitionen
GROUND_TRUTH_COLUMN = "label"
EXERCISE_LABEL_VALUES = ["exercise", "exercising", "Übung", 1, True]
LABEL_STUDIO_BASE_URL = "http://localhost:8081"
SHOW_BATCH_PLOTS = False

# Machine Learning Parameter
RANDOM_STATE = 42
TEST_SIZE = 0.20
RUN_GROUP_CROSS_VALIDATION = True
N_GROUP_FOLDS = 5

# Feature-Spalten, die in das Random Forest Modell einfliessen
FEATURE_COLUMNS = [
    "duration_seconds",         # Laenge des Segments in Sekunden
    "acc_std",                  # Standardabweichung der Beschleunigungsmagnitude
    "gyro_std",                 # Standardabweichung der Gyroskopmagnitude
    "acc_energy",               # Mittlere quadratische Energie (Beschleunigung)
    "gyro_energy",              # Mittlere quadratische Energie (Gyroskop)
    "periodicity_score",        # Autokorrelations-Periodizitaets-Score (0.0 bis 1.0)
    "dominant_period_seconds",  # Dominante Periodendauer in Sekunden
]

# Hyperparameter fuer den GGS Kern-Algorithmus (Gauß-Segmentierung)
GGS_PARAMETERS = {
    "max_breakpoints": 200,             # Maximale Anzahl an Schnittpunkten (K)
    "min_segment_size": 100,            # Minimale Segmentlaenge (100 Samples = 1s bei 100Hz)
    "candidate_step": 25,               # Schrittweite bei der Kandidatensuche fuer Splits
    "covariance_regularization": 1e-6,   # Diagonal-Regularisierung der Kovarianzmatrix
    "lambda_regularization": 1e-3,       # Penalty-Gewicht fuer inverse Kovarianz
}

# Vorverarbeitungsparameter als Mapping
PREPROCESSING_PARAMETERS = {
    "fs": DEFAULT_FS,
    "cutoff": DEFAULT_BUTTERWORTH_CUTOFF,
    "order": DEFAULT_BUTTERWORTH_ORDER,
}

# Schwellenwerte fuer das Ground-Truth-Matching von Segmenten
GROUND_TRUTH_THRESHOLDS = {
    "exercise_iou_threshold": 0.50,         # Mindest-IoU fuer Klasse 'exercise'
    "not_exercise_iou_threshold": 0.10,     # Max-IoU fuer Klasse 'not_exercise'
    "min_exercise_coverage": 0.80,          # Mindestabdeckung durch Uebungsannotation
    "contained_exercise_coverage": 0.95,    # Abdeckung fuer vollstaendig enthaltene Uebungen
    "max_negative_coverage": 0.10,          # Max. tolerierte Ueberlappung fuer Pause
}

# Parameter fuer den Random Forest Klassifikator
RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 300,            # Anzahl der Entscheidungsbaeume
    "class_weight": "balanced",     # Automatisches Ausbalancieren unausgeglichener Klassen
    "random_state": RANDOM_STATE,   # Reproduzierbarkeit
    "n_jobs": -1,                   # Parallele Ausfuehrung ueber alle CPU-Kerne
}

# Liste der Sensor-Kernachsen
SENSOR_COLS = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z"
]

# Mapping zur Vereinheitlichung unterschiedlicher Schreibweisen in Rohdaten
COLUMN_ALIASES = {
    "acc_x": ["acc_x", "accX", "AccX", "accelerometer_x"],
    "acc_y": ["acc_y", "accY", "AccY", "accelerometer_y"],
    "acc_z": ["acc_z", "accZ", "AccZ", "accelerometer_z"],
    "gyro_x": ["gyro_x", "gyroX", "gyrX", "GyroX", "GyrX", "gyroscope_x"],
    "gyro_y": ["gyro_y", "gyroY", "gyrY", "GyroY", "GyrY", "gyroscope_y"],
    "gyro_z": ["gyro_z", "gyroZ", "gyrZ", "GyroZ", "GyrZ", "gyroscope_z"],
    "acc_earth_x": ["acc_earth_x", "accEarthX", "acc_earthX"],
    "acc_earth_y": ["acc_earth_y", "accEarthY", "acc_earthY"],
    "acc_earth_z": ["acc_earth_z", "accEarthZ", "acc_earthZ"],
    "timestamp_s": ["timestamp_s", "time_s", "seconds", "timestamp", "time"],
}


# ==============================================================================
# 2. VORVERARBEITUNG & SIGNALFILTERUNG
# ==============================================================================

def normalize_dataset_columns(dataset: pd.DataFrame, fs: float = DEFAULT_FS) -> pd.DataFrame:
    """
    Normalisiert Spaltennamen eines Rohdatensatzes anhand von COLUMN_ALIASES.
    
    1. Prueft vorhandene Spalten und benennt Synonyme (z.B. 'accX' -> 'acc_x') um.
    2. Stellt sicher, dass alle 6 Standard-Sensorachsen (acc/gyro x,y,z) vorhanden sind.
    3. Erzeugt bei fehlender Zeitspalte synthetische Zeitstempel anhand der Abtastrate fs.
    
    Args:
        dataset: Originaler DataFrame aus der CSV-Datei.
        fs: Abtastrate in Hertz fuer synthetische Zeitstempel.
        
    Returns:
        pd.DataFrame mit vereinheitlichten Spaltennamen.
    """
    normalized_dataset = dataset.copy()
    rename_map = {}

    for target_col, aliases in COLUMN_ALIASES.items():
        if target_col in normalized_dataset.columns:
            continue

        source_col = next(
            (alias for alias in aliases if alias in normalized_dataset.columns),
            None
        )

        if source_col is not None:
            rename_map[source_col] = target_col

    if rename_map:
        normalized_dataset = normalized_dataset.rename(columns=rename_map)
        print("Spalten fuer GGS normalisiert:")
        for source_col, target_col in rename_map.items():
            print(f"  {source_col} -> {target_col}")

    missing_sensor_cols = [
        col for col in SENSOR_COLS
        if col not in normalized_dataset.columns
    ]

    if missing_sensor_cols:
        raise ValueError(
            "Fehlende erforderliche Sensorachsen nach Normalisierung: "
            + ", ".join(missing_sensor_cols)
        )

    if "timestamp_s" not in normalized_dataset.columns:
        normalized_dataset.insert(
            0,
            "timestamp_s",
            np.arange(len(normalized_dataset)) / fs
        )
        print(f"Synthetische 'timestamp_s'-Spalte erzeugt (fs={fs} Hz).")

    return normalized_dataset


def save_prepared_dataset(dataset: pd.DataFrame, output_path: Path | str) -> None:
    """Speichert den vorverarbeiteten Datensatz als CSV-Datei am Zielpfad."""
    dataset.to_csv(output_path, index=False)
    print(f"Vorverarbeiteter Datensatz gespeichert: {output_path}")


def inspect_dataset_quality(
    dataset: pd.DataFrame,
    time_col: str = "timestamp_s",
    sensor_cols: list[str] | None = None,
    fs: float = DEFAULT_FS,
    gap_tolerance_factor: float = 1.5,
    max_missing_seconds: float = 10,
) -> pd.DataFrame:
    """
    Prueft die Datenqualitaet: Fehlwerte, Zeitstempel-Abstaende und Luecken.
    
    - Zaehlt NaNs pro Sensorachse.
    - Entfernt zusammenhaengende Ausfallperioden (> max_missing_seconds).
    - Analysiert die zeitlichen Differenzen (Sampling-Konsistenz).
    
    Args:
        dataset: Zu pruefender DataFrame.
        time_col: Name der Zeitstempelspalte.
        sensor_cols: Liste der zu analysierenden Sensorspalten.
        fs: Erwartete Abtastfrequenz in Hz.
        gap_tolerance_factor: Faktor, ab dem ein Zeitabstand als Luecke gilt.
        max_missing_seconds: Maximale Laenge tolerierter Luecken vor Loeschung.
        
    Returns:
        Bereinigter DataFrame ohne lange Ausfallphasen.
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_COLS

    inspected_dataset = dataset.copy()

    print("Fehlwerte (NaNs) pro Sensorspalte:")
    print(inspected_dataset[sensor_cols].isna().sum())

    missing_any_sensor = inspected_dataset[sensor_cols].isna().any(axis=1)
    group_id = (missing_any_sensor != missing_any_sensor.shift()).cumsum()

    max_missing_samples = int(max_missing_seconds * fs)
    indices_to_remove = []

    for _, group in inspected_dataset.groupby(group_id):
        group_is_missing = missing_any_sensor.loc[group.index].all()

        if group_is_missing and len(group) > max_missing_samples:
            indices_to_remove.extend(group.index)

    if len(indices_to_remove) > 0:
        print(f"\nEntferne {len(indices_to_remove)} Samples aus langen Luecken (> {max_missing_seconds}s).")
        inspected_dataset = inspected_dataset.drop(index=indices_to_remove)
        inspected_dataset = inspected_dataset.reset_index(drop=True)
    else:
        print(f"\nKeine kritischen Luecken ueber {max_missing_seconds}s erkannt.")

    if time_col in inspected_dataset.columns:
        timestamp = pd.to_numeric(inspected_dataset[time_col], errors="coerce")
        dt = timestamp.diff()
        expected_dt = 1 / fs

        print("\nStatistik der Abtastintervall-Abstaende (dt):")
        print(dt.describe())

        print("\nErwartetes dt:", expected_dt)
        print("Geschaetzte mittlere Frequenz:", 1 / dt.median())

        large_gaps = dt[dt > expected_dt * gap_tolerance_factor]

        print("\nAnzahl auffaellig grosser Luecken:", len(large_gaps))

        if len(large_gaps) > 0:
            print("\nGroesste Zeitstempel-Luecken:")
            print(large_gaps.sort_values(ascending=False).head())
    else:
        print(f"\nZeitspalte '{time_col}' nicht gefunden; Intervallpruefung uebersprungen.")

    return inspected_dataset


def apply_median_filter_to_dataset(
    dataset: pd.DataFrame,
    sensor_cols: list[str],
    kernel_size: int = 3,
    output_suffix: str = "_median",
) -> pd.DataFrame:
    """
    Wendet einen Median-Filter auf die Sensorspalten an.
    
    Der Median-Filter entfernt effektiv isolierte Spikes und Sensorausreisser,
    ohne Kanten und rhythmische Bewegungsmuster zu verschleifen.
    """
    filtered_dataset = dataset.copy()

    for col in sensor_cols:
        signal = pd.to_numeric(filtered_dataset[col], errors="coerce")
        signal = signal.interpolate().bfill().ffill().to_numpy()
        filtered_dataset[col + output_suffix] = medfilt(signal, kernel_size=kernel_size)

    return filtered_dataset


def butter_lowpass_filter(
    signal: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 3,
) -> np.ndarray:
    """
    Fuer ein 1D-Signal: Butterworth-Tiefpassfilterung mit Phasenkorrektur (filtfilt).
    
    Glaettet hochfrequentes Sensorrauschen oberhalb der Grenzfrequenz 'cutoff'.
    """
    b, a = butter(order, cutoff, btype="low", fs=fs)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def apply_butterworth_filter_to_dataset(
    dataset: pd.DataFrame,
    sensor_cols: list[str],
    cutoff: float = DEFAULT_BUTTERWORTH_CUTOFF,
    fs: float = DEFAULT_FS,
    order: int = DEFAULT_BUTTERWORTH_ORDER,
    input_suffix: str = "_median",
    output_suffix: str = "_butter",
) -> pd.DataFrame:
    """
    Wendet den Butterworth-Tiefpassfilter auf alle zuvor median-gefilterten Achsen an.
    """
    filtered_dataset = dataset.copy()

    for col in sensor_cols:
        input_col = col + input_suffix
        output_col = col + output_suffix

        signal = pd.to_numeric(filtered_dataset[input_col], errors="coerce")
        signal = signal.interpolate().bfill().ffill().to_numpy()

        filtered_dataset[output_col] = butter_lowpass_filter(
            signal,
            cutoff=cutoff,
            fs=fs,
            order=order
        )

    return filtered_dataset


def add_euclidean_norms_and_derivatives(
    dataset: pd.DataFrame,
    fs: float = DEFAULT_FS,
) -> pd.DataFrame:
    """
    Ergaenzt Signalableitungen (Gradienten) und richtungsunabhaengige Betraege (Normen).
    
    - acc_x_butter_derivative ...: Momentane zeitliche Aenderung der Beschleunigung
    - acc_norm_butter: Euklidische Gesamtnorm sqrt(ax^2 + ay^2 + az^2)
    - gyro_norm_butter: Euklidische Gesamtnorm sqrt(gx^2 + gy^2 + gz^2)
    """
    transformed_dataset = dataset.copy()

    acc_cols = ["acc_x_butter", "acc_y_butter", "acc_z_butter"]
    gyro_cols = ["gyro_x_butter", "gyro_y_butter", "gyro_z_butter"]
    time_col = "timestamp_s"
    base_cols = acc_cols + gyro_cols

    if time_col in transformed_dataset.columns:
        time_values = pd.to_numeric(transformed_dataset[time_col], errors="coerce")
        time_values = time_values.interpolate().bfill().ffill().to_numpy()
    else:
        time_values = None

    # Ableitungen (Differenzialgradienten) fuer jede gefilterte Achse berechnen
    for col in base_cols:
        derivative_col = col + "_derivative"

        signal = pd.to_numeric(transformed_dataset[col], errors="coerce")
        signal = signal.interpolate().bfill().ffill().to_numpy()

        if time_values is not None:
            transformed_dataset[derivative_col] = np.gradient(signal, time_values)
        else:
            transformed_dataset[derivative_col] = np.gradient(signal) * fs

    # Euklidische Gesamtnormen (Betraege) berechnen
    transformed_dataset["acc_norm_butter"] = np.sqrt(
        transformed_dataset["acc_x_butter"] ** 2
        + transformed_dataset["acc_y_butter"] ** 2
        + transformed_dataset["acc_z_butter"] ** 2
    )

    transformed_dataset["gyro_norm_butter"] = np.sqrt(
        transformed_dataset["gyro_x_butter"] ** 2
        + transformed_dataset["gyro_y_butter"] ** 2
        + transformed_dataset["gyro_z_butter"] ** 2
    )

    return transformed_dataset


# ==============================================================================
# 3. GGS KERN-ALGORITHMUS (GREEDY GAUSSIAN SEGMENTATION)
# ==============================================================================

# Sensorachsen fuer die Gaußsche Segmentierung
GGS_COLS = [
    "acc_x_butter", "acc_y_butter", "acc_z_butter",
    "gyro_x_butter", "gyro_y_butter", "gyro_z_butter",
]


def prepare_ggs_input_matrix(
    dataset: pd.DataFrame,
    ggs_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Erstellt die z-standardisierte Eingabematrix X fuer den GGS-Algorithmus.
    
    1. Extrahiert die gefilterten Sensorachsen (acc/gyro butter).
    2. Interpoliert eventuell verbliebene NaNs.
    3. Z-standardisiert jedes Signal auf Mittelwert = 0 und Standardabweichung = 1.
    
    Returns:
        2D-NumPy-Array der Dimension (n_samples, n_features).
    """
    if ggs_cols is None:
        ggs_cols = GGS_COLS

    ggs_data = dataset[ggs_cols].copy()

    for col in ggs_cols:
        ggs_data[col] = pd.to_numeric(ggs_data[col], errors="coerce")
        ggs_data[col] = ggs_data[col].interpolate().bfill().ffill()

    x = ggs_data.to_numpy()

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1

    x_standardized = (x - mean) / std
    return x_standardized


def calculate_prefix_statistics(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet kumulative Summen (Praefix-Statistiken) fuer O(1) Segmentberechnungen.
    
    Durch Praefix-Summen von x und den aeusseren Produkten (x * x^T) kann die
    Kovarianzmatrix fuer jedes beliebige Teilsegment [start, end) in konstanter
    Zeit O(1) ohne erneutes Iterieren ueber die Rohdaten berechnet werden.
    
    Args:
        x: Standardisierte Datenmatrix (n_samples, n_features).
        
    Returns:
        Tuple aus:
        - prefix_sum: Kumulative Summe der Dimension (n_samples + 1, n_features).
        - prefix_outer_sum: Kumulative aeussere Produkte (n_samples + 1, n_feat, n_feat).
    """
    n_samples, n_features = x.shape

    prefix_sum = np.zeros((n_samples + 1, n_features))
    prefix_outer_sum = np.zeros((n_samples + 1, n_features, n_features))

    prefix_sum[1:] = np.cumsum(x, axis=0)
    prefix_outer_sum[1:] = np.cumsum(
        np.einsum("ni,nj->nij", x, x),
        axis=0
    )

    return prefix_sum, prefix_outer_sum


def calculate_segment_regularized_log_likelihood(
    start: int,
    end: int,
    prefix_sum: np.ndarray,
    prefix_outer_sum: np.ndarray,
    covariance_regularization: float = 1e-6,
    lambda_regularization: float = 1e-3,
) -> float:
    """
    Berechnet die kovarianz-regulierte Gauß-Log-Likelihood fuer ein Teilsegment [start, end).
    
    Gemaess GGS-Paper:
    1. Mittelwert und empirische Kovarianzmatrix ueber Praefix-Summen bilden.
    2. Kovarianz mit Diagonalelement (covariance_regularization) konditionieren.
    3. Gaußsche Log-Likelihood ueber Determinante berechnen.
    4. Penalty-Term (lambda * Spur der inversen Kovarianz) abziehen, um uebermaessig
       spezifische Kovarianzen zu bestrafen.
    """
    n_segment_samples = end - start

    if n_segment_samples <= 1:
        return -np.inf

    segment_sum = prefix_sum[end] - prefix_sum[start]
    segment_outer_sum = prefix_outer_sum[end] - prefix_outer_sum[start]

    segment_mean = segment_sum / n_segment_samples
    segment_covariance = (segment_outer_sum / n_segment_samples) - np.outer(segment_mean, segment_mean)

    n_features = segment_covariance.shape[0]
    segment_covariance = segment_covariance + covariance_regularization * np.eye(n_features)

    sign, log_determinant = np.linalg.slogdet(segment_covariance)

    if sign <= 0:
        return -np.inf

    inverse_covariance = np.linalg.inv(segment_covariance)

    gaussian_log_likelihood = -0.5 * n_segment_samples * (
        n_features * np.log(2 * np.pi)
        + log_determinant
        + n_features
    )

    covariance_penalty = lambda_regularization * np.trace(inverse_covariance)
    regularized_log_likelihood = gaussian_log_likelihood - covariance_penalty

    return regularized_log_likelihood


def calculate_total_regularized_log_likelihood(
    segments: list[tuple[int, int]],
    prefix_sum: np.ndarray,
    prefix_outer_sum: np.ndarray,
    covariance_regularization: float = 1e-6,
    lambda_regularization: float = 1e-3,
) -> float:
    """Berechnet die gesamte regulierte Log-Likelihood ueber alle Segmente."""
    total_phi = 0.0

    for start, end in segments:
        total_phi += calculate_segment_regularized_log_likelihood(
            start,
            end,
            prefix_sum,
            prefix_outer_sum,
            covariance_regularization=covariance_regularization,
            lambda_regularization=lambda_regularization
        )

    return total_phi


def find_best_split_for_segment(
    start: int,
    end: int,
    prefix_sum: np.ndarray,
    prefix_outer_sum: np.ndarray,
    min_segment_size: int = 100,
    candidate_step: int = 25,
    covariance_regularization: float = 1e-6,
    lambda_regularization: float = 1e-3,
) -> tuple[int | None, float]:
    """
    Findet innerhalb eines Segments den optimalen Schnittpunkt (Split).
    
    Testet alle zulaessigen Kandidaten-Indizes im Abstand 'candidate_step' unter
    Einhaltung der Mindestsegmentgroesse 'min_segment_size'.
    
    Returns:
        Tuple aus:
        - best_split: Bester Schnitt-Index (oder None falls kein Gewinn).
        - best_improvement: Likelihood-Verbesserung gegenueber dem ungeteilten Segment.
    """
    parent_phi = calculate_segment_regularized_log_likelihood(
        start,
        end,
        prefix_sum,
        prefix_outer_sum,
        covariance_regularization=covariance_regularization,
        lambda_regularization=lambda_regularization
    )

    best_split = None
    best_improvement = 0.0

    first_candidate = start + min_segment_size
    last_candidate = end - min_segment_size

    for split in range(first_candidate, last_candidate + 1, candidate_step):
        left_phi = calculate_segment_regularized_log_likelihood(
            start,
            split,
            prefix_sum,
            prefix_outer_sum,
            covariance_regularization=covariance_regularization,
            lambda_regularization=lambda_regularization
        )
        right_phi = calculate_segment_regularized_log_likelihood(
            split,
            end,
            prefix_sum,
            prefix_outer_sum,
            covariance_regularization=covariance_regularization,
            lambda_regularization=lambda_regularization
        )

        improvement = (left_phi + right_phi) - parent_phi

        if improvement > best_improvement:
            best_improvement = improvement
            best_split = split

    return best_split, best_improvement


def plot_regularized_log_likelihood_curve(
    ggs_history: list[dict],
    selected_k: int | None = None,
) -> None:
    """Visualisiert den Verlauf der regulierten Log-Likelihood-Kurve ueber K."""
    k_values = [entry["k"] for entry in ggs_history]
    phi_values = [entry["regularized_log_likelihood"] for entry in ggs_history]

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, phi_values, marker="o")

    if selected_k is not None:
        selected_phi = phi_values[selected_k]
        plt.axvline(selected_k, linestyle="--", label=f"Ausgewaehltes K = {selected_k}")
        plt.scatter([selected_k], [selected_phi], zorder=3)

    plt.xlabel("Anzahl Breakpoints K")
    plt.ylabel("Regulierte Log-Likelihood")
    plt.title("GGS Log-Likelihood Kurve ueber K")
    plt.grid(True)
    plt.legend()
    plt.show()


def select_optimal_k_from_history(ggs_history: list[dict]) -> int:
    """
    Waehlt automatisch das optimale K anhand der Knick-/Elbow-Methode.
    
    Normalisiert K und die Log-Likelihood auf [0, 1] und maximiert den Abstand
    zur Diagonale (maximaler zusaetzlicher Informationsgewinn pro Breakpoint).
    """
    if len(ggs_history) == 0:
        raise ValueError("ggs_history ist leer; kein K auswaehlbar.")

    if len(ggs_history) == 1:
        only_entry = ggs_history[0]
        print("Nur K=0 verfuegbar; keine Breakpoints waehlbar.")
        print(
            f"Automatisch gewaehltes K: {only_entry['k']}, "
            f"Likelihood={only_entry['regularized_log_likelihood']:.4f}, "
            f"Gewinn={only_entry['improvement']:.4f}"
        )
        return only_entry["k"]

    k_values = np.array([entry["k"] for entry in ggs_history], dtype=float)
    phi_values = np.array(
        [entry["regularized_log_likelihood"] for entry in ggs_history],
        dtype=float
    )

    k_range = k_values.max() - k_values.min()
    phi_range = phi_values.max() - phi_values.min()

    if k_range == 0:
        k_normalized = np.zeros_like(k_values)
    else:
        k_normalized = (k_values - k_values.min()) / k_range

    if phi_range == 0:
        phi_normalized = np.zeros_like(phi_values)
    else:
        phi_normalized = (phi_values - phi_values.min()) / phi_range

    if len(ggs_history) < 3:
        candidate_indices = np.where(k_values > 0)[0]
    else:
        candidate_indices = np.arange(1, len(ggs_history) - 1)
        candidate_indices = candidate_indices[k_values[candidate_indices] > 0]

    if len(candidate_indices) == 0:
        candidate_indices = np.where(k_values > 0)[0]

    if len(candidate_indices) == 0:
        selected_index = 0
    else:
        elbow_scores = phi_normalized - k_normalized
        selected_index = candidate_indices[np.argmax(elbow_scores[candidate_indices])]

    selected_entry = ggs_history[int(selected_index)]
    selected_k = selected_entry["k"]

    print(
        f"Automatisch gewaehltes K: {selected_k}, "
        f"Likelihood={selected_entry['regularized_log_likelihood']:.4f}, "
        f"Gewinn={selected_entry['improvement']:.4f}"
    )

    return selected_k


def plot_k_selection(ggs_history: list[dict], selected_k: int) -> None:
    """Zeigt diagnostische Plots zur automatischen Wahl von K via Elbow-Methode."""
    k_values = np.array([entry["k"] for entry in ggs_history])
    phi_values = np.array([entry["regularized_log_likelihood"] for entry in ggs_history])
    improvement_values = np.array([entry["improvement"] for entry in ggs_history])

    selected_matches = np.where(k_values == selected_k)[0]
    if len(selected_matches) == 0:
        raise ValueError(f"selected_k={selected_k} nicht in ggs_history gefunden.")

    selected_index = selected_matches[0]
    selected_phi = phi_values[selected_index]

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, phi_values, marker="o")
    plt.axvline(selected_k, linestyle="--", label=f"Gewaehltes K = {selected_k}")
    plt.scatter([selected_k], [selected_phi], zorder=3)
    plt.xlabel("Anzahl Breakpoints K")
    plt.ylabel("Regulierte Log-Likelihood")
    plt.title("Automatische K-Auswahl via Elbow-Methode")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, improvement_values, marker="o")
    plt.axvline(selected_k, linestyle="--", label=f"Gewaehltes K = {selected_k}")
    plt.xlabel("Anzahl Breakpoints K")
    plt.ylabel("Marginaler Informationsgewinn")
    plt.title("Marginaler Gewinn pro K")
    plt.grid(True)
    plt.legend()
    plt.show()


def calculate_regularized_log_likelihood_curve(
    dataset: pd.DataFrame,
    ggs_cols: list[str] | None = None,
    max_breakpoints: int = 200,
    min_segment_size: int = 100,
    candidate_step: int = 25,
    covariance_regularization: float = 1e-6,
    lambda_regularization: float = 1e-3,
    plot_curve: bool = True,
) -> list[dict]:
    """
    Hauptschleife des GGS-Verfahrens: Findet iterativ K = 0 ... max_breakpoints.
    
    In jeder Iteration wird ueber alle bestehenden Segmente iteriert, um dasjenige
    Segment und denjenigen Schnittpunkt zu finden, der die gesamte regulierte
    Log-Likelihood maximal erhoeht (Greedy Search).
    
    Returns:
        Historie aller Iterationsschritte K mit Breakpoints, Segmenten und Werten.
    """
    x = prepare_ggs_input_matrix(dataset, ggs_cols=ggs_cols)
    prefix_sum, prefix_outer_sum = calculate_prefix_statistics(x)

    n_samples = x.shape[0]
    segments = [(0, n_samples)]
    breakpoints = []
    history = []

    initial_phi = calculate_total_regularized_log_likelihood(
        segments,
        prefix_sum,
        prefix_outer_sum,
        covariance_regularization=covariance_regularization,
        lambda_regularization=lambda_regularization
    )

    history.append({
        "k": 0,
        "regularized_log_likelihood": initial_phi,
        "breakpoints": [],
        "segments": segments.copy(),
        "improvement": 0
    })

    for k in range(1, max_breakpoints + 1):
        best_segment_index = None
        best_split = None
        best_improvement = 0.0

        for segment_index, (start, end) in enumerate(segments):
            if end - start < 2 * min_segment_size:
                continue

            split, improvement = find_best_split_for_segment(
                start,
                end,
                prefix_sum,
                prefix_outer_sum,
                min_segment_size=min_segment_size,
                candidate_step=candidate_step,
                covariance_regularization=covariance_regularization,
                lambda_regularization=lambda_regularization
            )

            if split is not None and improvement > best_improvement:
                best_segment_index = segment_index
                best_split = split
                best_improvement = improvement

        if best_split is None:
            print("Kein weiterer Breakpoint verbessert die regulierte Log-Likelihood.")
            break

        start, end = segments.pop(best_segment_index)
        segments.append((start, best_split))
        segments.append((best_split, end))
        segments = sorted(segments)
        breakpoints.append(best_split)

        current_phi = calculate_total_regularized_log_likelihood(
            segments,
            prefix_sum,
            prefix_outer_sum,
            covariance_regularization=covariance_regularization,
            lambda_regularization=lambda_regularization
        )

        history.append({
            "k": k,
            "regularized_log_likelihood": current_phi,
            "breakpoints": sorted(breakpoints.copy()),
            "segments": segments.copy(),
            "improvement": best_improvement
        })

        print(f"K={k}: Phi={current_phi:.4f}, Gewinn={best_improvement:.4f}, Breakpoint={best_split}")

    if plot_curve:
        plot_regularized_log_likelihood_curve(history)

    return history


def get_segmentation_for_selected_k(
    ggs_history: list[dict],
    selected_k: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Extrahiert die Breakpoints und Segmentgrenzen fuer das gewaehlte K."""
    if selected_k < 0 or selected_k >= len(ggs_history):
        raise ValueError(f"selected_k muss zwischen 0 und {len(ggs_history) - 1} liegen.")

    selected_result = ggs_history[selected_k]
    selected_breakpoints = selected_result["breakpoints"]
    selected_segments = selected_result["segments"]

    print(f"Ausgewaehltes K: {selected_k}")
    print(f"Breakpoints ({len(selected_breakpoints)}): {selected_breakpoints}")
    print(f"Segmente ({len(selected_segments)}): {selected_segments}")

    return selected_breakpoints, selected_segments


# ==============================================================================
# 4. MERKMALSEXTRAKTION (FEATURE EXTRACTION)
# ==============================================================================

def calculate_segment_duration(start: int, end: int, fs: float = DEFAULT_FS) -> float:
    """Berechnet die zeitliche Dauer eines Segments in Sekunden."""
    return (end - start) / fs


def calculate_segment_activity_features(
    dataset: pd.DataFrame,
    start: int,
    end: int,
) -> dict[str, float]:
    """
    Berechnet statistische Aktivitaetsmerkmale fuer ein Segment [start, end).
    
    Merkmale:
    - acc_std: Standardabweichung der Beschleunigungsmagnitude (Bewegungsvarianz)
    - gyro_std: Standardabweichung der Drehratenmagnitude (Rotationsvarianz)
    - acc_energy: Mittlere quadratische Energie der Beschleunigung
    - gyro_energy: Mittlere quadratische Energie der Drehraten
    """
    segment = dataset.iloc[start:end]

    if segment.empty:
        return {
            "acc_std": 0.0,
            "gyro_std": 0.0,
            "acc_energy": 0.0,
            "gyro_energy": 0.0,
        }

    acc_signal = pd.to_numeric(segment["acc_norm_butter"], errors="coerce")
    gyro_signal = pd.to_numeric(segment["gyro_norm_butter"], errors="coerce")

    acc_signal = acc_signal.dropna().to_numpy()
    gyro_signal = gyro_signal.dropna().to_numpy()

    acc_std = float(np.std(acc_signal)) if len(acc_signal) > 0 else 0.0
    gyro_std = float(np.std(gyro_signal)) if len(gyro_signal) > 0 else 0.0
    acc_energy = float(np.mean(acc_signal ** 2)) if len(acc_signal) > 0 else 0.0
    gyro_energy = float(np.mean(gyro_signal ** 2)) if len(gyro_signal) > 0 else 0.0

    return {
        "acc_std": acc_std,
        "gyro_std": gyro_std,
        "acc_energy": acc_energy,
        "gyro_energy": gyro_energy,
    }


def calculate_segment_periodicity_features(
    dataset: pd.DataFrame,
    start: int,
    end: int,
    fs: float = DEFAULT_FS,
) -> dict[str, float | int]:
    """
    Berechnet Periodizitaetsmerkmale via Autokorrelation der Gyroskop-Magnitude.
    
    Untersucht Verzoegerungen (Lags) im sporttypischen Bereich von 0.3s bis 5.0s:
    - periodicity_score: Hoehe des maximalen Autokorrelationspeaks (0.0 = chaotisch/ruhig, 1.0 = perfekt periodisch)
    - dominant_lag_samples: Stichprobenabstand des dominanten Wiederholungszyklus
    - dominant_period_seconds: Geschaetzte Dauer einer einzelnen Wiederholung in Sekunden
    """
    segment = dataset.iloc[start:end]
    min_lag = int(0.3 * fs)
    max_lag = int(5.0 * fs)

    if segment.empty or len(segment) <= min_lag:
        return {
            "periodicity_score": 0.0,
            "dominant_lag_samples": 0,
            "dominant_period_seconds": 0.0,
        }

    signal = pd.to_numeric(segment["gyro_norm_butter"], errors="coerce")
    signal = signal.interpolate().bfill().ffill().to_numpy()

    if len(signal) == 0 or np.std(signal) < 1e-12:
        return {
            "periodicity_score": 0.0,
            "dominant_lag_samples": 0,
            "dominant_period_seconds": 0.0,
        }

    signal = signal - np.mean(signal)
    autocorrelation = np.correlate(signal, signal, mode="full")
    autocorrelation = autocorrelation[len(signal) - 1:]

    if autocorrelation[0] <= 1e-12:
        return {
            "periodicity_score": 0.0,
            "dominant_lag_samples": 0,
            "dominant_period_seconds": 0.0,
        }

    autocorrelation = autocorrelation / autocorrelation[0]
    valid_max_lag = min(max_lag, len(autocorrelation) - 1)

    if valid_max_lag < min_lag:
        return {
            "periodicity_score": 0.0,
            "dominant_lag_samples": 0,
            "dominant_period_seconds": 0.0,
        }

    valid_autocorrelation = autocorrelation[min_lag:valid_max_lag + 1]
    dominant_lag_offset = int(np.argmax(valid_autocorrelation))
    dominant_lag_samples = min_lag + dominant_lag_offset
    periodicity_score = float(valid_autocorrelation[dominant_lag_offset])

    return {
        "periodicity_score": periodicity_score,
        "dominant_lag_samples": dominant_lag_samples,
        "dominant_period_seconds": dominant_lag_samples / fs,
    }


def analyze_ggs_segments(
    dataset: pd.DataFrame,
    segments: list[tuple[int, int]],
    fs: float = DEFAULT_FS,
) -> pd.DataFrame:
    """
    Erstellt die vollstaendige Feature-Tabelle fuer alle Segmente eines Datensatzes.
    
    Verbindet Segmentgrenzen, Dauer, Aktivitaets- und Periodizitaetsmerkmale
    zu einem strukturierten DataFrame fuer Modell-Inferenz oder Training.
    """
    rows = []

    for segment_index, (start, end) in enumerate(segments):
        activity_features = calculate_segment_activity_features(
            dataset,
            start,
            end
        )
        periodicity_features = calculate_segment_periodicity_features(
            dataset,
            start,
            end,
            fs=fs
        )

        rows.append({
            "segment": segment_index,
            "start": start,
            "end": end,
            "number_of_samples": end - start,
            "duration_seconds": calculate_segment_duration(start, end, fs=fs),
            **activity_features,
            **periodicity_features
        })

    return pd.DataFrame(rows)


# ==============================================================================
# 5. GROUND TRUTH MATCHING & LABEL STUDIO ANBINDUNG
# ==============================================================================

def normalize_label_studio_filename(file_reference: str | Path) -> str:
    """
    Bereinigt Dateinamen aus Label Studio (entfernt URL-Encoding, Querys und UUID-Praefixe).
    
    Beispiel: 'http://localhost/d9e2f1a0-workout_123.csv?token=xyz' -> 'workout_123.csv'
    """
    filename = unquote(str(file_reference)).split("?")[0].split("#")[0]
    filename = Path(filename).name
    return re.sub(r"^[0-9a-fA-F]{8}-", "", filename)


def get_task_csv_filenames(task: dict) -> list[str]:
    """Extrahiert alle referenzierten CSV-Dateinamen aus einem Label Studio Task."""
    filenames = []

    for value in task.get("data", {}).values():
        if isinstance(value, str) and value.lower().split("?")[0].endswith(".csv"):
            filenames.append(normalize_label_studio_filename(value))

    file_upload = task.get("file_upload")
    if isinstance(file_upload, str):
        filenames.append(normalize_label_studio_filename(file_upload))

    return filenames


def extract_exercising_regions_from_results(
    results: list[dict],
    target_label: str = "exercising",
) -> list[dict[str, int]]:
    """Liest Uebungsregionen mit Start- und End-Index aus Label Studio Result-Eintraegen."""
    regions = []

    for result in results:
        value = result.get("value", result)
        labels = value.get("timeserieslabels", value.get("labels", []))

        if isinstance(labels, str):
            labels = [labels]

        if target_label not in labels:
            continue

        if "start" not in value or "end" not in value:
            continue

        regions.append({
            "start": int(value["start"]),
            "end": int(value["end"])
        })

    return regions


def extract_exercising_regions_from_task(
    task: dict,
    target_label: str = "exercising",
) -> list[dict[str, int]]:
    """Extrahiert alle annotierten Uebungsbereiche aus den Annotationen eines Tasks."""
    regions = []

    for annotation in task.get("annotations", []):
        regions.extend(
            extract_exercising_regions_from_results(
                annotation.get("result", []),
                target_label=target_label
            )
        )

    simplified_labels = task.get("label", [])
    if isinstance(simplified_labels, dict):
        simplified_labels = [simplified_labels]

    if isinstance(simplified_labels, list):
        regions.extend(
            extract_exercising_regions_from_results(
                simplified_labels,
                target_label=target_label
            )
        )

    return regions


def load_label_studio_ground_truth_regions(
    ground_truth_json_paths: list[str | Path],
    csv_path: str | Path,
    target_label: str = "exercising",
) -> list[dict[str, int]]:
    """Laedt die Ground Truth Uebungsregionen fuer eine bestimmte CSV aus Label Studio JSON-Dateien."""
    expected_filename = normalize_label_studio_filename(csv_path)
    available_filenames = []

    for json_path in ground_truth_json_paths:
        with open(json_path, "r", encoding="utf-8") as file:
            loaded_json = json.load(file)

        if isinstance(loaded_json, list):
            tasks = loaded_json
        elif isinstance(loaded_json, dict):
            tasks = loaded_json.get("tasks", loaded_json.get("results", [loaded_json]))
        else:
            tasks = []

        for task in tasks:
            if not isinstance(task, dict):
                continue

            task_filenames = get_task_csv_filenames(task)
            available_filenames.extend(task_filenames)

            if expected_filename not in task_filenames:
                continue

            regions = extract_exercising_regions_from_task(
                task,
                target_label=target_label
            )
            print(
                f"{len(regions)} Ground Truth Uebungsbereiche fuer {expected_filename} geladen."
            )
            return regions

    raise ValueError(
        f"Kein Ground Truth Eintrag fuer CSV-Datei '{expected_filename}' gefunden. "
        f"Verfuegbare Datensaetze: " + ", ".join(sorted(set(available_filenames)))
    )


def calculate_segment_ground_truth_overlap(
    segment_start: int,
    segment_end: int,
    gt_region: dict[str, int],
) -> dict[str, float | int]:
    """
    Berechnet die Ueberlappung (Intersection, IoU, Coverage) zwischen einem Segment und Ground Truth.
    
    Returns:
        dict mit Schnittlaenge, IoU (Jaccard), Segment-Abdeckung und GT-Abdeckung.
    """
    gt_start = gt_region["start"]
    gt_end = gt_region["end"]
    intersection = max(0, min(segment_end, gt_end) - max(segment_start, gt_start))
    segment_length = max(0, segment_end - segment_start)
    gt_length = max(0, gt_end - gt_start)
    union = segment_length + gt_length - intersection

    iou = (intersection / union) if union > 0 else 0.0
    segment_gt_coverage = (intersection / segment_length) if segment_length > 0 else 0.0
    gt_segment_coverage = (intersection / gt_length) if gt_length > 0 else 0.0

    return {
        "intersection_length": intersection,
        "iou": iou,
        "segment_gt_coverage": segment_gt_coverage,
        "gt_segment_coverage": gt_segment_coverage,
        "matched_gt_start": gt_start,
        "matched_gt_end": gt_end
    }


def assign_ground_truth_labels_to_segments(
    segment_features: pd.DataFrame,
    ground_truth_regions: list[dict[str, int]],
    exercise_iou_threshold: float = 0.50,
    not_exercise_iou_threshold: float = 0.10,
    min_exercise_coverage: float = 0.80,
    contained_exercise_coverage: float = 0.95,
    max_negative_coverage: float = 0.10,
) -> pd.DataFrame:
    """
    Weist GGS-Segmenten Ground Truth Klassen ('exercise', 'not_exercise', 'ambiguous') zu.
    
    Logik:
    - 'exercise' (1): Mindestens 80% Abdeckung durch Uebung und IoU >= 0.50 (oder 95% Coverage).
    - 'not_exercise' (0): Maximal 10% Ueberlappung mit Uebungen und IoU <= 0.10.
    - 'ambiguous' (NaN): Uebergangsphasen und unklare Randbereiche (werden vom Training ausgeschlossen).
    """
    thresholds = [exercise_iou_threshold, not_exercise_iou_threshold,
                  min_exercise_coverage, contained_exercise_coverage,
                  max_negative_coverage]
    if not all(0 <= value <= 1 for value in thresholds):
        raise ValueError("Ground Truth Schwellenwerte muessen zwischen 0 und 1 liegen.")
    if not (not_exercise_iou_threshold < exercise_iou_threshold
            and max_negative_coverage < min_exercise_coverage
            <= contained_exercise_coverage):
        raise ValueError("Ground Truth Positiv-/Negativ-Schwellenwerte ueberlappen.")

    labeled_segments = segment_features.copy()
    rows = []

    for row in labeled_segments.itertuples(index=False):
        if row.end <= row.start:
            raise ValueError("GGS-Segmente muessen eine positive Laenge besitzen.")
        intersections = []
        best_overlap = {
            "intersection_length": 0,
            "iou": 0,
            "segment_gt_coverage": 0,
            "gt_segment_coverage": 0,
            "matched_gt_start": np.nan,
            "matched_gt_end": np.nan
        }

        for gt_region in ground_truth_regions:
            overlap = calculate_segment_ground_truth_overlap(
                int(row.start),
                int(row.end),
                gt_region
            )

            if overlap["iou"] > best_overlap["iou"]:
                best_overlap = overlap
            if overlap["intersection_length"] > 0:
                intersections.append((max(row.start, gt_region["start"]),
                                      min(row.end, gt_region["end"])))

        max_gt_iou = best_overlap["iou"]
        # Vereinigung der Schnittflaechen verhindert, dass mehrere Uebungsbloecke als negativ gewertet werden
        covered = 0
        previous_end = row.start
        for start, end in sorted(intersections):
            covered += max(0, end - max(start, previous_end))
            previous_end = max(previous_end, end)
        total_coverage = covered / (row.end - row.start)
        coverage = best_overlap["segment_gt_coverage"]

        if (coverage >= min_exercise_coverage
                and (max_gt_iou >= exercise_iou_threshold
                     or coverage >= contained_exercise_coverage)):
            ground_truth_label = "exercise"
            ground_truth_binary = 1
        elif (max_gt_iou <= not_exercise_iou_threshold
              and total_coverage <= max_negative_coverage):
            ground_truth_label = "not_exercise"
            ground_truth_binary = 0
        else:
            ground_truth_label = "ambiguous"
            ground_truth_binary = np.nan

        rows.append({
            "ground_truth_label": ground_truth_label,
            "ground_truth_binary": ground_truth_binary,
            "max_gt_iou": max_gt_iou,
            "intersection_length": best_overlap["intersection_length"],
            "total_gt_coverage": total_coverage,
            "segment_gt_coverage": best_overlap["segment_gt_coverage"],
            "gt_segment_coverage": best_overlap["gt_segment_coverage"],
            "matched_gt_start": best_overlap["matched_gt_start"],
            "matched_gt_end": best_overlap["matched_gt_end"]
        })

    ground_truth_columns = pd.DataFrame(rows, columns=[
        "ground_truth_label", "ground_truth_binary", "max_gt_iou",
        "intersection_length", "total_gt_coverage", "segment_gt_coverage",
        "gt_segment_coverage", "matched_gt_start", "matched_gt_end",
    ])

    return pd.concat(
        [labeled_segments.reset_index(drop=True), ground_truth_columns],
        axis=1
    )


def create_training_rows_from_ground_truth(
    segment_features_with_ground_truth: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Filtert mehrdeutige Segmente heraus und stellt die Trainingsdatenmatrix zusammen.
    
    Verwirft alle Zeilen mit 'ambiguous' und behaelt nur eindeutig als Uebung (1)
    oder Pause (0) zugeordnete Segmente.
    """
    training_rows = segment_features_with_ground_truth[
        segment_features_with_ground_truth["ground_truth_label"] != "ambiguous"
    ].copy()

    training_rows.insert(0, "dataset_name", dataset_name)

    training_columns = [
        "dataset_name",
        "start",
        "end",
        "duration_seconds",
        "acc_std",
        "gyro_std",
        "acc_energy",
        "gyro_energy",
        "periodicity_score",
        "dominant_lag_samples",
        "dominant_period_seconds",
        "ground_truth_binary"
    ]

    return training_rows[training_columns]


def classify_exercise_segments(
    segment_features: pd.DataFrame,
    min_duration_seconds: float = 15,
    max_duration_seconds: float = 40,
    min_gyro_std: float = 20,
    min_periodicity_score: float = 0.50,
    min_dominant_period_seconds: float = 1.0,
    max_dominant_period_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    Regelbasierte Baseline-Klassifikation ohne Machine-Learning-Modell.
    
    Ueberprueft feste Schwellenwerte fuer Dauer, Rotation und Periodizitaet.
    """
    classified_segments = segment_features.copy()

    classified_segments["duration_ok"] = (
        (classified_segments["duration_seconds"] >= min_duration_seconds)
        & (classified_segments["duration_seconds"] <= max_duration_seconds)
    )
    classified_segments["activity_ok"] = (
        classified_segments["gyro_std"] >= min_gyro_std
    )
    classified_segments["periodicity_ok"] = (
        (classified_segments["periodicity_score"] >= min_periodicity_score)
        & (
            classified_segments["dominant_period_seconds"]
            >= min_dominant_period_seconds
        )
        & (
            classified_segments["dominant_period_seconds"]
            <= max_dominant_period_seconds
        )
    )
    classified_segments["predicted_exercise_set"] = (
        classified_segments["duration_ok"]
        & classified_segments["activity_ok"]
        & classified_segments["periodicity_ok"]
    )

    return classified_segments


def get_predicted_exercise_segments(
    classified_segments: pd.DataFrame,
) -> list[tuple[int, int]]:
    """Liefert eine Liste von (start, end) Tuples aller als Uebung klassifizierten Segmente."""
    predicted_segments = classified_segments[
        classified_segments["predicted_exercise_set"] == True
    ]

    return [
        (int(row.start), int(row.end))
        for row in predicted_segments.itertuples(index=False)
    ]


def export_exercise_segments_to_label_studio_json(
    exercise_segments: list[tuple[int, int]],
    output_path: Path | str,
    csv_path: str,
    label_name: str = "Übung",
    from_name: str = "label",
    to_name: str = "ts",
    model_version: str = "ggs_segment_classifier",
) -> None:
    """Exportiert erkannte Uebungsintervalle im Label-Studio Task-JSON-Format."""
    results = []

    for segment_index, (start, end) in enumerate(exercise_segments):
        results.append({
            "id": f"exercise_segment_{segment_index}",
            "from_name": from_name,
            "to_name": to_name,
            "type": "timeserieslabels",
            "value": {
                "start": int(start),
                "end": int(end),
                "instant": False,
                "timeserieslabels": [label_name]
            }
        })

    task = {
        "data": {
            "csv": csv_path
        },
        "predictions": [{
            "model_version": model_version,
            "score": 1.0,
            "result": results
        }]
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump([task], file, indent=2, ensure_ascii=False)

    print(f"Uebungssegment-Vorhersagen gespeichert: {output_path}")


def export_breakpoints_to_label_studio_json(
    dataset: pd.DataFrame,
    breakpoints: list[int],
    output_path: Path | str,
    csv_path: str,
    label_name: str = "Predicted Change Point",
    from_name: str = "label",
    to_name: str = "ts",
    model_version: str = "ggs_selected_k",
    window_size_samples: int = 50,
) -> None:
    """Exportiert die gewaehlten GGS-Breakpoints als Label-Studio Vorhersage-Bereiche."""
    results = []
    n_samples = len(dataset)

    for breakpoint in breakpoints:
        start = max(0, breakpoint - window_size_samples)
        end = min(n_samples - 1, breakpoint + window_size_samples)

        results.append({
            "from_name": from_name,
            "to_name": to_name,
            "type": "timeserieslabels",
            "value": {
                "start": int(start),
                "end": int(end),
                "instant": False,
                "timeserieslabels": [label_name]
            }
        })

    task = {
        "data": {
            "csv": csv_path
        },
        "predictions": [{
            "model_version": model_version,
            "score": 1.0,
            "result": results
        }]
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump([task], file, indent=2)

    print(f"Label Studio Vorhersagen gespeichert: {output_path}")


# ==============================================================================
# 6. DATENSATZ-VERARBEITUNG & BATCH-PIPELINE
# ==============================================================================

def dataPreparation(
    path: Path | str,
    sensor_cols: list[str] | None = None,
    fs: float = DEFAULT_FS,
    cutoff: float = DEFAULT_BUTTERWORTH_CUTOFF,
    order: int = DEFAULT_BUTTERWORTH_ORDER,
) -> pd.DataFrame:
    """
    Fuehrt die vollstaendige Vorverarbeitungskette fuer eine Roh-CSV-Datei aus.
    
    1. Einlesen & Spaltennormalisierung (normalize_dataset_columns).
    2. Qualitaetspruefung & Lueckenentfernung (inspect_dataset_quality).
    3. Median-Filterung (apply_median_filter_to_dataset).
    4. Butterworth-Tiefpassfilterung (apply_butterworth_filter_to_dataset).
    5. Gradienten- und Normberechnung (add_euclidean_norms_and_derivatives).
    """
    dataset = pd.read_csv(path)
    dataset = normalize_dataset_columns(dataset, fs=fs)

    if sensor_cols is None:
        sensor_cols = SENSOR_COLS

    dataset = inspect_dataset_quality(
        dataset,
        time_col="timestamp_s",
        sensor_cols=sensor_cols,
        fs=fs,
        max_missing_seconds=10
    )

    dataset = apply_median_filter_to_dataset(
        dataset,
        sensor_cols=sensor_cols,
        kernel_size=3
    )

    dataset = apply_butterworth_filter_to_dataset(
        dataset,
        sensor_cols=sensor_cols,
        cutoff=cutoff,
        fs=fs,
        order=order,
        input_suffix="_median",
        output_suffix="_butter"
    )

    dataset = add_euclidean_norms_and_derivatives(
        dataset,
        fs=fs
    )

    return dataset


def extract_ground_truth_regions_from_csv(
    csv_path: Path | str,
    ground_truth_column: str = GROUND_TRUTH_COLUMN,
    exercise_label_values: list | None = None,
) -> list[dict[str, int]]:
    """
    Extrahiert zusammenhaengende Uebungsbereiche [start, end) aus einer gelabelten CSV-Datei.
    """
    dataset = pd.read_csv(csv_path)
    if not ground_truth_column or ground_truth_column not in dataset.columns:
        raise ValueError(
            f"{Path(csv_path).name}: Ground Truth Spalte {ground_truth_column!r} "
            f"nicht gefunden. Verfuegbare Spalten: {list(dataset.columns)}"
        )
    labels = dataset[ground_truth_column]
    if labels.isna().any():
        raise ValueError("Fehlende Ground Truth Labels in der CSV-Datei.")

    def normalize(value):
        if isinstance(value, str):
            return value.strip().casefold()
        return str(value).casefold()

    values = EXERCISE_LABEL_VALUES if exercise_label_values is None else exercise_label_values
    accepted = {normalize(value) for value in values}
    mask = labels.map(normalize).isin(accepted) | labels.isin(
        [value for value in values if not isinstance(value, str)]
    )
    edges = np.diff(np.r_[False, mask.to_numpy(dtype=bool), False].astype(int))
    regions = [
        {"start": int(start), "end": int(end)}
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]
    print(f"{Path(csv_path).name}: {len(regions)} Ground Truth Uebungsbereiche extrahiert.")
    return regions


def run_dataset_segmentation(
    csv_path: Path | str,
    output_dir: Path | str = OUTPUT_DIR,
    ggs_parameters: dict | None = None,
    preprocessing_parameters: dict | None = None,
    show_plots: bool = False,
    label_studio_base_url: str = LABEL_STUDIO_BASE_URL,
) -> tuple[pd.DataFrame, int, pd.DataFrame, str]:
    """
    Fuehrt die vollstaendige Vorverarbeitung, GGS-Segmentierung und Merkmalsextraktion aus.
    
    Returns:
        Tuple aus:
        - features: DataFrame mit Segmentmerkmalen
        - selected_k: Automatisch gewaehlte Anzahl Breakpoints
        - prepared: Vorverarbeiteter DataFrame
        - csv_url: URL fuer Label-Studio-Referenz
    """
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = csv_path.stem
    preprocessing = dict(PREPROCESSING_PARAMETERS if preprocessing_parameters is None
                         else preprocessing_parameters)
    ggs_parameters = dict(GGS_PARAMETERS if ggs_parameters is None else ggs_parameters)
    prepared = dataPreparation(csv_path, **preprocessing)
    prepared_name = f"{name}_prepared_for_ggs.csv"
    save_prepared_dataset(prepared, output_dir / prepared_name)
    csv_url = f"{label_studio_base_url.rstrip('/')}/{quote(prepared_name)}"
    history = calculate_regularized_log_likelihood_curve(
        prepared, ggs_cols=GGS_COLS, plot_curve=False, **ggs_parameters
    )
    selected_k = select_optimal_k_from_history(history)
    if show_plots:
        plot_k_selection(history, selected_k)
    breakpoints, segments = get_segmentation_for_selected_k(history, selected_k)
    features = analyze_ggs_segments(prepared, segments, fs=preprocessing["fs"])
    features.to_csv(output_dir / f"{name}_ggs_segment_features_k_{selected_k}.csv", index=False)
    baseline = classify_exercise_segments(features)
    baseline.to_csv(
        output_dir / f"{name}_ggs_segment_classification_k_{selected_k}.csv", index=False
    )
    export_exercise_segments_to_label_studio_json(
        get_predicted_exercise_segments(baseline),
        output_dir / f"{name}_ggs_exercise_predictions_k_{selected_k}.json",
        csv_url, model_version=f"ggs_segment_classifier_k_{selected_k}",
    )
    export_breakpoints_to_label_studio_json(
        prepared, breakpoints,
        output_dir / f"{name}_ggs_predictions_k_{selected_k}.json",
        csv_url, model_version=f"ggs_k_{selected_k}",
    )
    return features, selected_k, prepared, csv_url


def process_labeled_dataset(
    csv_path: Path | str,
    output_dir: Path | str = OUTPUT_DIR,
    ground_truth_column: str = GROUND_TRUTH_COLUMN,
    exercise_label_values: list | None = None,
    ggs_parameters: dict | None = None,
    preprocessing_parameters: dict | None = None,
    ground_truth_thresholds: dict | None = None,
    show_plots: bool = SHOW_BATCH_PLOTS,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Verarbeitet eine einzelne gelabelte CSV-Datei fuer das Modell-Training.
    
    Segmentiert die Datei, ordnet jedem Segment das Ground-Truth-Label zu
    und liefert die fertigen Trainingszeilen zurueck.
    """
    regions = extract_ground_truth_regions_from_csv(
        csv_path, ground_truth_column, exercise_label_values
    )
    features, selected_k, prepared, _ = run_dataset_segmentation(
        csv_path, output_dir, ggs_parameters, preprocessing_parameters, show_plots
    )
    original_labels = pd.read_csv(csv_path, usecols=[ground_truth_column])[ground_truth_column]
    if (len(original_labels) != len(prepared)
            or not original_labels.reset_index(drop=True).equals(
                prepared[ground_truth_column].reset_index(drop=True))):
        raise ValueError(
            "Vorverarbeitung hat Zeilenausrichtung veraendert. Ground Truth "
            "stimmt nicht mehr mit vorbereiteten Indizes ueberein."
        )
    labeled = assign_ground_truth_labels_to_segments(
        features, regions,
        **(GROUND_TRUTH_THRESHOLDS if ground_truth_thresholds is None
           else ground_truth_thresholds),
    )
    rows = create_training_rows_from_ground_truth(labeled, Path(csv_path).stem)
    if not np.isfinite(rows[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Nicht-endliche Merkmalswerte im Segment-Datensatz.")
    prefix = Path(output_dir) / Path(csv_path).stem
    labeled.to_csv(f"{prefix}_ggs_segment_features_with_ground_truth_k_{selected_k}.csv", index=False)
    rows.to_csv(f"{prefix}_ggs_training_rows_k_{selected_k}.csv", index=False)
    print(labeled[[
        "segment", "start", "end", "duration_seconds", "max_gt_iou",
        "segment_gt_coverage", "ground_truth_label", "ground_truth_binary",
    ]].to_string(index=False))
    return rows, labeled, selected_k


def write_json(path: Path | str, value: object) -> None:
    """Schreibt ein Python-Objekt formatiert als JSON-Datei."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)


def build_combined_training_dataset(
    csv_dir: Path | str = GROUND_TRUTH_CSV_DIR,
    output_dir: Path | str = OUTPUT_DIR,
    ground_truth_column: str = GROUND_TRUTH_COLUMN,
    **processing_options,
) -> tuple[pd.DataFrame, dict]:
    """
    Batch-Verarbeitung: Iteriert ueber alle gelabelten CSVs und erstellt Trainingsdatensatz.
    
    Aggregiert alle extrahierten Segmente in 'ggs_combined_training_dataset.csv'
    und schreibt eine Zusammenfassung nach 'ggs_batch_summary.json'.
    """
    csv_dir, output_dir = Path(csv_dir), Path(output_dir)
    paths = sorted(csv_dir.glob("*.csv"))
    print(f"Gefundene CSV-Datensaetze: {len(paths)}")
    if not paths:
        raise ValueError(f"Keine gelabelten *.csv-Dateien in {csv_dir} gefunden.")
    if csv_dir.resolve() == output_dir.resolve():
        raise ValueError("Original-CSVs und Output-Verzeichnis muessen getrennt sein.")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, failures, processed = [], [], []
    counts = {"exercise": 0, "not_exercise": 0, "ambiguous": 0}
    seen_names = set()
    for index, path in enumerate(paths, 1):
        print(f"\nDatensatz {index}/{len(paths)}: {path.name}")
        try:
            group_name = Path(normalize_label_studio_filename(str(path))).stem
            if group_name in seen_names:
                raise ValueError(f"Doppelte Aufnahme-Identitaet: {group_name}")
            rows, labeled, k = process_labeled_dataset(
                path, output_dir, ground_truth_column, **processing_options
            )
            rows["dataset_name"] = group_name
            all_rows.append(rows)
            seen_names.add(group_name)
            processed.append({"dataset_name": group_name, "filename": path.name, "selected_k": int(k)})
            for label, count in labeled["ground_truth_label"].value_counts().items():
                counts[label] += int(count)
        except Exception as exc:
            failures.append({"dataset": path.name, "reason": str(exc)})
            print(f"FEHLER bei {path.name}: {exc}")
    summary = {
        "datasets_found": len(paths), "datasets_processed": len(processed),
        "processed": processed, "failed_datasets": failures,
        "total_segments": sum(counts.values()), "segment_counts": counts,
        "ambiguous_excluded": counts["ambiguous"],
        "class_distribution": {key: counts[key] for key in ("not_exercise", "exercise")},
    }
    write_json(output_dir / "ggs_batch_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not all_rows:
        raise ValueError("Alle Datensaetze schlugen fehl. Siehe ggs_batch_summary.json.")
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output_dir / "ggs_combined_training_dataset.csv", index=False)
    if combined.empty:
        raise ValueError("Keine eindeutigen Trainingssegmente vorhanden.")
    return combined, summary


# ==============================================================================
# 7. RANDOM FOREST TRAINING, EVALUIERUNG & EXPORT
# ==============================================================================


def split_training_data_by_dataset(
    rows: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Teilt die Trainingsdaten auf Aufnahme-Ebene in Train- und Test-Sets auf (Group Shuffle Split).
    
    Stellt sicher, dass alle Segmente einer physischen Aufnahme entweder vollstaendig im
    Trainings- oder im Testset landen, um 'Recording Leakage' (Datenleckage) zu verhindern.
    
    Parameter:
    ----------
    rows : pd.DataFrame
        Tabelle aller extrahierten Segmente mit Spalte 'dataset_name'.
    test_size : float
        Anteil der Aufnahmen fuer das Testset (Standard: TEST_SIZE = 0.25).
    random_state : int
        Zufallssamen fuer reproduzierbare Aufteilung.
        
    Rueckgabe:
    ----------
    train : pd.DataFrame
        Segmente fuer das Modelltraining.
    test : pd.DataFrame
        Segmente fuer die Hold-Out-Evaluierung.
    """
    if rows["dataset_name"].nunique() < 2:
        raise ValueError("Mindestens zwei verschiedene Aufnahmen erforderlich fuer Train/Test-Split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(rows, groups=rows["dataset_name"]))
    train, test = rows.iloc[train_index].copy(), rows.iloc[test_index].copy()
    train_names = sorted(train["dataset_name"].unique())
    test_names = sorted(test["dataset_name"].unique())
    assert set(train_names).isdisjoint(set(test_names)), "Aufnahme-Leckage im Train/Test-Split erkannt!"
    print("Trainings-Datensaetze:", train_names)
    print("Test-Datensaetze:     ", test_names)
    return train, test


def classifier_inputs(rows: pd.DataFrame, feature_columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """
    Extrahiert und validiert die Merkmalsmatrix fuer das Random-Forest-Modell.
    
    Prueft, ob alle angeforderten Spalten mit der globalen Feature-Whitelist
    (FEATURE_COLUMNS) uebereinstimmen und nur endliche numerische Werte enthalten.
    
    Parameter:
    ----------
    rows : pd.DataFrame
        Segment-Tabelle mit den berechneten Merkmalen.
    feature_columns : tuple[str, ...] | list[str]
        Liste der zu verwendenden Merkmalsspalten.
        
    Rueckgabe:
    ----------
    predictors : pd.DataFrame
        Bereinigte Merkmalsmatrix (X) als Float-Werte.
    """
    if list(feature_columns) != FEATURE_COLUMNS:
        raise ValueError("Merkmals-Schema stimmt nicht mit der FEATURE_COLUMNS Whitelist ueberein.")
    predictors = rows.loc[:, feature_columns].astype(float)
    if not np.isfinite(predictors.to_numpy()).all():
        raise ValueError("Klassifikator-Merkmale muessen endliche numerische Werte sein (kein NaN/Inf).")
    return predictors


def train_random_forest_classifier(
    rows: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    parameters: dict | None = None,
) -> RandomForestClassifier:
    """
    Trainiert einen Random Forest Klassifikator auf den Segmentmerkmalen.
    
    Klassifiziert Segmente anhand ihrer Merkmale in Uebung (1, 'exercise')
    oder Pause/Nicht-Uebung (0, 'not_exercise').
    
    Parameter:
    ----------
    rows : pd.DataFrame
        Trainingsdaten mit Merkmalsspalten und 'ground_truth_binary' (0 oder 1).
    feature_columns : tuple[str, ...]
        Zu verwendende Merkmalsspalten (Standard: FEATURE_COLUMNS).
    parameters : dict, optional
        Hyperparameter fuer den RandomForestClassifier (Standard: RANDOM_FOREST_PARAMETERS).
        
    Rueckgabe:
    ----------
    model : RandomForestClassifier
        Das trainierte scikit-learn Modell.
    """
    if rows.empty or not rows["ground_truth_binary"].isin([0, 1]).all():
        raise ValueError("Training erfordert nicht-leere Segmente mit binaerem Ground Truth (0 oder 1).")
    model = RandomForestClassifier(
        **(RANDOM_FOREST_PARAMETERS if parameters is None else parameters)
    )
    model.fit(classifier_inputs(rows, feature_columns), rows["ground_truth_binary"].astype(int))
    return model


def predict_exercise_segments_with_model(
    segment_features: pd.DataFrame,
    model: RandomForestClassifier,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Wendet das trainierte Random Forest Modell auf extrahierte Segmente an.
    
    Fuegt der Tabelle zwei neue Spalten hinzu:
    - 'predicted_exercise': Boolean (True bei Vorhersage Klasse 1 / Uebung)
    - 'exercise_probability': Float (Vorhersagewahrscheinlichkeit fuer Klasse 1)
    
    Parameter:
    ----------
    segment_features : pd.DataFrame
        Tabelle der Segmentmerkmale.
    model : RandomForestClassifier
        Das trainierte Klassifikationsmodell.
    feature_columns : tuple[str, ...]
        Liste der Merkmalsspalten.
        
    Rueckgabe:
    ----------
    result : pd.DataFrame
        Kopie der Segmenttabelle inklusive Vorhersagen und Wahrscheinlichkeiten.
    """
    result = segment_features.copy()
    predictors = classifier_inputs(result, feature_columns)
    if result.empty:
        result["predicted_exercise"] = pd.Series(index=result.index, dtype=bool)
        result["exercise_probability"] = pd.Series(index=result.index, dtype=float)
        return result
    result["predicted_exercise"] = model.predict(predictors) == 1
    probabilities = model.predict_proba(predictors)
    exercise_column = np.flatnonzero(model.classes_ == 1)
    result["exercise_probability"] = (
        probabilities[:, exercise_column[0]] if len(exercise_column) else 0.0
    )
    return result


def export_test_predictions_to_label_studio(
    test_predictions: pd.DataFrame | str | Path,
    output_dir: Path | str = OUTPUT_DIR,
    label_studio_base_url: str = LABEL_STUDIO_BASE_URL,
    dataset_csv_urls: dict[str, str] | None = None,
) -> list[dict]:
    """
    Exportiert bestehende Vorhersagen des Testsets in das Label Studio JSON-Format.
    
    Ermoeglicht die visuelle Inspektion der Modellergebnisse im Vergleich zu
    den manuell annotierten Ground-Truth-Intervallen in Label Studio.
    
    Parameter:
    ----------
    test_predictions : pd.DataFrame | str | Path
        Tabelle oder Pfad zur CSV mit Testvorhersagen.
    output_dir : Path | str
        Zielverzeichnis fuer die JSON-Dateien.
    label_studio_base_url : str
        Basis-URL des Label Studio Webservers fuer CSV-Verknuepfungen.
    dataset_csv_urls : dict, optional
        Optionales Mapping von dataset_name auf existierende CSV-URLs.
        
    Rueckgabe:
    ----------
    exports : list[dict]
        Metadaten ueber die exportierten JSON-Vorhersagedateien.
    """
    rows = (pd.read_csv(test_predictions) if isinstance(test_predictions, (str, Path))
            else test_predictions.copy())
    required = {"dataset_name", "start", "end", "predicted_exercise"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Fehlende Spalten in Testvorhersagen: {sorted(required - set(rows.columns))}")
    if rows["dataset_name"].isna().any():
        raise ValueError("Testvorhersagen enthalten fehlende Datensatz-Namen (NaN).")
    decisions = rows["predicted_exercise"].astype(str).str.strip().str.casefold().map({
        "true": True, "false": False, "1": True, "0": False,
    })
    if decisions.isna().any():
        raise ValueError("predicted_exercise muss gueltige boolesche Werte enthalten.")
    rows["predicted_exercise"] = decisions
    output_dir = Path(output_dir)
    export_dir = output_dir / "random_forest_test_label_studio_predictions"
    export_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for name, group in rows.groupby("dataset_name", sort=True):
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Ungueltiger dataset_name: {name!r}")
        if dataset_csv_urls is not None:
            csv_url = dataset_csv_urls[name]
        else:
            prepared_name = f"{name}_prepared_for_ggs.csv"
            candidates = [path for path in output_dir.glob("*_prepared_for_ggs.csv")
                          if normalize_label_studio_filename(path.name) == prepared_name]
            if len(candidates) != 1:
                raise ValueError(f"Genau eine aufbereitete CSV fuer {name} erwartet; {len(candidates)} gefunden.")
            csv_url = f"{label_studio_base_url.rstrip('/')}/{quote(candidates[0].name)}"
        selected = group.loc[group["predicted_exercise"], ["start", "end"]]
        indices = selected.to_numpy(dtype=float)
        if (not np.isfinite(indices).all() or (indices < 0).any()
                or (indices != np.floor(indices)).any()
                or (indices[:, 1] <= indices[:, 0]).any()):
            raise ValueError(f"Ungueltige Sample-Intervalle fuer {name}.")
        intervals = [(int(row.start), int(row.end)) for row in selected.itertuples(index=False)]
        path = export_dir / f"{name}_random_forest_test_predictions.json"
        export_exercise_segments_to_label_studio_json(
            intervals, path, csv_url, label_name="RF Prediction",
            model_version="ggs_random_forest_holdout_test",
        )
        with path.open(encoding="utf-8") as file:
            task = json.load(file)
        task[0]["data"]["dataset_name"] = name
        write_json(path, task)
        print(f"{name}: {len(intervals)} vorhergesagte Uebungsintervalle -> {path.resolve()}")
        exports.append({"dataset_name": name, "path": str(path.resolve()),
                        "number_of_intervals": len(intervals)})
    return exports


def evaluate_classifier(
    model: RandomForestClassifier,
    test_rows: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict:
    """
    Umfassende Evaluierung des trainierten Random Forest Modells auf dem Hold-Out-Testset.
    
    Berechnet:
    - Classification Report (Precision, Recall, F1-Score pro Klasse)
    - Confusion Matrix (Wahrheiten vs. Vorhersagen)
    - Accuracy und Balanced Accuracy
    - Feature Importances (Wichtigkeit der einzelnen Signalmerkmale)
    
    Speichert alle Metriken, Vorhersagen und Feature-Importances im Output-Verzeichnis.
    
    Parameter:
    ----------
    model : RandomForestClassifier
        Das zu evaluierende Modell.
    test_rows : pd.DataFrame
        Hold-Out-Testsegmente mit Ground Truth.
    output_dir : Path | str
        Ausgabeverzeichnis fuer Metriken und CSV-Reports.
    feature_columns : tuple[str, ...]
        Verwendete Merkmalsspalten.
        
    Rueckgabe:
    ----------
    metrics : dict
        Dictionary mit allen berechneten Evaluierungsmetriken.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted = predict_exercise_segments_with_model(test_rows, model, feature_columns)
    y_true = test_rows["ground_truth_binary"].astype(int)
    y_pred = predicted["predicted_exercise"].astype(int)
    names = ["not_exercise", "exercise"]
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "classification_report": report,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "exercise_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": matrix.tolist(), "class_order": names,
        "test_datasets": sorted(test_rows["dataset_name"].unique()),
        "test_contains_both_classes": y_true.nunique() == 2,
        "evaluation_scope": "Nicht-mehrdeutige GGS-Segmente in Hold-Out-Aufnahmen",
    }
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=names, zero_division=0))
    print("Konfusionsmatrix (Zeilen=Wahrheit, Spalten=Vorhersage):\n", matrix)
    print(f"Accuracy: {metrics['accuracy']:.4f}; Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    if y_true.nunique() < 2:
        print("Hinweis: Testsplit enthaelt nur eine Klasse; Metriken fuer abwesende Klasse nicht aussagekraeftig.")
    importance = pd.DataFrame({
        "feature": feature_columns, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature-Wichtigkeit (Random Forest):")
    print(importance.to_string(index=False))
    write_json(output_dir / "ggs_random_forest_metrics.json", metrics)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        output_dir / "ggs_random_forest_confusion_matrix.csv", index_label="true_label"
    )
    importance.to_csv(output_dir / "ggs_random_forest_feature_importance.csv", index=False)
    predicted.to_csv(output_dir / "ggs_random_forest_test_predictions.csv", index=False)
    export_test_predictions_to_label_studio(predicted, output_dir)
    return metrics


def run_group_cross_validation(
    rows: pd.DataFrame,
    n_splits: int = N_GROUP_FOLDS,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    parameters: dict | None = None,
) -> dict:
    """
    Fuehrt eine aufnahmebasierte Kreuzvalidierung (GroupKFold) auf den Trainingsdaten durch.
    
    Validiert die Generalisierungsfaehigkeit des Modells ueber verschiedene Aufnahmen hinweg,
    ohne Daten aus dem Hold-Out-Testset zu beruehren.
    
    Parameter:
    ----------
    rows : pd.DataFrame
        Trainingsdaten mit 'dataset_name' als Gruppierungsmerkmal.
    n_splits : int
        Anzahl der Folds (Standard: N_GROUP_FOLDS = 5).
    feature_columns : tuple[str, ...]
        Verwendete Merkmale.
    parameters : dict, optional
        Hyperparameter fuer den Random Forest.
        
    Rueckgabe:
    ----------
    result : dict
        Ergebnisse der einzelnen Folds sowie Mittelwert und Standardabweichung des F1-Scores.
    """
    if n_splits < 2:
        raise ValueError("N_GROUP_FOLDS muss mindestens 2 betragen.")
    folds = min(n_splits, rows["dataset_name"].nunique())
    if folds < 2:
        print("Group CV uebersprungen: weniger als zwei Trainingsaufnahmen verfuegbar.")
        return {"folds": [], "mean_exercise_f1": None, "std_exercise_f1": None}
    if folds != n_splits:
        print(f"Group CV auf {folds} Folds reduziert (begrenzt durch verfuegbare Aufnahmen).")
    results = []
    for number, (train_index, val_index) in enumerate(
        GroupKFold(n_splits=folds).split(rows, groups=rows["dataset_name"]), 1
    ):
        train, validation = rows.iloc[train_index], rows.iloc[val_index]
        train_names = sorted(train["dataset_name"].unique())
        val_names = sorted(validation["dataset_name"].unique())
        assert set(train_names).isdisjoint(val_names), "Aufnahme-Leckage in der Kreuzvalidierung!"
        model = train_random_forest_classifier(train, feature_columns, parameters)
        predictions = predict_exercise_segments_with_model(validation, model, feature_columns)
        score = float(f1_score(
            validation["ground_truth_binary"], predictions["predicted_exercise"], zero_division=0
        ))
        results.append({
            "fold": number, "exercise_f1": score, "train_datasets": train_names,
            "validation_datasets": val_names,
            "train_contains_both_classes": train["ground_truth_binary"].nunique() == 2,
            "validation_contains_both_classes": validation["ground_truth_binary"].nunique() == 2,
        })
        print(f"Fold {number}/{folds}: Uebungs-F1={score:.4f}")
        if train["ground_truth_binary"].nunique() < 2 or validation["ground_truth_binary"].nunique() < 2:
            print("  Hinweis: Eine Klasse fehlt im Train- oder Validierungsset dieses Folds.")
    scores = [result["exercise_f1"] for result in results]
    result = {
        "scope": "Nur Trainings-Partition", "folds": results,
        "mean_exercise_f1": float(np.mean(scores)), "std_exercise_f1": float(np.std(scores)),
    }
    print(f"Group CV Uebungs-F1: {result['mean_exercise_f1']:.4f} +/- {result['std_exercise_f1']:.4f}")
    return result


def save_classifier(
    model: RandomForestClassifier,
    rows: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
    metadata: dict | None = None,
) -> None:
    """
    Speichert das trainierte Modell und umfassende Metadaten auf der Festplatte.
    
    Erzeugt:
    - 'ggs_exercise_classifier.joblib': Serialisiertes scikit-learn Modell
    - 'ggs_exercise_classifier_metadata.json': Verwendete Features, Hyperparameter,
      Klassenverteilungen und Vorverarbeitungs-Konfiguration.
    
    Parameter:
    ----------
    model : RandomForestClassifier
        Das trainierte Modell.
    rows : pd.DataFrame
        Fuer das Training verwendete Segmente.
    output_dir : Path | str
        Zielverzeichnis fuer Modell und Metadaten.
    metadata : dict, optional
        Zusaetzliche projektspezifische Metadaten.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    information = {
        "feature_columns": FEATURE_COLUMNS,
        "random_forest_parameters": model.get_params(),
        "number_of_datasets": int(rows["dataset_name"].nunique()),
        "number_of_training_segments": len(rows),
        "class_distribution": {
            str(int(label)): int(count)
            for label, count in rows["ground_truth_binary"].value_counts().items()
        },
        "preprocessing_parameters": PREPROCESSING_PARAMETERS,
        "fixed_preprocessing_parameters": {"median_kernel_size": 3, "max_missing_seconds": 10},
        "ggs_parameters": GGS_PARAMETERS, "ggs_columns": GGS_COLS,
        "ground_truth_thresholds": GROUND_TRUTH_THRESHOLDS,
        "exercise_label_values": EXERCISE_LABEL_VALUES,
        "sample_interval_convention": "[start, end), vorbereitete CSV Sample-Indizes",
        "model_scope": "Finales Modell trainiert auf allen nicht-mehrdeutigen Segmenten",
    }
    if metadata:
        information.update(metadata)
    joblib.dump(model, output_dir / "ggs_exercise_classifier.joblib")
    write_json(output_dir / "ggs_exercise_classifier_metadata.json", information)


def train_pipeline(
    csv_dir: Path | str,
    output_dir: Path | str,
    ground_truth_column: str,
) -> RandomForestClassifier:
    """
    End-to-End Trainingspipeline fuer das GGS + Random Forest Modell.
    
    Ablauf:
    1. Extrahiert Segmente und berechnet Merkmale fuer alle Aufnahmen in csv_dir
    2. Teilt Aufnahmen in Train- und Test-Partitionen auf (GroupShuffleSplit)
    3. Trainiert ein Zwischenmodell auf Train und evaluiert auf Test
    4. Fuehrt optional eine Group-K-Fold Kreuzvalidierung durch
    5. Trainiert das finale Modell auf allen gueltigen Segmenten und serialisiert es
    
    Parameter:
    ----------
    csv_dir : Path | str
        Verzeichnis mit gelabelten Sensor-CSVs.
    output_dir : Path | str
        Ausgabeverzeichnis fuer Modelle, Metriken und Segmente.
    ground_truth_column : str
        Name der Ground-Truth-Spalte (z. B. 'label').
        
    Rueckgabe:
    ----------
    final_model : RandomForestClassifier
        Das finale, auf allen Daten trainierte Modell.
    """
    if not ground_truth_column:
        raise ValueError("Setze GROUND_TRUTH_COLUMN oder uebergib --gt-column mit dem Namen der Label-Spalte.")
    rows, summary = build_combined_training_dataset(csv_dir, output_dir, ground_truth_column)
    train, test = split_training_data_by_dataset(rows)
    if train["ground_truth_binary"].nunique() < 2:
        raise ValueError("Trainingspartition enthaelt nur eine Klasse. Mehr gelabelte Aufnahmen benoetigt.")
    split = {
        "train_datasets": sorted(train["dataset_name"].unique()),
        "test_datasets": sorted(test["dataset_name"].unique()),
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
    }
    write_json(Path(output_dir) / "ggs_dataset_split.json", split)
    model = train_random_forest_classifier(train)
    evaluate_classifier(model, test, output_dir)
    if RUN_GROUP_CROSS_VALIDATION:
        cv_results = run_group_cross_validation(train)
        write_json(Path(output_dir) / "ggs_group_cross_validation.json", cv_results)
    final_model = train_random_forest_classifier(rows)
    save_classifier(final_model, rows, output_dir, {
        "ground_truth_column": ground_truth_column, "dataset_split": split,
        "datasets_processed_successfully": summary["datasets_processed"],
        "ground_truth_csv_dir": str(Path(csv_dir).resolve()),
    })
    print(f"Finales Modell erfolgreich gespeichert: {Path(output_dir) / 'ggs_exercise_classifier.joblib'}")
    return final_model


# ==============================================================================
# 8. INFERENZ & BENCHMARK-ADAPTER
# ==============================================================================


def predict_unlabeled_dataset(
    csv_path: Path | str,
    model_path: Path | str,
    output_dir: Path | str = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Segmentiert und klassifiziert eine ungelabelte Sensordatei mit einem vortrainierten Modell.
    
    Ablauf:
    1. Laedt das trainierte Random Forest Modell und die zugehoerigen Metadaten (.json)
    2. Fuehrt Signalvorverarbeitung und GGS-Segmentierung durch
    3. Extrahiert Merkmale fuer alle Segmente
    4. Klassifiziert Segmente (Uebung vs. Pause) via Random Forest
    5. Exportiert Ergebnisse als CSV und im Label Studio JSON-Format
    
    Parameter:
    ----------
    csv_path : Path | str
        Pfad zur Rohdaten-CSV.
    model_path : Path | str
        Pfad zur serialisierten Modell-Datei (ggs_exercise_classifier.joblib).
    output_dir : Path | str
        Ausgabeverzeichnis fuer Vorhersage-CSV und Label Studio Export.
        
    Rueckgabe:
    ----------
    predicted : pd.DataFrame
        Segment-Tabelle inklusive Spalten 'predicted_exercise' und 'exercise_probability'.
    """
    model_path = Path(model_path)
    metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata["ggs_columns"] != GGS_COLS:
        raise ValueError("GGS-Spalten des gespeicherten Modells stimmen nicht mit der aktuellen Pipeline ueberein.")
    model = joblib.load(model_path)
    features, k, _, csv_url = run_dataset_segmentation(
        csv_path, output_dir, metadata["ggs_parameters"], metadata["preprocessing_parameters"]
    )
    predicted = predict_exercise_segments_with_model(features, model, metadata["feature_columns"])
    # Konvertierung der Modellaussagen fuer die bestehenden Intervall- und Exporthilfen
    intervals = get_predicted_exercise_segments(
        predicted.assign(predicted_exercise_set=predicted["predicted_exercise"])
    )
    prefix = Path(output_dir) / f"{Path(csv_path).stem}_ggs_random_forest"
    predicted.to_csv(f"{prefix}_segments_k_{k}.csv", index=False)
    export_exercise_segments_to_label_studio_json(
        intervals, f"{prefix}_predictions_k_{k}.json", csv_url,
        model_version=f"ggs_random_forest_k_{k}",
    )
    print("Random Forest Uebungsintervalle:", intervals)
    return predicted


def main():
    """
    Kommandozeilenschnittstelle (CLI) fuer Training, Vorhersage und Label-Studio-Export.
    
    Subkommandos:
    - 'train':       Fuehrt die komplette Trainings-Pipeline aus
    - 'predict':     Klassifiziert eine einzelne ungelabelte CSV-Datei (--csv benoetigt)
    - 'export-test': Exportiert Hold-Out-Testvorhersagen nach Label Studio
    """
    parser = argparse.ArgumentParser(description="GGS segment-level supervised exercise classifier")
    parser.add_argument("mode", nargs="?", choices=["train", "predict", "export-test"], default="train")
    parser.add_argument("--csv-dir", type=Path, default=GROUND_TRUTH_CSV_DIR)
    parser.add_argument("--gt-column", default=GROUND_TRUTH_COLUMN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--csv", type=Path, help="Neue ungelabelte CSV fuer den predict-Modus")
    parser.add_argument("--model", type=Path, default=OUTPUT_DIR / "ggs_exercise_classifier.joblib")
    parser.add_argument("--test-predictions", type=Path, help="Gespeicherte Test-Vorhersage-CSV fuer export-test")
    args = parser.parse_args()
    try:
        if args.mode == "train":
            train_pipeline(args.csv_dir, args.output_dir, args.gt_column)
        elif args.mode == "export-test":
            export_test_predictions_to_label_studio(
                args.test_predictions or args.output_dir / "ggs_random_forest_test_predictions.csv",
                args.output_dir,
            )
        else:
            if args.csv is None:
                parser.error("Modus 'predict' erfordert das Argument --csv")
            predict_unlabeled_dataset(args.csv, args.model, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Fehler: {exc}\n")


# ------------------------------------------------------------------------------
# Framework-Adapter: Integration in Benchmark & Batch-Verarbeitung (run_methods)
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
INPUT_FOLDER = HERE / "InputCSV"
OUTPUT_FOLDER_NAME = "ggs_output"
OUTPUT_FOLDER = HERE / OUTPUT_FOLDER_NAME
MODEL_PATH = OUTPUT_DIR / "ggs_exercise_classifier.joblib"


def predict(rows: list[dict]) -> list[bool]:
    """
    Benchmark-Adapter: Weist jedem Abtastpunkt der Rohdaten ein boolesches Label zu.
    
    Verwendet das vortrainierte GGS + Random Forest Modell:
    1. Schreibt Zeilen in eine isolierte temporaere CSV-Datei
    2. Fuehrt Vorverarbeitung, GGS-Segmentierung und RF-Klassifikation durch
    3. Weist jedem Sample im Zeitfenster das Segment-Label zu
    
    Voraussetzung:
    --------------
    Das trainierte Modell unter MODEL_PATH und die dazugehoerige JSON-Metadaten-Datei
    muessen existieren (vorab im Modus 'train' erzeugt).
    
    Parameter:
    ----------
    rows : list[dict]
        Liste von Rohdaten-Zeilen (Dictionaries mit Sensorwerten).
        
    Rueckgabe:
    ----------
    labels : list[bool]
        Liste von Booleans (True = 'exercising', False = 'pause') exakt in Laenge von 'rows'.
    """
    if not rows:
        return []
    with tempfile.TemporaryDirectory(prefix="ggs_benchmark_") as directory:
        temporary = Path(directory)
        csv_path = temporary / "recording.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        predicted = predict_unlabeled_dataset(csv_path, MODEL_PATH, temporary)
        prepared = pd.read_csv(temporary / "recording_prepared_for_ggs.csv")
        if len(prepared) != len(rows):
            raise ValueError(
                "GGS-Vorverarbeitung hat Samples entfernt. Zuweisung 1:1 auf urspruengliche "
                "Zeilen nicht moeglich; Datensatz vorab bereinigen/ausrichten."
            )
        labels = [False] * len(rows)
        for segment in predicted.itertuples(index=False):
            start, end = int(segment.start), int(segment.end)
            if not 0 <= start < end <= len(rows):
                raise ValueError("GGS lieferte ein Segment ausserhalb des gueltigen Indexbereichs.")
            labels[start:end] = [bool(segment.predicted_exercise)] * (end - start)
        return labels


def ggs_method():
    """
    Batch-Verarbeitung fuer das Benchmark-Framework (kompatibel mit run_methods.py).
    
    Verarbeitet alle CSV-Dateien im Verzeichnis 'InputCSV/', fuehrt fuer jede Aufnahme
    die GGS+RF-Vorhersage durch und schreibt das Ergebnis mit einer 'label'-Spalte
    (exercising/pause) nach 'ggs_output/'.
    
    Enthaelt Fortschrittsanzeige auf der Konsole und Zeitmessung pro Datensatz.
    """
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    input_csvs = sorted(INPUT_FOLDER.glob("*.csv"))
    if not input_csvs:
        raise RuntimeError(f"Keine CSV-Dateien in {INPUT_FOLDER} gefunden!")

    print(f"Starte GGS-Segmentierung fuer {len(input_csvs)} Dateien in '{INPUT_FOLDER.name}'...\n")

    for index, input_path in enumerate(input_csvs, 1):
        t0 = time.time()
        print(f"[{index}/{len(input_csvs)}] Verarbeite: {input_path.name}")
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(65536)
            file.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            reader = csv.DictReader(file, dialect=dialect)
            rows = list(reader)
            source_fields = list(reader.fieldnames or [])

        # 1. Vorhersage ueber predict()-Methode (GGS + Random Forest)
        labels = list(predict(rows))
        if len(labels) != len(rows):
            raise RuntimeError(
                f"{input_path.name}: predict() lieferte {len(labels)} Labels fuer {len(rows)} Zeilen"
            )

        # 2. Output CSV mit 'label'-Spalte schreiben
        output_path = OUTPUT_FOLDER / input_path.name
        partial_path = output_path.with_name(output_path.name + ".partial")
        fields = [name for name in source_fields if name != "label"] + ["label"]
        with partial_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=fields, dialect=dialect, extrasaction="ignore"
            )
            writer.writeheader()
            for row, exercising in zip(rows, labels):
                row["label"] = "exercising" if exercising else "pause"
                writer.writerow(row)
        os.replace(partial_path, output_path)

        t1 = time.time()
        print(f"   -> Gespeichert in: {output_path.name} (Dauer: {t1 - t0:.2f}s)\n")

    print("Alle Dateien wurden erfolgreich mit GGS verarbeitet!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        ggs_method()
    else:
        main()
