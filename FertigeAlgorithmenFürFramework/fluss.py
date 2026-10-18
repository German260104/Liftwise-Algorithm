"""FLUSS-based exercise detector adapted to method_template.py.txt."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, correlate, find_peaks, savgol_filter
from scipy.spatial.distance import cdist

try:
    import matplotlib
    matplotlib.use("Agg")  # Verhindert Popups, rendert rein in Bilddateien
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable



# =============================================================================
# ORDNER UND DATEIPFAD KONFIGURATION
# =============================================================================

HERE = Path(__file__).resolve().parent
INPUT_FOLDER = HERE / "InputCSV"

# This must be unique and must end in _output.
OUTPUT_FOLDER_NAME = "fluss_output"
OUTPUT_FOLDER = HERE / OUTPUT_FOLDER_NAME
OUTPUT_PLOT_FOLDER_NAME = "fluss_plots"
OUTPUT_PLOT_FOLDER = HERE / OUTPUT_PLOT_FOLDER_NAME


# =============================================================================
# HYPERPARAMETER
# =============================================================================

# Savgol Filter Parameter
WINDOW_LENGTH = 15
POLYORDER = 3

# Länge der Subsequenzen
SUBSEQUENCE_LENGTH = 200

# Schrittweite zwischen Subsequenzen
STEP_SIZE = 40

# Maximale Toleranz zwischen zwei erkannten Grenzen, welche zu einer semantischen Grenze gehören
TOLERANCE = 90

# Anteil der Dimensionen, in welchen eine semantische Grenze vorhanden sein muss
PERCENTAGE = 0.2

# Wie viele Zeilen schreibt der Sensor pro Sekunde?
SAMPLING_RATE_HZ = 100

# Maximale Dauer einer Übungseinheit in Sekunden
MAX_EXERCISE_DURATION_S = 50

IGNORE_COLUMNS = [
    "timestamp_s",
    "acc_earth_y",
    "acc_earth_x",
    "acc_earth_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "altitude_m",
    "label",
]

# windowing (ReCoFit-style)
WINDOW_SIZE_SECONDS = 5
STEP_SIZE_SECONDS = 0.2
WINDOW_SIZE = int(WINDOW_SIZE_SECONDS * SAMPLING_RATE_HZ)  # 500 samples

# lowpass filter
USE_FILTER = True
FILTER_CUTOFF_HZ = 10
FILTER_ORDER = 4

DECISION_THRESHOLD = 0.3
DECISION_MIN_DURATION_SECONDS = 10

# autocorrelation
AUTOCORR_MAX_LAG_SECONDS = 4
AUTOCORR_MAX_LAG = int(AUTOCORR_MAX_LAG_SECONDS * SAMPLING_RATE_HZ)
PEAK_THRESHOLD_RATIO = 0.3
AUTOCORR_MIN_LAG_SECONDS = 0.1
AUTOCORR_MIN_LAG = int(AUTOCORR_MIN_LAG_SECONDS * SAMPLING_RATE_HZ)

# Plotting Steuerung (speichert Uebersichts-PNGs in fluss_plots/)
SAVE_PLOTS = True


# =============================================================================
# VORVERARBEITUNG & FILTER
# =============================================================================

def butter_lowpass(
    signal, fs=SAMPLING_RATE_HZ, cutoff=FILTER_CUTOFF_HZ, order=FILTER_ORDER
):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def preprocess(df):
    df = df.copy().interpolate(method="linear").bfill().ffill().fillna(0.0)

    if USE_FILTER:
        sensor_cols = [
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "acc_earth_x",
            "acc_earth_y",
            "acc_earth_z",
        ]

        for col in sensor_cols:
            if col in df.columns:
                df[col] = butter_lowpass(df[col].values)

    # magnitudes (core representation)
    if all(c in df.columns for c in ["acc_x", "acc_y", "acc_z"]):
        df["acc_magnitude"] = np.sqrt(
            df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
        )
    else:
        df["acc_magnitude"] = 0.0

    if all(c in df.columns for c in ["gyro_x", "gyro_y", "gyro_z"]):
        df["gyro_magnitude"] = np.sqrt(
            df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
        )
    else:
        df["gyro_magnitude"] = 0.0

    return df


def autocorr(x):
    x = x - np.mean(x)
    c = correlate(x, x, mode="full")
    c = c[len(c) // 2 :]
    max_c = np.max(c)
    if max_c > 0:
        c = c / (max_c + 1e-8)
    return c


def autocorr_features(x):
    ac = autocorr(x)[AUTOCORR_MIN_LAG:AUTOCORR_MAX_LAG]

    if len(ac) < 10:
        return 0.0, 0, 0

    peaks = np.where((ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]))[0] + 1
    vals = ac[peaks] if len(peaks) > 0 else np.array([])

    max_val = float(np.max(ac))

    if len(vals) == 0:
        return max_val, 0, 0

    thresh = PEAK_THRESHOLD_RATIO * max_val

    prom = int(np.sum(vals > thresh))
    weak = int(np.sum(vals <= thresh))

    return max_val, prom, weak


def window_features(df):
    rows = []
    total = len(df)

    for start in range(0, total - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        w = df.iloc[start:end]

        acc = w["acc_magnitude"].values
        acc_ac = autocorr_features(acc)

        center_idx = WINDOW_SIZE // 2
        rows.append(
            {
                "timestamp_s": start + center_idx,
                "acc_magnitude": float(acc[center_idx]),
                "acc_ac_max": acc_ac[0],
            }
        )

    return pd.DataFrame(rows)


def apply_decision_logic(
    features,
    threshold=DECISION_THRESHOLD,
    min_duration_seconds=DECISION_MIN_DURATION_SECONDS,
):
    if features.empty:
        features = pd.DataFrame(columns=["timestamp_s", "acc_magnitude", "acc_ac_max", "decision"])
        return features

    min_windows = int(np.ceil(min_duration_seconds / STEP_SIZE_SECONDS))
    mask = (features["acc_ac_max"] >= threshold)
    decisions = np.zeros(len(mask), dtype=float)

    start = None
    for idx, active in enumerate(mask):
        if active:
            if start is None:
                start = idx
        else:
            if start is not None and idx - start >= min_windows:
                decisions[start:idx] = 1.0
            start = None

    if start is not None and len(mask) - start >= min_windows:
        decisions[start:] = 1.0

    features["decision"] = decisions
    return features


# =============================================================================
# FLUSS KERN-LOGIK
# =============================================================================

def get_sequence_borders(signal, col=""):
    """Berechnet semantische Kanten zwischen Übung und Pause mit FLUSS."""
    subsequences = [
        signal[i : i + SUBSEQUENCE_LENGTH]
        for i in range(0, len(signal) - SUBSEQUENCE_LENGTH + 1, STEP_SIZE)
    ]

    if len(subsequences) > 0 and len(subsequences[-1]) < SUBSEQUENCE_LENGTH:
        corrected_subsequences = subsequences[:-1]
    else:
        corrected_subsequences = subsequences

    if len(corrected_subsequences) < 2:
        return [0, len(signal) - 1]

    # Berechnet euklidische Distanz
    dist_matrix = cdist(
        corrected_subsequences, corrected_subsequences, metric="euclidean"
    )

    num_subsequences = len(dist_matrix)
    exclusion_zone = 2 * (SUBSEQUENCE_LENGTH // STEP_SIZE)
    for i in range(num_subsequences):
        start = max(0, i - exclusion_zone)
        end = min(num_subsequences, i + exclusion_zone + 1)
        dist_matrix[i, start:end] = np.inf

    indices = np.arange(num_subsequences)
    time_distances = np.abs(indices[:, None] - indices)

    # 1. Wie viele Datenpunkte entsprechen der maximalen Übungsdauer?
    max_duration_datapoints = MAX_EXERCISE_DURATION_S * SAMPLING_RATE_HZ

    # 2. Wie viele Subsequenz-Schritte (Matrix-Indizes) sind das?
    dyn_distance_penalty = max_duration_datapoints / STEP_SIZE
    dyn_distance_penalty = max(1.0, dyn_distance_penalty)

    valid_mask = ~np.isinf(dist_matrix)
    if not np.any(valid_mask):
        return [0, len(signal) - 1]

    avg_dist = float(np.mean(dist_matrix[valid_mask]))

    penalty_matrix = (
        (time_distances**2) / (2 * (dyn_distance_penalty**2)) * avg_dist
    )
    dist_matrix = dist_matrix + penalty_matrix

    # MPIndex
    closest_indices = np.argmin(dist_matrix, axis=1)

    del dist_matrix
    del penalty_matrix

    j_indices = np.arange(num_subsequences)[:, None]
    i_indices = np.arange(num_subsequences)[None, :]
    cj = closest_indices[:, None]

    cond1 = (j_indices < i_indices) & (i_indices < cj)
    cond2 = (j_indices > i_indices) & (i_indices > cj)

    arc_curve = np.sum(cond1 | cond2, axis=0)

    max_arc = np.max(arc_curve)
    if max_arc > 0:
        corrected_arc_curve = arc_curve / max_arc
    else:
        corrected_arc_curve = np.zeros(num_subsequences, dtype=float)

    inverted_cac = 1.0 - corrected_arc_curve
    inverted_cac = np.clip(inverted_cac, 0, 1)

    detected_subsequence_indices, _ = find_peaks(
        inverted_cac,
        prominence=0.15,
        distance=max(1, int((2 * SUBSEQUENCE_LENGTH) // STEP_SIZE)),
    )

    valid_peaks = []
    global_axis_std = np.std(signal)

    for peak in detected_subsequence_indices:
        peak_data_idx = peak * STEP_SIZE
        start_w = max(0, peak_data_idx - int(SAMPLING_RATE_HZ))
        end_w = min(len(signal), peak_data_idx + int(SAMPLING_RATE_HZ))
        local_movement = np.std(signal[start_w:end_w])

        if local_movement > (global_axis_std * 0.15):
            valid_peaks.append(peak)

    detected_subsequence_indices = [int(detsubind) for detsubind in valid_peaks]
    true_signal_borders = [0]
    true_signal_borders.extend(
        [trueind * STEP_SIZE for trueind in detected_subsequence_indices]
    )
    true_signal_borders.append(len(signal) - 1)

    return true_signal_borders


# =============================================================================
# PREDICTION & BENCHMARK METHODE
# =============================================================================

def predict(rows, return_plot_data=False):
    """Return one Boolean per input row: True=exercising, False=pause."""
    if not rows:
        if return_plot_data:
            return [], {}
        return []

    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(rows)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    total_samples = len(df)
    if total_samples < max(SUBSEQUENCE_LENGTH, WINDOW_SIZE):
        labels = [False] * total_samples
        if return_plot_data:
            return labels, {}
        return labels

    # Glättung der Daten mit savgol_filter
    df_smoothed = df.copy()
    df_smoothed = df_smoothed.interpolate(method="linear").bfill().ffill().fillna(0.0)
    numeric_cols = df_smoothed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in IGNORE_COLUMNS and col != "label":
            try:
                df_smoothed[col] = savgol_filter(
                    df_smoothed[col].values, window_length=WINDOW_LENGTH, polyorder=POLYORDER
                )
            except Exception:
                pass

    cac_border_list = []
    # Für alle Sensor-Spalten semantische Grenzen berechnen
    sensor_cols = [
        c for c in df_smoothed.columns
        if c not in IGNORE_COLUMNS and np.issubdtype(df_smoothed[c].dtype, np.number)
    ]
    print(f"   Analysiere {len(sensor_cols)} Achsen ({', '.join(sensor_cols)})...")
    for col in sensor_cols:
        borders = get_sequence_borders(df_smoothed[col].values, col=col)
        cac_border_list.append(borders)

    if not cac_border_list:
        labels = [False] * total_samples
        if return_plot_data:
            return labels, {}
        return labels

    sets = [set(p) for p in cac_border_list]
    simultaneous = []
    all_peaks = sorted(list(set.union(*sets)))
    mindest_treffer = len(sets) * PERCENTAGE

    for idx in all_peaks:
        treffer_anzahl = sum(1 for s in sets if any(abs(idx - o) <= TOLERANCE for o in s))
        if treffer_anzahl >= mindest_treffer:
            simultaneous.append(idx)

    simultaneous_clean = []
    if simultaneous:
        current_cluster = [simultaneous[0]]
        for p in simultaneous[1:]:
            if p - current_cluster[-1] <= TOLERANCE:
                current_cluster.append(p)
            else:
                simultaneous_clean.append(int(np.mean(current_cluster)))
                current_cluster = [p]
        simultaneous_clean.append(int(np.mean(current_cluster)))
    simultaneous = sorted(simultaneous_clean)

    # Magnitude als Feature erstellen
    if all(c in df_smoothed.columns for c in ["acc_x", "acc_y", "acc_z"]):
        df_smoothed["Mag"] = np.sqrt(
            df_smoothed["acc_x"] ** 2
            + df_smoothed["acc_y"] ** 2
            + df_smoothed["acc_z"] ** 2
        )

    df_prep = preprocess(df_smoothed)
    feature_df = window_features(df_prep)
    feature_df_dec = apply_decision_logic(feature_df)

    # Labels für alle Zeilen erzeugen
    labels = [False] * total_samples
    borders = sorted(list(set([0] + simultaneous + [total_samples])))

    for i in range(len(borders) - 1):
        start = borders[i]
        end = borders[i + 1]
        if start >= end:
            continue

        corrected_start = max(0, start // STEP_SIZE)
        corrected_end = min(len(feature_df_dec), max(corrected_start + 1, end // STEP_SIZE))

        segment_data = feature_df_dec.iloc[corrected_start:corrected_end]

        if not segment_data.empty and segment_data["decision"].mean() >= 0.3:
            for idx in range(start, min(end, total_samples)):
                labels[idx] = True

    exercise_count = sum(labels)
    exercise_pct = (exercise_count / total_samples * 100) if total_samples > 0 else 0
    print(f"   -> {len(borders) - 1} Segmente analysiert | {exercise_count}/{total_samples} Zeilen ({exercise_pct:.1f}%) als 'exercising'")

    if return_plot_data:
        plot_data = {
            "df": df_smoothed,
            "borders": borders,
            "feature_df_dec": feature_df_dec,
        }
        return labels, plot_data

    return labels


def fluss_method():
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    if SAVE_PLOTS:
        OUTPUT_PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    input_csvs = sorted(INPUT_FOLDER.glob("*.csv"))
    if not input_csvs:
        raise RuntimeError(f"No CSV files found in {INPUT_FOLDER}")

    print(f"Starte FLUSS-Segmentierung für {len(input_csvs)} Dateien in '{INPUT_FOLDER.name}'...\n")

    for idx, input_path in enumerate(input_csvs, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(input_csvs)}] Verarbeite: {input_path.name}")
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(65536)
            file.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            reader = csv.DictReader(file, dialect=dialect)
            rows = list(reader)
            source_fields = list(reader.fieldnames or [])

        labels, plot_data = predict(rows, return_plot_data=True)
        if len(labels) != len(rows):
            raise RuntimeError(
                f"{input_path.name}: predict() returned {len(labels)} labels "
                f"for {len(rows)} rows"
            )

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

        # Plot als PNG in fluss_plots/ speichern (ohne Pop-up-Fenster)
        if SAVE_PLOTS and plot_data:
            plot_path = OUTPUT_PLOT_FOLDER / f"{input_path.stem}_fluss_plot.png"
            save_fluss_plot(
                plot_data.get("df"),
                labels,
                plot_data.get("borders", []),
                plot_data.get("feature_df_dec"),
                plot_path,
            )
            print(f"   -> Plot: {plot_path.name}")

        t1 = time.time()
        print(f"   -> Gespeichert in: {output_path.name} (Dauer: {t1 - t0:.2f}s)\n")

    print("FLUSS: Alle Dateien wurden erfolgreich verarbeitet!")


# Aliasse für Flexibilität beim Importieren
fluss = fluss_method
your_method = fluss_method


# =============================================================================
# PLOTS (SPEICHERN IN fluss_plots/)
# =============================================================================

def save_fluss_plot(df, labels, borders, feature_df_dec, output_plot_path):
    """
    Speichert einen aussagekraeftigen Uebersichtsplot der FLUSS-Segmentierung als PNG.
    - Oben: Beschleunigungssignal (Magnitude) mit FLUSS-Segmentgrenzen und Uebungsschattierung
    - Unten: Autokorrelations-Feature (acc_ac_max) und Segmentierungsentscheidung mit Schwellenwert
    """
    if plt is None:
        return
    try:
        OUTPUT_PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
        total_samples = len(df)
        if total_samples == 0:
            return

        if "timestamp_s" in df.columns:
            try:
                timestamps = df["timestamp_s"].values.astype(float)
            except Exception:
                timestamps = np.arange(total_samples) / SAMPLING_RATE_HZ
        else:
            timestamps = np.arange(total_samples) / SAMPLING_RATE_HZ

        if "acc_magnitude" in df.columns:
            mag = df["acc_magnitude"].values
        elif all(c in df.columns for c in ["acc_x", "acc_y", "acc_z"]):
            mag = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
        else:
            mag = np.zeros(total_samples)

        # Uebungssegmente fuer Schattierung
        changes = np.diff(np.concatenate(([0], np.array(labels, dtype=int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        has_features = feature_df_dec is not None and not feature_df_dec.empty

        if has_features:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]}
            )
        else:
            fig, ax1 = plt.subplots(figsize=(15, 5))
            ax2 = None

        # --- Subplot 1: Sensorsignal + Grenzen + Uebungen ---
        ax1.plot(timestamps, mag, label="Beschleunigung (Magnitude)", color="#1f77b4", alpha=0.6, lw=1.0)

        # FLUSS-Schnittstellen als vertikale Linien
        for b_idx in borders:
            if 0 < b_idx < total_samples:
                ax1.axvline(timestamps[b_idx], color="#7f7f7f", linestyle=":", alpha=0.7, lw=1.0)

        # Uebungsschattierung
        for idx, (s, e) in enumerate(zip(starts, ends)):
            e_idx = min(total_samples - 1, e)
            ax1.axvspan(timestamps[s], timestamps[e_idx], color="#2ca02c", alpha=0.3, label="Erkannte Uebung" if idx == 0 else "")

        ax1.set_title(f"FLUSS-Segmentierung: {output_plot_path.stem}", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Signal Amplitude")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper right")

        # --- Subplot 2: Autokorrelation & Segmentierungsentscheidung ---
        if ax2 is not None:
            if "timestamp_s" in feature_df_dec.columns:
                feat_times = []
                for t_idx in feature_df_dec["timestamp_s"].values:
                    idx = int(np.clip(t_idx, 0, total_samples - 1))
                    feat_times.append(timestamps[idx])
                feat_times = np.array(feat_times)
            else:
                feat_times = np.linspace(timestamps[0], timestamps[-1], len(feature_df_dec))

            if "acc_ac_max" in feature_df_dec.columns:
                ax2.plot(feat_times, feature_df_dec["acc_ac_max"], label="Autokorrelation (acc_ac_max)", color="#ff7f0e", lw=1.5)
            if "decision" in feature_df_dec.columns:
                ax2.plot(feat_times, feature_df_dec["decision"], label="Entscheidung (1=Uebung, 0=Pause)", color="#d62728", lw=1.8, linestyle="--")

            ax2.axhline(DECISION_THRESHOLD, color="black", linestyle=":", alpha=0.7, label=f"Schwelle ({DECISION_THRESHOLD})")

            for s, e in zip(starts, ends):
                e_idx = min(total_samples - 1, e)
                ax2.axvspan(timestamps[s], timestamps[e_idx], color="#2ca02c", alpha=0.15)

            ax2.set_xlabel("Zeit (s)")
            ax2.set_ylabel("Score / Entscheidung")
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend(loc="upper right")
        else:
            ax1.set_xlabel("Zeit (s)")

        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"   Hinweis: Plot konnte nicht gespeichert werden ({e})")


# =============================================================================
# EINSTIEGSPUNKT
# =============================================================================

if __name__ == "__main__":
    fluss_method()