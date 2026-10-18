import pandas as pd
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import find_peaks
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False
import json
import os
import glob
import time
from pathlib import Path
import csv

# =============================================================================
# ORDNER UND DATEIPFAD KONFIGURATION
# =============================================================================

HERE = Path(__file__).resolve().parent
INPUT_FOLDER = HERE / "InputCSV"

OUTPUT_FOLDER_NAME = "fft_ac_hybrid_output"
OUTPUT_FOLDER = HERE / OUTPUT_FOLDER_NAME
OUTPUT_PLOT_FOLDER_NAME = "fft_ac_hybrid_plots"
OUTPUT_PLOT_FOLDER = HERE / OUTPUT_PLOT_FOLDER_NAME

# =============================================================================
# HYPERPARAMETER KONFIGURATION (HYBRID: AUTOKORRELATION + FFT)
# =============================================================================

# Grundlegendes
SAMPLING_RATE_HZ = 100             # Aufnahmefrequenz in Hertz

# 1. Frequenz- und Rhythmus-Grenzen für Krafttraining
# 0.15 Hz bis 0.85 Hz entspricht 1.2s bis 6.6s pro Wiederholung
FFT_MIN_FREQ_HZ = 0.15              # Minimale Wiederholungs-Frequenz (Hz)
FFT_MAX_FREQ_HZ = 0.85                 # Maximale Wiederholungs-Frequenz (Hz)
ACF_MIN_LAG_SEC = 0.8               # Minimale Wiederholungs-Dauer (Sekunden)
ACF_MAX_LAG_SEC = 7.0               # Maximale Wiederholungs-Dauer (Sekunden)

# 2. Hybrid-Schwellenwert
# Der Hybrid-Score ist das geometrische Mittel aus Formtreue (ACF) und spektraler Bündelung (FFT).
# Wertebereich: 0.0 bis 1.0 (über 0.50 = extrem hohe Sicherheit)
HYBRID_THRESHOLD = 0.45            # Mindest-Score für rhythmische Übung

# 3. Sliding Window
SLIDE_WINDOW_SEC = 10.0            # Betrachtetes Zeitfenster (Sekunden)
SLIDE_STEP_SEC = 1.0               # Schrittweite des Fensters (Sekunden)

# 4. Glättung & Segmentierung
MAX_PAUSE_GAP_SEC = 5.0            # Max. tolerierte Atempausen innerhalb eines Satzes (Sekunden)
MIN_CONSECUTIVE_SEC = 10.0         # Hybrid-Score muss min. so lange am Stück hoch sein (Spike-Schutz)
MIN_EXERCISE_DURATION_SEC = 8.0    # Mindestdauer eines fertigen Satzes (Sekunden)

# =============================================================================
# HYBRID KERN-LOGIK
# =============================================================================

def compute_hybrid_channel(sig, hanning_win, freqs, rep_mask, min_lag, max_lag):
    """
    Berechnet für eine einzelne Sensor-Achse sowohl den FFT- als auch den ACF-Score
    und kombiniert beide mathematisch.
    """
    x = sig - np.mean(sig)
    var = np.var(x)
    if var < 1e-4:
        return 0.0, 0.0, 0.0

    # 1. FFT Spektralanalyse
    fft_vals = np.abs(np.fft.rfft(x * hanning_win))
    psd = fft_vals ** 2
    norm_psd = psd / (np.sum(psd) + 1e-6)
    f_peaks, f_props = find_peaks(norm_psd[rep_mask], height=0.08, prominence=0.03)
    fft_score = np.max(f_props['peak_heights']) if len(f_peaks) > 0 else 0.0

    # 2. Autokorrelation (Form-Wiederholung)
    acf = np.correlate(x, x, mode='full')[len(x)-1:]
    if acf[0] == 0:
        return 0.0, 0.0, 0.0
    acf = acf / acf[0]
    
    a_peaks, a_props = find_peaks(acf[min_lag:max_lag], height=0.15, prominence=0.03)
    acf_score = np.max(a_props['peak_heights']) if len(a_peaks) > 0 else 0.0

    # 3. Hybrid-Kombination (Geometrisches Mittel aus ACF und skalierter FFT)
    fft_norm = min(1.0, fft_score / 0.35)
    hybrid_score = np.sqrt(acf_score * fft_norm)
    
    return float(hybrid_score), float(acf_score), float(fft_score)

def compute_max_hybrid_multichannel(window_data, hanning_win, freqs, rep_mask, min_lag, max_lag):
    """
    Scannt alle verfügbaren Achsen und synthetischen Kanäle
    (Acc, Gyro, Acc_Earth + jeweilige Magnituden) und nimmt den Kanal
    mit dem höchsten gemeinsamen Hybrid-Score.
    """
    best_hybrid = 0.0
    best_acf = 0.0
    best_fft = 0.0
    max_acf = 0.0
    max_fft = 0.0
    for ax in range(window_data.shape[1]):
        h, a, f = compute_hybrid_channel(window_data[:, ax], hanning_win, freqs, rep_mask, min_lag, max_lag)
        if a > max_acf:
            max_acf = a
        if f > max_fft:
            max_fft = f
        if h > best_hybrid:
            best_hybrid = h
            best_acf = a
            best_fft = f
    if best_hybrid == 0.0:
        best_acf = max_acf
        best_fft = max_fft
    return best_hybrid, best_acf, best_fft


def predict(rows, return_scores: bool = False):
    """
    Führt den Hybrid-Algorithmus (ACF + FFT) auf den übergebenen Zeilen aus.
    Gibt für jede Zeile einen Boolean zurück: True=exercising, False=pause.
    
    Args:
        rows: Liste von Dicts (CSV-Zeilen).
        return_scores: Wenn True, wird ein Tupel (labels, score_dict) zurückgegeben.
    """
    if not rows:
        if return_scores:
            return [], {}
        return []

    total_samples = len(rows)
    
    # 1. Relevante Spalten identifizieren & extrahieren
    first_row = rows[0]
    expected_cols = [
        'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'acc_earth_x', 'acc_earth_y', 'acc_earth_z'
    ]
    present_cols = [c for c in expected_cols if c in first_row]
    if not present_cols:
        labels = [False] * total_samples
        if return_scores:
            return labels, {}
        return labels

    raw_dict = {col: np.array([float(r[col]) for r in rows], dtype=np.float64) for col in present_cols}

    # 2. Kanäle zusammenstellen (Einzelachsen + synthetische Beträge/Magnituden)
    channels = []

    # Beschleunigung Körperkoordinaten (Einzelachsen + Betrag)
    if all(k in raw_dict for k in ['acc_x', 'acc_y', 'acc_z']):
        channels.extend([raw_dict['acc_x'], raw_dict['acc_y'], raw_dict['acc_z']])
        acc_mag = np.sqrt(raw_dict['acc_x']**2 + raw_dict['acc_y']**2 + raw_dict['acc_z']**2)
        channels.append(acc_mag)

    # Drehraten (Einzelachsen + Betrag)
    if all(k in raw_dict for k in ['gyro_x', 'gyro_y', 'gyro_z']):
        channels.extend([raw_dict['gyro_x'], raw_dict['gyro_y'], raw_dict['gyro_z']])
        gyro_mag = np.sqrt(raw_dict['gyro_x']**2 + raw_dict['gyro_y']**2 + raw_dict['gyro_z']**2)
        channels.append(gyro_mag)

    # Beschleunigung Erdkoordinaten (Einzelachsen + Betrag, falls vorhanden)
    if all(k in raw_dict for k in ['acc_earth_x', 'acc_earth_y', 'acc_earth_z']):
        channels.extend([raw_dict['acc_earth_x'], raw_dict['acc_earth_y'], raw_dict['acc_earth_z']])
        acc_earth_mag = np.sqrt(raw_dict['acc_earth_x']**2 + raw_dict['acc_earth_y']**2 + raw_dict['acc_earth_z']**2)
        channels.append(acc_earth_mag)

    if not channels:
        labels = [False] * total_samples
        if return_scores:
            return labels, {}
        return labels

    data = np.column_stack(channels)

    if 'timestamp_s' in first_row:
        timestamps = np.array([float(r['timestamp_s']) for r in rows], dtype=np.float64)
    else:
        timestamps = np.arange(total_samples) / SAMPLING_RATE_HZ

    win_len = int(SLIDE_WINDOW_SEC * SAMPLING_RATE_HZ)
    step = int(SLIDE_STEP_SEC * SAMPLING_RATE_HZ)
    min_lag = int(ACF_MIN_LAG_SEC * SAMPLING_RATE_HZ)
    max_lag = int(ACF_MAX_LAG_SEC * SAMPLING_RATE_HZ)

    if total_samples < win_len:
        labels = [False] * total_samples
        if return_scores:
            return labels, {}
        return labels

    n_windows = (total_samples - win_len) // step + 1
    hanning_win = np.hanning(win_len)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / SAMPLING_RATE_HZ)
    rep_mask = (freqs >= FFT_MIN_FREQ_HZ) & (freqs <= FFT_MAX_FREQ_HZ)

    # 3. Sliding Window Hybrid Scan
    hybrid_scores = np.zeros(n_windows)
    acf_scores = np.zeros(n_windows)
    fft_scores = np.zeros(n_windows)
    fft_norm_scores = np.zeros(n_windows)
    window_timestamps = np.zeros(n_windows)

    for i in range(n_windows):
        idx = i * step
        win_data = data[idx:idx + win_len]
        h_score, a_score, f_score = compute_max_hybrid_multichannel(
            win_data, hanning_win, freqs, rep_mask, min_lag, max_lag
        )
        hybrid_scores[i] = h_score
        acf_scores[i] = a_score
        fft_scores[i] = f_score
        fft_norm_scores[i] = min(1.0, f_score / 0.35)
        # Fenstermitte als Zeitstempel für den Plot
        mid_idx = min(total_samples - 1, idx + win_len // 2)
        window_timestamps[i] = timestamps[mid_idx]

    # 4. Glättung & Segmentierung
    is_rhythmic = hybrid_scores >= HYBRID_THRESHOLD

    closing_steps = max(1, int(MAX_PAUSE_GAP_SEC / SLIDE_STEP_SEC))
    opening_steps = max(1, int(MIN_CONSECUTIVE_SEC / SLIDE_STEP_SEC))

    clean_mask = binary_closing(is_rhythmic, structure=np.ones(closing_steps))
    clean_mask = binary_opening(clean_mask, structure=np.ones(opening_steps))

    changes = np.diff(np.concatenate(([0], clean_mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    # 5. Boolean Label-Array erzeugen (True = exercising, False = pause)
    labels = [False] * total_samples
    exercise_count = 0
    for s, e in zip(starts, ends):
        s_sample = s * step
        e_sample = min(total_samples - 1, (e - 1) * step + win_len)
        t_s = timestamps[s_sample]
        t_e = timestamps[e_sample]
        if (t_e - t_s) >= MIN_EXERCISE_DURATION_SEC:
            exercise_count += 1
            for j in range(s_sample, e_sample + 1):
                labels[j] = True

    print(f"   -> {exercise_count} Übungssätze erkannt ({sum(labels)} von {total_samples} Zeilen als 'exercising').")

    if return_scores:
        score_dict = {
            'timestamps': window_timestamps,
            'hybrid': hybrid_scores,
            'acf': acf_scores,
            'fft': fft_scores,
            'fft_norm': fft_norm_scores,
        }
        return labels, score_dict

    return labels


def save_hybrid_plot(rows, labels, output_plot_path, scores=None):
    """
    Erstellt einen visualisierenden Plot für die segmentierten Daten mit zwei Subplots:
    1. Oben: Bewegungsaktivität (Beschleunigungsvarianz) + erkannte Übungsblöcke.
    2. Unten: FFT-Score (normiert 0-1), AC-Score (Autokorrelation) und Hybrid-Score
             über die Zeit inklusive Schwellenwertlinie.
    """
    if not HAS_MATPLOTLIB:
        return
    try:
        OUTPUT_PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
        total_samples = len(rows)
        first_row = rows[0]
        
        if 'timestamp_s' in first_row:
            timestamps = np.array([float(r['timestamp_s']) for r in rows], dtype=np.float64)
        else:
            timestamps = np.arange(total_samples) / SAMPLING_RATE_HZ
            
        if all(k in first_row for k in ['acc_x', 'acc_y', 'acc_z']):
            acc_x = np.array([float(r['acc_x']) for r in rows], dtype=np.float64)
            acc_y = np.array([float(r['acc_y']) for r in rows], dtype=np.float64)
            acc_z = np.array([float(r['acc_z']) for r in rows], dtype=np.float64)
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            activity = pd.Series(acc_mag).rolling(window=int(2.0 * SAMPLING_RATE_HZ), center=True).std().fillna(0).values
        else:
            activity = np.zeros(total_samples)

        # Übungssegmente für Schattierung ermitteln
        changes = np.diff(np.concatenate(([0], np.array(labels, dtype=int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        has_scores = (
            scores is not None
            and "timestamps" in scores
            and len(scores["timestamps"]) > 0
        )

        if has_scores:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]}
            )
        else:
            fig, ax1 = plt.subplots(figsize=(15, 5))
            ax2 = None

        # --- Subplot 1: Sensor-Aktivität & Segmentierung ---
        ax1.plot(timestamps, activity, label='Bewegungsaktivität (Varianz)', color='#7f7f7f', alpha=0.6, lw=1.0)
        for idx, (s, e) in enumerate(zip(starts, ends)):
            e_idx = min(total_samples - 1, e)
            ax1.axvspan(timestamps[s], timestamps[e_idx], color='#2ca02c', alpha=0.35, label='Erkannte Übung' if idx == 0 else "")
            
        ax1.set_title(f'HYBRID-Erkennung (ACF + FFT): {output_plot_path.stem}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Aktivität (Varianz)')
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.legend(loc='upper right')

        # --- Subplot 2: FFT, AC und Hybrid Scores ---
        if ax2 is not None:
            score_ts = scores['timestamps']
            fft_curve = scores.get('fft_norm', scores.get('fft', []))
            acf_curve = scores.get('acf', [])
            hybrid_curve = scores.get('hybrid', [])

            ax2.plot(score_ts, fft_curve, label='FFT-Score (skaliert)', color='#ff7f0e', alpha=0.8, lw=1.5)
            ax2.plot(score_ts, acf_curve, label='AC-Score (Autokorrelation)', color='#1f77b4', alpha=0.8, lw=1.5)
            ax2.plot(score_ts, hybrid_curve, label='Hybrid-Score (AC × FFT)', color='#d62728', lw=2.0)

            # Schwellenwert-Linie
            ax2.axhline(HYBRID_THRESHOLD, color='black', linestyle='--', alpha=0.7, lw=1.2, label=f'Schwelle ({HYBRID_THRESHOLD})')

            # Zarte Schattierung der erkannten Übungsblöcke auch im Score-Plot
            for s, e in zip(starts, ends):
                e_idx = min(total_samples - 1, e)
                ax2.axvspan(timestamps[s], timestamps[e_idx], color='#2ca02c', alpha=0.15)

            ax2.set_ylabel('Score (0.0 - 1.0)')
            ax2.set_xlabel('Zeit (s)')
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, linestyle=':', alpha=0.5)
            ax2.legend(loc='upper right')
        else:
            ax1.set_xlabel('Zeit (s)')

        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"   Hinweis: Plot konnte nicht gespeichert werden ({e})")


def fft_ac_hybrid():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    input_csvs = sorted(INPUT_FOLDER.glob("*.csv"))
    if not input_csvs:
        raise RuntimeError(f"Keine CSV-Dateien in {INPUT_FOLDER} gefunden!")

    print(f"Starte Hybrid-Segmentierung für {len(input_csvs)} Dateien in '{INPUT_FOLDER.name}'...\n")

    for input_path in input_csvs:
        t0 = time.time()
        print(f"Verarbeite: {input_path.name}")
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(65536)
            file.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            reader = csv.DictReader(file, dialect=dialect)
            rows = list(reader)
            source_fields = list(reader.fieldnames or [])

        # 1. Vorhersage über die predict()-Methode (inkl. Score-Daten für Plot)
        labels, score_info = predict(rows, return_scores=True)
        if len(labels) != len(rows):
            raise RuntimeError(
                f"{input_path.name}: predict() returned {len(labels)} labels "
                f"for {len(rows)} rows"
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

        # 3. Plot zur visuellen Überprüfung speichern (mit Aktivität, FFT, AC und Hybrid Score)
        plot_path = OUTPUT_PLOT_FOLDER / f"{input_path.stem}_hybrid_plot.png"
        save_hybrid_plot(rows, labels, plot_path, scores=score_info)
        
        t1 = time.time()
        print(f"   -> Gespeichert in: {output_path.name} (Dauer: {t1 - t0:.2f}s)\n")

    print("Alle Dateien wurden erfolgreich verarbeitet!")


if __name__ == '__main__':
    fft_ac_hybrid()
