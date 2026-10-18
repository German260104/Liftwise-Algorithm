"""
RUN VOTING - Ensemble-Generator fuer mehrere Erkennungsmethoden
==============================================================
Dieses Skript kombiniert die Vorhersagen (exercising / pause) aus mehreren
*_output-Ordnern zeilenweise zu einer gemeinsamen Konsens-Vorhersage.

Unterstuetzt:
- Prozentuale Abstimmung (z. B. einfache Mehrheit oder Konsens)
- Gewichtetes Voting (Verteilung individueller Stimmgewichte)
- Zeitliche Mindestdauer-Glaettung zur Vermeidung von hochfrequentem Flackern

Das Ergebnis wird als vollstaendige CSVs im Ordner voting_output/ gespeichert
und kann anschliessend ueber run_benchmark.py wie eine Methode bewertet werden.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

# ==============================================================================
# 1. HYPERPARAMETER & VOTING-KONFIGURATION
# ==============================================================================

# Voting-Strategie:
#   - "percentage": Prozentuale Huerde (wie viel % der Algorithmen muessen zustimmen).
#   - "weighted":   Gewichtetes Voting (Algorithmen haben unterschiedliche Gewichte).
VOTING_MODE: Literal["percentage", "weighted"] = "percentage"

# --- Einstellungen fuer Modus "percentage" ---
# Wie viel Prozent der Algorithmen muessen mindestens 'exercising' erkennen,
# damit das Ensemble 'exercising' setzt? (Wertebereich: 0.0 bis 100.0)
# Beispiele:
#   - 50.0  -> Mehrheitsentscheid (>= 50% der Detektoren)
#   - 33.3  -> Bei z. B. 3 Detektoren genuegt bereits mindestens 1 Detektor
#   - 100.0 -> Einstimmigkeit (alle Detektoren muessen zustimmen)
MIN_ALGORITHMS_PERCENT: float = 30.0

# --- Einstellungen fuer Modus "weighted" ---
# Individuelle Gewichte pro Algorithmus (Schluessel = Name des _output Ordners).
# Nicht aufgefuehrte Methoden erhalten DEFAULT_WEIGHT.
METHOD_WEIGHTS: Dict[str, float] = {

}
DEFAULT_WEIGHT: float = 1.0

# Schwellenwert fuer gewichtetes Voting (0.0 bis 1.0):
# Liegt der Anteil (Summe der 'exercising'-Gewichte / Summe aller Gewichte)
# bei >= WEIGHTED_THRESHOLD, wird 'exercising' ausgegeben.
WEIGHTED_THRESHOLD: float = 0.50

# --- Zu beruecksichtigende Algorithmus-Ordner ---
# None: Findet automatisch alle vorhandenen *_output Ordner (ausser dem eigenen).
# Alternativ Liste von Ordnernamen, z. B. ["fft_ac_hybrid_output", "fluss_output"]
SELECTED_METHOD_FOLDERS: Optional[List[str]] = ["fft_ac_hybrid_output", "ggs_output", "fluss_output"]

# --- Zeitliche Glaettung / Mindestdauer ---
# Minimale Dauer eines Uebungs- oder Pausenblocks in Sekunden.
# Unterdrueckt hochfrequentes Flackern. 0.0 schaltet die Glaettung ab.
MIN_DURATION_SECONDS: float = 10.0

# Name des Ausgabe-Ordners fuer dieses Voting-Verfahren
OUTPUT_FOLDER_NAME: str = "voting_output"

# ==============================================================================
# 2. PFADE & GLOBALE KONSTANTEN
# ==============================================================================

HERE = Path(__file__).resolve().parent
OUTPUT_FOLDER = HERE / OUTPUT_FOLDER_NAME

# Gueltige Bezeichnungen fuer die beiden Zustandsklassen in den CSV-Dateien
EXERCISING_LABELS = {"exercise", "exercising", "uebung", "übung"}
PAUSE_LABELS = {"pause", "rest"}


class VotingError(RuntimeError):
    """Wird bei Fehlern waehrend des Voting-Prozesses geworfen."""
    pass


# ==============================================================================
# 3. HILFSFUNKTIONEN FUER DATEI- & DATENVERARBEITUNG
# ==============================================================================

def discover_input_folders() -> List[Path]:
    """
    Ermittelt alle Methoden-Ordner, die in die Abstimmung einbezogen werden sollen.
    
    Falls SELECTED_METHOD_FOLDERS definiert ist, werden nur diese Ordner genutzt.
    Andernfalls werden alle Unterordner gesucht, die auf '_output' enden
    (ausgenommen der eigene Ausgabeordner).
    """
    if SELECTED_METHOD_FOLDERS is not None:
        folders = [HERE / name for name in SELECTED_METHOD_FOLDERS]
    else:
        folders = [
            path
            for path in HERE.iterdir()
            if path.is_dir()
            and path.name.endswith("_output")
            and path.name != OUTPUT_FOLDER_NAME
        ]
    return sorted(folders, key=lambda p: p.name.casefold())


def detect_dialect(path: Path) -> csv.Dialect:
    """Erkennt automatisch das Trennzeichen einer CSV-Datei (z. B. Komma oder Semikolon)."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(65536)
    return csv.Sniffer().sniff(sample, delimiters=",;\t|")


def read_predictions(path: Path) -> Tuple[List[float], List[bool], List[dict], List[str]]:
    """
    Liest eine Vorhersage-CSV ein und extrahiert Zeitstempel und Boolean-Labels.
    
    Returns:
        timestamps:  Liste der Zeitpunkte in Sekunden (timestamp_s)
        labels:      Liste von Booleans (True = exercising, False = pause)
        rows:        Originale Zeilen als Liste von Dictionaries
        fieldnames:  Liste der urspruenglichen Spaltennamen
    """
    dialect = detect_dialect(path)
    timestamps: List[float] = []
    labels: List[bool] = []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, dialect=dialect)
        fieldnames = list(reader.fieldnames or [])
        if "timestamp_s" not in fieldnames or "label" not in fieldnames:
            raise VotingError(f"{path.name} fehlen erforderliche Spalten ('timestamp_s' oder 'label').")
        for row in reader:
            timestamps.append(float(row["timestamp_s"]))
            labels.append(row["label"].strip().casefold() in EXERCISING_LABELS)
            rows.append(row)
    return timestamps, labels, rows, fieldnames


# ==============================================================================
# 4. ENSEMBLE- & GLAETTUNGSLOGIK
# ==============================================================================

def decide_row(votes: List[bool], folder_names: List[str]) -> bool:
    """
    Entscheidet fuer eine einzelne Zeile anhand der konfigurierten Strategie ueber das Label.
    
    Args:
        votes:        Liste von Booleans aller Methoden fuer die aktuelle Zeile.
        folder_names: Namen der zugehoerigen Methoden-Ordner (fuer Gewichtung).
        
    Returns:
        True (exercising) oder False (pause).
    """
    if not votes:
        return False

    if VOTING_MODE == "percentage":
        # Prozentualer Anteil der Algorithmen, die fuer 'exercising' stimmen
        exercising_count = sum(votes)
        percent = (exercising_count / len(votes)) * 100.0
        return percent >= MIN_ALGORITHMS_PERCENT

    elif VOTING_MODE == "weighted":
        # Gewichtete Summe der Ja-Stimmen im Verhaeltnis zur Gesamtgewichtung
        weights = [METHOD_WEIGHTS.get(name, DEFAULT_WEIGHT) for name in folder_names]
        active_weight = sum(w for w, v in zip(weights, votes) if v)
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return False
        return (active_weight / total_weight) >= WEIGHTED_THRESHOLD

    else:
        raise VotingError(f"Unbekannter VOTING_MODE: {VOTING_MODE!r}. Erlaubt sind 'percentage' oder 'weighted'.")


def smooth_labels(labels: List[bool], timestamps: List[float], min_duration: float) -> List[bool]:
    """
    Filtert kurze Fragmente heraus, die eine vorgegebene Mindestdauer unterschreiten.
    
    Verhindert hochfrequentes Hin- und Herspringen zwischen Uebung und Pause.
    Zu kurze Bloecke werden an den vorhergehenden Zustand angepasst.
    """
    if not labels or min_duration <= 0.0:
        return labels

    smoothed = list(labels)
    n = len(smoothed)
    start_idx = 0

    while start_idx < n:
        current_val = smoothed[start_idx]
        end_idx = start_idx
        while end_idx < n and smoothed[end_idx] == current_val:
            end_idx += 1

        duration = timestamps[end_idx - 1] - timestamps[start_idx]
        if duration < min_duration:
            # Segment ist kuerzer als Mindestdauer -> an Vorgaenger anpassen
            if start_idx > 0:
                replacement = smoothed[start_idx - 1]
            elif end_idx < n:
                replacement = smoothed[end_idx]
            else:
                replacement = current_val

            for i in range(start_idx, end_idx):
                smoothed[i] = replacement

        start_idx = end_idx

    return smoothed


# ==============================================================================
# 5. HAUPTFUNKTION (VOTING-VERFAHREN)
# ==============================================================================

def voting_method() -> None:
    """
    Fuehrt die Ensemble-Abstimmung ueber alle gemeinsamen Dateien durch
    und speichert die fertigen CSVs im Ausgabeordner.
    """
    input_folders = discover_input_folders()
    if not input_folders:
        raise VotingError("Keine Quellordner fuer das Voting gefunden.")
    if len(input_folders) < 2:
        print(
            f"HINWEIS: Nur {len(input_folders)} Quellordner gefunden: "
            f"{[f.name for f in input_folders]}. Voting benoetigt idealerweise mindestens 2 Methoden."
        )

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    folder_names = [f.name for f in input_folders]

    # Statusausgabe der aktiven Konfiguration
    print(f"--- Starte Voting ({VOTING_MODE}) ---")
    print(f"Eingebundene Methoden ({len(folder_names)}): {', '.join(folder_names)}")
    if VOTING_MODE == "percentage":
        print(f"Bedingung: Mindestens {MIN_ALGORITHMS_PERCENT:.1f}% der Algorithmen muessen zustimmen.")
    else:
        print(f"Bedingung: Gewichtete Summe >= {WEIGHTED_THRESHOLD * 100:.1f}%.")
        print(f"Gewichte: {[f'{name}: {METHOD_WEIGHTS.get(name, DEFAULT_WEIGHT)}' for name in folder_names]}")
    if MIN_DURATION_SECONDS > 0:
        print(f"Zeitliche Mindestdauer-Glaettung: {MIN_DURATION_SECONDS} Sekunden.")

    # Gemeinsame CSV-Dateien aus dem ersten Ordner ermitteln
    first_folder = input_folders[0]
    csv_files = sorted(
        [p.name for p in first_folder.glob("*.csv") if not p.name.endswith(".partial")],
        key=lambda x: x.casefold(),
    )
    if not csv_files:
        raise VotingError(f"Keine CSV-Dateien in {first_folder.name} gefunden.")

    # Jede Datei ueber alle Methoden hinweg zeilenweise abstimmen
    for filename in csv_files:
        all_method_labels: List[List[bool]] = []
        base_rows: List[dict] = []
        base_timestamps: List[float] = []
        base_fields: List[str] = []

        # Vorhersagen aus jedem Quellordner einlesen und auf Konsistenz pruefen
        for folder in input_folders:
            csv_path = folder / filename
            if not csv_path.exists():
                raise VotingError(f"Datei {filename} fehlt im Ordner {folder.name}.")

            timestamps, labels, rows, fields = read_predictions(csv_path)

            if not base_timestamps:
                base_timestamps = timestamps
                base_rows = rows
                base_fields = fields
            else:
                # Pruefen auf gleiche Zeilenanzahl
                if len(timestamps) != len(base_timestamps):
                    raise VotingError(
                        f"Zeilenanzahl-Abweichung in {filename}: {folder.name} hat {len(timestamps)} Zeilen, "
                        f"erwartet wurden {len(base_timestamps)}."
                    )
                # Pruefen auf exakt identische Zeitstempel
                for idx, (t1, t2) in enumerate(zip(base_timestamps, timestamps), start=2):
                    if not math.isclose(t1, t2, rel_tol=0.0, abs_tol=1e-9):
                        raise VotingError(
                            f"Timestamp-Abweichung in {filename} ({folder.name}), Zeile {idx}: {t1} != {t2}"
                        )

            all_method_labels.append(labels)

        # Zeilenweise Abstimmung durchfuehren
        num_rows = len(base_timestamps)
        ensemble_labels: List[bool] = []
        for row_idx in range(num_rows):
            votes = [all_method_labels[m][row_idx] for m in range(len(input_folders))]
            ensemble_labels.append(decide_row(votes, folder_names))

        # Zeitliche Glaettung anwenden
        if MIN_DURATION_SECONDS > 0.0:
            ensemble_labels = smooth_labels(ensemble_labels, base_timestamps, MIN_DURATION_SECONDS)

        # Atomares Schreiben ueber temporaere .partial-Datei
        output_path = OUTPUT_FOLDER / filename
        partial_path = output_path.with_name(output_path.name + ".partial")
        out_fields = [f for f in base_fields if f != "label"] + ["label"]

        with partial_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()
            for row, exercising in zip(base_rows, ensemble_labels):
                row["label"] = "exercising" if exercising else "pause"
                writer.writerow(row)

        os.replace(partial_path, output_path)
        print(f"  -> {filename} erfolgreich erzeugt.")

    print(f"Voting abgeschlossen! CSVs gespeichert in {OUTPUT_FOLDER_NAME}/\n")


if __name__ == "__main__":
    try:
        voting_method()
    except VotingError as err:
        print(f"FEHLER beim Voting: {err}")
        raise SystemExit(1)
