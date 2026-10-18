"""
RUN BENCHMARK - Zentrale Benchmark-Auswertung & Scoring aller Methoden
=====================================================================
Dieses Skript scannt alle vorhandenen Ausgabeordner (*_output), gleicht
die Vorhersagen zeilenweise mit der manuellen Ground-Truth aus manualCSV/ ab,
berechnet alle Guetemasse und schreibt das Gesamtranking nach results.txt.

Unterstuetzte Metriken:
- F2-Score (primaere Ranking-Metrik: Recall wird doppelt so stark wie Precision gewichtet)
- Recall, Precision, F1-Score, F0.5-Score, Accuracy
- Zeilenweises Pooling: TP, FP, TN, FN ueber alle Datensaetze summiert
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

# ==============================================================================
# 1. PFADE & GLOBALE KONSTANTEN
# ==============================================================================

HERE = Path(__file__).resolve().parent
INPUT_FOLDER = HERE / "InputCSV"
MANUAL_FOLDER = HERE / "manualCSV"
RESULTS_FILE = HERE / "results.txt"
OUTPUT_FOLDER_SUFFIX = "_output"

# Zu benchmarkende Ordner:
#   - None:                    Alle vorhandenen *_output-Ordner benchmarken (Standard)
#   - "ordner_name" / Path:    Nur einen bestimmten Ordner benchmarken (z. B. "fft_ac_hybrid_output" oder "ggs")
#   - ["ordner1", "ordner2"]:  Liste bestimmter Ordner
TARGET_FOLDER: Optional[Union[str, List[str], Path]] = ["fft_ac_hybrid_output", "ggs_output", "voting_output", "fluss_output"] 

# Gueltige Bezeichnungen fuer die beiden Zustandsklassen in den CSV-Dateien
EXERCISING_LABELS = {"exercise", "exercising", "uebung", "übung"}
PAUSE_LABELS = {"pause", "rest"}


class BenchmarkError(RuntimeError):
    """Wird bei Fehlern waehrend des Benchmarks geworfen (z. B. Mismatch in Zeilenzahl)."""
    pass


# ==============================================================================
# 2. METRIK-DATENSTRUKTUREN & BERECHNUNG
# ==============================================================================

@dataclass(frozen=True)
class Metrics:
    """
    Kapselt die Zaehlwerte der Konfusionsmatrix (TP, FP, TN, FN)
    und berechnet daraus alle relevanten Guetemasse.
    """
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @staticmethod
    def divide(numerator: float, denominator: float) -> float:
        """Sichere Division mit Rueckgabe von 0.0 bei Division durch Null."""
        return numerator / denominator if denominator else 0.0

    @property
    def rows(self) -> int:
        """Gesamtzahl aller ausgewerteten Zeilen."""
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        return self.divide(
            self.true_positive,
            self.true_positive + self.false_positive,
        )

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        return self.divide(
            self.true_positive,
            self.true_positive + self.false_negative,
        )

    @property
    def f1(self) -> float:
        """Harmonisches Mittel aus Precision und Recall (Beta = 1.0)."""
        return self.divide(
            2 * self.true_positive,
            2 * self.true_positive + self.false_positive + self.false_negative,
        )

    @property
    def f2(self) -> float:
        """F2-Score: Gewichtet Recall doppelt so stark wie Precision (Beta = 2.0)."""
        return self.divide(
            5.0 * self.true_positive,
            5.0 * self.true_positive
            + self.false_positive
            + 4.0 * self.false_negative,
        )

    @property
    def f0_5(self) -> float:
        """F0.5-Score: Gewichtet Precision doppelt so stark wie Recall (Beta = 0.5)."""
        return self.divide(
            1.25 * self.true_positive,
            1.25 * self.true_positive
            + self.false_positive
            + 0.25 * self.false_negative,
        )

    @property
    def accuracy(self) -> float:
        """Klassifikationsgenauigkeit = (TP + TN) / Gesamtzeilen"""
        return self.divide(
            self.true_positive + self.true_negative,
            self.rows,
        )

    def __add__(self, other: "Metrics") -> "Metrics":
        """Summiert die Konfusionsmatrizen zweier Datensaetze zeilenweise auf."""
        return Metrics(
            self.true_positive + other.true_positive,
            self.false_positive + other.false_positive,
            self.true_negative + other.true_negative,
            self.false_negative + other.false_negative,
        )


@dataclass(frozen=True)
class Evaluation:
    """Ergebnis der Gesamtevaluierung eines Methoden-Ausgabeordners."""
    output_folder: Path
    combined: Metrics
    per_file: tuple[tuple[str, Metrics], ...]
    ignored_files: tuple[str, ...]


# ==============================================================================
# 3. DATEI- & FORMAT-HILFSFUNKTIONEN
# ==============================================================================

def csv_files(folder: Path) -> list[Path]:
    """Liefert alle CSV-Dateien eines Ordners alphabetisch sortiert zurueck."""
    if not folder.is_dir():
        return []
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.casefold() == ".csv"
        ),
        key=lambda path: path.name.casefold(),
    )


def unique_filename_map(paths: list[Path], folder: Path) -> dict[str, Path]:
    """Erstellt ein Mapping von kleingeschriebenen Dateinamen zu Dateipfaden (Duplikatspruefung)."""
    result: dict[str, Path] = {}
    for path in paths:
        key = path.name.casefold()
        if key in result:
            raise BenchmarkError(f"Duplicate CSV filename in {folder}: {path.name}")
        result[key] = path
    return result


def detect_dialect(path: Path) -> csv.Dialect:
    """Erkennt automatisch Trennzeichen und Dialekt der CSV-Datei."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(65536)
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except (OSError, csv.Error) as error:
        raise BenchmarkError(f"Could not determine CSV separator for {path.name}.") from error


def normalize_label(value: object, path: Path, row_number: int) -> bool:
    """
    Normalisiert beliebige Label-Schreibweisen auf einen einheitlichen Boolean:
    True = exercising, False = pause.
    """
    normalized = str(value).strip().casefold()
    if normalized in EXERCISING_LABELS:
        return True
    if normalized in PAUSE_LABELS:
        return False
    raise BenchmarkError(
        f"Unknown label {value!r} in {path.name}, row {row_number}."
    )


def read_time_and_labels(path: Path) -> tuple[list[float], list[bool]]:
    """Liest die Spalten 'timestamp_s' und 'label' einer Datei vollstaendig ein."""
    dialect = detect_dialect(path)
    timestamps: list[float] = []
    labels: list[bool] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, dialect=dialect)
            fields = set(reader.fieldnames or [])
            missing = {"timestamp_s", "label"} - fields
            if missing:
                raise BenchmarkError(
                    f"{path.name} is missing: {', '.join(sorted(missing))}"
                )
            for row_number, row in enumerate(reader, start=2):
                timestamps.append(float(row["timestamp_s"]))
                labels.append(normalize_label(row["label"], path, row_number))
    except BenchmarkError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise BenchmarkError(f"Could not read {path}: {error}") from error
    if not timestamps:
        raise BenchmarkError(f"{path.name} has no data rows.")
    return timestamps, labels


# ==============================================================================
# 4. AUSWERTUNG & DATENABGLEICH
# ==============================================================================

def compare_csvs(manual_path: Path, prediction_path: Path) -> Metrics:
    """
    Vergleicht eine Vorhersage-Datei mit ihrer manuellen Ground-Truth zeilenweise.
    Prueft zusaetzlich auf uebereinstimmende Zeilenanzahl und identische Timestamps.
    """
    manual_time, manual = read_time_and_labels(manual_path)
    predicted_time, predicted = read_time_and_labels(prediction_path)
    if len(manual) != len(predicted):
        raise BenchmarkError(
            f"Row-count mismatch for {manual_path.name}: manual={len(manual)}, "
            f"predicted={len(predicted)}"
        )

    # Pruefen, ob die Zeitstempel exakt identisch sind
    for index, (expected, actual) in enumerate(zip(manual_time, predicted_time), start=2):
        if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-9):
            raise BenchmarkError(
                f"Timestamp mismatch in {manual_path.name}, row {index}: "
                f"manual={expected}, predicted={actual}"
            )

    # Konfusionsmatrix berechnen
    tp = fp = tn = fn = 0
    for truth, prediction in zip(manual, predicted):
        if prediction and truth:
            tp += 1
        elif prediction:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    return Metrics(tp, fp, tn, fn)


def evaluate_output_folder(
    output_folder: Path,
    manual_map: dict[str, Path],
) -> Evaluation:
    """
    Vergleicht alle Vorhersagen eines Methoden-Ordners mit den Ground-Truth-Dateien.
    Summiert die Metriken zeilenweise (Pooling) und speichert Einzelwerte pro Datei.
    """
    outputs = unique_filename_map(csv_files(output_folder), output_folder)
    if not outputs:
        raise BenchmarkError("No CSV files found.")

    # Pruefen, ob Dateien aus manualCSV fehlen
    missing = [manual_map[key].name for key in manual_map.keys() - outputs.keys()]
    if missing:
        raise BenchmarkError(
            "Missing prediction CSV(s): " + ", ".join(sorted(missing))
        )

    ignored = tuple(
        sorted(outputs[key].name for key in outputs.keys() - manual_map.keys())
    )
    per_file: list[tuple[str, Metrics]] = []
    combined = Metrics()
    for key in sorted(manual_map):
        metrics = compare_csvs(manual_map[key], outputs[key])
        per_file.append((manual_map[key].name, metrics))
        combined = combined + metrics
    return Evaluation(output_folder, combined, tuple(per_file), ignored)


def format_metric_line(metrics: Metrics) -> str:
    """Formatiert alle berechneten Kennzahlen einheitlich als Textzeile."""
    return (
        f"F2={metrics.f2:.4f}, recall={metrics.recall:.4f}, "
        f"precision={metrics.precision:.4f}, F1={metrics.f1:.4f}, "
        f"F0.5={metrics.f0_5:.4f}, accuracy={metrics.accuracy:.4f}"
    )


# ==============================================================================
# 5. ERGEBNISBERICHT (results.txt)
# ==============================================================================

def write_results(
    evaluations: list[Evaluation],
    errors: list[tuple[str, str]],
) -> None:
    """
    Sortiert die Methoden nach F2 und Recall, baut den Bericht auf
    und speichert diesen atomar ueber eine .partial-Datei in results.txt.
    """
    ranked = sorted(
        evaluations,
        key=lambda item: (item.combined.f2, item.combined.recall),
        reverse=True,
    )
    lines = [
        "EXERCISE DETECTION RESULTS",
        "==========================",
        "",
        "Positive class: exercising",
        "Aggregation: pooled row-level counts across all manually labeled CSVs.",
        "Longer recordings therefore have proportionally more weight.",
        "Ranking: F2 (recall is weighted twice as strongly as precision).",
        "",
        "RANKING",
        "-------",
    ]
    if ranked:
        for rank, item in enumerate(ranked, start=1):
            lines.append(
                f"{rank}. {item.output_folder.name}: "
                f"{format_metric_line(item.combined)}"
            )
    else:
        lines.append("No method produced a complete valid result.")

    # Aufgetretene Fehler dokumentieren
    if errors:
        lines.extend(["", "ERRORS", "------"])
        for name, message in sorted(errors, key=lambda item: item[0].casefold()):
            lines.append(f"{name}: {message}")

    # Detaillierte Aufschluesselung pro Methode und Datensatz
    for item in ranked:
        metrics = item.combined
        lines.extend(
            [
                "",
                item.output_folder.name,
                "-" * len(item.output_folder.name),
                format_metric_line(metrics),
                (
                    f"rows={metrics.rows}, TP={metrics.true_positive}, "
                    f"FP={metrics.false_positive}, TN={metrics.true_negative}, "
                    f"FN={metrics.false_negative}"
                ),
            ]
        )
        if item.ignored_files:
            lines.append(
                "Not scored because no matching manual CSV exists: "
                + ", ".join(item.ignored_files)
            )
        lines.append("Per dataset:")
        for filename, file_metrics in item.per_file:
            lines.append(f"  {filename}: {format_metric_line(file_metrics)}")

    # Atomares Schreiben via .partial und rename
    partial = RESULTS_FILE.with_name(RESULTS_FILE.name + ".partial")
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(partial, RESULTS_FILE)


# ==============================================================================
# 6. HAUPTFUNKTION (BENCHMARK-RUNNER)
# ==============================================================================

def get_all_output_folders() -> list[Path]:
    """Sucht und liefert alle verfuegbaren *_output-Ordner im Projektverzeichnis."""
    return sorted(
        (
            path
            for path in HERE.iterdir()
            if path.is_dir() and path.name.endswith(OUTPUT_FOLDER_SUFFIX)
        ),
        key=lambda path: path.name.casefold(),
    )


def resolve_output_folders(
    target: Optional[Union[str, List[str], Path]] = None,
) -> list[Path]:
    """
    Ermittelt die zu benchmarkenden Ausgabeordner basierend auf dem Zielparameter
    oder findet alle verfuegbaren *_output Ordner.
    
    Unterstuetzt:
    - None oder "all": Alle verfuegbaren *_output Ordner.
    - String oder Path: Einzelner Ordnername (mit oder ohne '_output') oder Pfad.
    - Liste: Mehrere spezifische Ordner.
    """
    all_folders = get_all_output_folders()
    if not all_folders:
        raise BenchmarkError(f"No *{OUTPUT_FOLDER_SUFFIX} folders found to benchmark.")

    # None oder "all" -> Alle Ordner
    if target is None or (isinstance(target, str) and target.strip().casefold() == "all"):
        return all_folders

    targets = [target] if isinstance(target, (str, Path)) else list(target)
    resolved: list[Path] = []
    available_names = [f.name for f in all_folders]

    for item in targets:
        if isinstance(item, Path):
            p = item if item.is_absolute() else HERE / item
        else:
            raw = str(item).strip()
            # Wenn "all" in einer Liste steht
            if raw.casefold() == "all":
                return all_folders
            p = Path(raw) if os.path.isabs(raw) else HERE / raw

        # Falls der Pfad so nicht existiert, Versuch mit angehaengtem '_output'
        if not p.exists() and not p.name.endswith(OUTPUT_FOLDER_SUFFIX):
            candidate = p.with_name(p.name + OUTPUT_FOLDER_SUFFIX)
            if candidate.exists():
                p = candidate

        if not p.exists():
            raise BenchmarkError(
                f"Output-Ordner nicht gefunden: '{item}' (geprueft: {p.name}).\n"
                f"Verfuegbare Ordner: {', '.join(available_names)}"
            )
        if not p.is_dir():
            raise BenchmarkError(f"Ziel ist kein Verzeichnis: '{p}'.")

        if p not in resolved:
            resolved.append(p)

    return sorted(resolved, key=lambda path: path.name.casefold())


def run_benchmark(target: Optional[Union[str, List[str], Path]] = None) -> None:
    """
    Hauptablauf des Benchmarks:
    1. Prueft Vorhandensein von InputCSV und manualCSV
    2. Ermittelt die gewuenschten Output-Ordner (alle oder spezifische Auswahl)
    3. Fuehrt die Evaluierung durch und erzeugt results.txt
    
    Args:
        target: Optionaler Name, Pfad oder Liste von Ordnern. Falls None,
                wird die globale Variable TARGET_FOLDER verwendet.
    """
    INPUT_FOLDER.mkdir(exist_ok=True)
    MANUAL_FOLDER.mkdir(exist_ok=True)
    input_csvs = csv_files(INPUT_FOLDER)
    manual_csvs = csv_files(MANUAL_FOLDER)
    if not input_csvs:
        raise BenchmarkError(f"No input CSVs found in {INPUT_FOLDER}.")
    if not manual_csvs:
        raise BenchmarkError(f"No manual CSVs found in {MANUAL_FOLDER}.")

    # Duplikatspruefung vorab durchfuehren
    unique_filename_map(input_csvs, INPUT_FOLDER)
    manual_map = unique_filename_map(manual_csvs, MANUAL_FOLDER)

    # Zielordner ermitteln (Parameter hat Vorrang vor TARGET_FOLDER)
    effective_target = target if target is not None else TARGET_FOLDER
    output_folders = resolve_output_folders(effective_target)

    scope_text = "ALLE" if effective_target is None or (isinstance(effective_target, str) and effective_target.strip().casefold() == "all") else "spezifische"
    print(f"Benchmark-Umfang ({scope_text}, {len(output_folders)} Ordner): {', '.join(f.name for f in output_folders)}")
    print("Bewerte Methoden gegen manualCSV...\n")

    evaluations: list[Evaluation] = []
    errors: list[tuple[str, str]] = []
    for folder in output_folders:
        try:
            evaluations.append(evaluate_output_folder(folder, manual_map))
        except Exception as error:
            errors.append(
                (
                    folder.name,
                    f"incomplete or invalid result; no partial metrics used: {error}",
                )
            )

    write_results(evaluations, errors)
    print(f"Benchmark abgeschlossen! Ergebnisse in {RESULTS_FILE.name}:\n")
    print(RESULTS_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    try:
        run_benchmark()
    except BenchmarkError as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
