"""
RUN UPLOAD - Visualisierung & Inspektion in Label Studio
========================================================
Dieses Skript startet einen lokalen HTTP-Dateiserver fuer die CSVs,
startet eine lokale Label-Studio-Instanz auf Port 8080, erstellt ein
neues Projekt und importiert die Vorhersage-Segmente als sichtbare
Regionen zur visuellen Pruefung.

Ablauf:
1. Lokalen CSV-Server auf Port 8081 starten.
2. Label Studio auf Port 8080 starten (falls nicht bereits aktiv).
3. Vorheriges Projekt mit demselben Namen loeschen und frisch anlegen.
4. Aufgaben (Tasks) mit Zeitreihen-URL und Vorhersage-Intervallen importieren.
5. Projekt direkt im Browser oeffnen.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

# ==============================================================================
# 1. KONFIGURATION & LABEL-STUDIO-PARAMETER
# ==============================================================================

# Name des Zielprojekts in Label Studio
PROJECT_NAME = "Workout labeling"

# Name des Ausgabe-Ordners, dessen CSVs visualisiert werden sollen
OUTPUT_FOLDER_NAME = "voting_output"

# Server-Adressen und Port-Einstellungen
LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_PORT = 8080
LABEL_STUDIO_START_TIMEOUT_SECONDS = 120

# Lokaler CSV-Fileserver fuer Label Studio
CSV_SERVER_HOST = "127.0.0.1"
CSV_SERVER_PORT = 8081

HERE = Path(__file__).resolve().parent
OUTPUT_FOLDER = HERE / OUTPUT_FOLDER_NAME
KEY_FILE = HERE / "KEY.txt"

# ==============================================================================
# 2. LABEL-STUDIO-INTERFACE-KONFIGURATION (XML)
# ==============================================================================

# Definiert die Ansicht der Sensordaten in Label Studio (Kanaele, Farben, Hoehe)
LABEL_CONFIG = """<View>
  <Header value="Wristband recording"/>
  <TimeSeriesLabels name="activity" toName="wristband" choice="single">
    <Label value="Exercising" background="#F59E0B"/>
    <Label value="Pause" background="#16A34A"/>
  </TimeSeriesLabels>
  <TimeSeries
      name="wristband"
      value="$csv_url"
      valueType="url"
      sep=","
      timeColumn="timestamp_s"
      timeDisplayFormat=".2f"
      durationDisplayFormat=".2f">
    <MultiChannel height="220">
      <Channel column="acc_x" legend="Acceleration X" strokeColor="#DC2626"/>
      <Channel column="acc_y" legend="Acceleration Y" strokeColor="#16A34A"/>
      <Channel column="acc_z" legend="Acceleration Z" strokeColor="#2563EB"/>
    </MultiChannel>
    <MultiChannel height="220">
      <Channel column="gyro_x" legend="Gyroscope X" strokeColor="#DC2626"/>
      <Channel column="gyro_y" legend="Gyroscope Y" strokeColor="#16A34A"/>
      <Channel column="gyro_z" legend="Gyroscope Z" strokeColor="#2563EB"/>
    </MultiChannel>
    <MultiChannel height="220">
      <Channel column="acc_earth_x" legend="Earth acceleration X" strokeColor="#DC2626"/>
      <Channel column="acc_earth_y" legend="Earth acceleration Y" strokeColor="#16A34A"/>
      <Channel column="acc_earth_z" legend="Earth acceleration Z" strokeColor="#2563EB"/>
    </MultiChannel>
    <Channel column="altitude_m" legend="Altitude" strokeColor="#7C3AED" height="180"/>
  </TimeSeries>
</View>"""


# ==============================================================================
# 3. LOKALER CSV-DATEISERVER
# ==============================================================================

class CSVHandler(SimpleHTTPRequestHandler):
    """HTTP-Handler zur Bereitstellung der lokalen CSV-Dateien fuer Label Studio mit CORS-Support."""
    
    def end_headers(self):
        """Ergaenzt CORS- und Cache-Control-Header, damit Label Studio die Daten laden kann."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format, *args):
        """Unterdrueckt Konsolenausgaben des HTTP-Servers zur Wahrung der Lesbarkeit."""
        return


def start_csv_server():
    """Startet den lokalen HTTP-Fileserver im Hintergrund-Thread auf Port 8081."""
    handler = partial(CSVHandler, directory=str(OUTPUT_FOLDER))
    try:
        server = ThreadingHTTPServer(
            (CSV_SERVER_HOST, CSV_SERVER_PORT), handler
        )
    except OSError as error:
        raise RuntimeError(
            f"Could not start the CSV server on port {CSV_SERVER_PORT}: {error}"
        ) from error
    Thread(target=server.serve_forever, daemon=True).start()
    return server


# ==============================================================================
# 4. DATEI- & SEGMENT-EXTRAKTION
# ==============================================================================

def read_api_key() -> str:
    """Liest den API-Token aus der Datei KEY.txt ein und validiert diesen."""
    if not KEY_FILE.is_file():
        raise RuntimeError(f"Missing {KEY_FILE.name} beside this script.")
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key or key == "PASTE_API_KEY_HERE":
        raise RuntimeError("Paste your Label Studio API key into KEY.txt first.")
    return key


def find_output_csvs() -> list[Path]:
    """Findet alle CSV-Dateien im konfigurierten Ausgabeordner."""
    if not OUTPUT_FOLDER.is_dir():
        raise RuntimeError(
            f"Output folder does not exist: {OUTPUT_FOLDER}. Run Main.py first."
        )
    csvs = sorted(OUTPUT_FOLDER.glob("*.csv"), key=lambda path: path.name.casefold())
    if not csvs:
        raise RuntimeError(f"No CSV files found in {OUTPUT_FOLDER}.")
    return csvs


def normalize_label(value: object, path: Path, row_number: int) -> str:
    """Konvertiert beliebige Label-Bezeichnungen in die Label-Studio-Namen 'Exercising' oder 'Pause'."""
    normalized = str(value).strip().casefold()
    if normalized in {"exercise", "exercising", "uebung", "übung"}:
        return "Exercising"
    if normalized in {"pause", "rest"}:
        return "Pause"
    raise RuntimeError(
        f"Unknown label {value!r} in {path.name}, row {row_number}."
    )


def read_segments(path: Path) -> list[dict]:
    """
    Liest eine CSV-Datei ein und konvertiert zeilenweise Labels in zusammenhaengende Zeitintervalle.
    
    Returns:
        Liste von Dictionaries mit 'start' (Sekunden), 'end' (Sekunden) und 'label'.
    """
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(65536)
            file.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            if dialect.delimiter != ",":
                raise RuntimeError(
                    f"{path.name} uses {dialect.delimiter!r} as separator. "
                    "This Label Studio configuration expects comma-separated CSVs."
                )
            reader = csv.DictReader(file, dialect=dialect)
            fields = set(reader.fieldnames or [])
            missing = {"timestamp_s", "label"} - fields
            if missing:
                raise RuntimeError(
                    f"{path.name} is missing: {', '.join(sorted(missing))}"
                )

            points: list[tuple[float, str]] = []
            for row_number, row in enumerate(reader, start=2):
                points.append(
                    (
                        float(row["timestamp_s"]),
                        normalize_label(row["label"], path, row_number),
                    )
                )
    except RuntimeError:
        raise
    except (OSError, TypeError, ValueError, csv.Error) as error:
        raise RuntimeError(f"Could not read {path.name}: {error}") from error

    if not points:
        raise RuntimeError(f"{path.name} contains no data rows.")
    if any(points[index][0] <= points[index - 1][0] for index in range(1, len(points))):
        raise RuntimeError(f"timestamp_s must strictly increase in {path.name}.")

    # Zusammenhaengende Laeufe gleicher Labels ermitteln
    runs: list[dict] = []
    run_start = 0
    for index in range(1, len(points) + 1):
        if index == len(points) or points[index][1] != points[run_start][1]:
            runs.append(
                {
                    "first_index": run_start,
                    "last_index": index - 1,
                    "label": points[run_start][1],
                }
            )
            run_start = index

    # Zeitgrenzen interpolieren (Halbdistanz zwischen aufeinanderfolgenden Abtastpunkten)
    segments: list[dict] = []
    for run in runs:
        first = run["first_index"]
        last = run["last_index"]
        start = (
            points[first][0]
            if first == 0
            else (points[first - 1][0] + points[first][0]) / 2.0
        )
        end = (
            points[last][0]
            if last + 1 == len(points)
            else (points[last][0] + points[last + 1][0]) / 2.0
        )
        segments.append({"start": start, "end": end, "label": run["label"]})
    return segments


def prediction_results(segments: list[dict]) -> list[dict]:
    """Wandelt Intervallsegmente in das JSON-Format fuer Label-Studio-Vorhersageregionen um."""
    return [
        {
            "id": f"segment-{index}",
            "from_name": "activity",
            "to_name": "wristband",
            "type": "timeserieslabels",
            "value": {
                "start": segment["start"],
                "end": segment["end"],
                "instant": False,
                "timeserieslabels": [segment["label"]],
            },
        }
        for index, segment in enumerate(segments, start=1)
    ]


def build_tasks(csvs: list[Path]) -> tuple[list[dict], int]:
    """
    Erstellt fuer jede CSV-Datei eine Label-Studio-Aufgabe (Task)
    inklusive der URL zur CSV und den visualisierbaren Segmenten.
    """
    tasks = []
    total_segments = 0
    for path in csvs:
        segments = read_segments(path)
        total_segments += len(segments)
        tasks.append(
            {
                "data": {
                    "csv_url": (
                        f"http://{CSV_SERVER_HOST}:{CSV_SERVER_PORT}/"
                        f"{quote(path.name)}"
                    ),
                    "filename": path.name,
                },
                "predictions": [
                    {
                        "model_version": "csv-label-column",
                        "result": prediction_results(segments),
                    }
                ],
            }
        )
    return tasks, total_segments


# ==============================================================================
# 5. LABEL-STUDIO-STEUERUNG (PROZESS & API)
# ==============================================================================

def label_studio_is_responding() -> bool:
    """Prueft per HTTP-Request, ob Label Studio auf Port 8080 laeuft und antwortet."""
    try:
        with urlopen(LABEL_STUDIO_URL, timeout=1):
            return True
    except HTTPError:
        # Auch ein HTTP-Fehlercode beweist, dass der Webserver aktiv ist
        return True
    except (OSError, URLError):
        return False


def find_label_studio_command() -> str:
    """Findet den ausfuehrbaren Befehl fuer Label Studio auch ausserhalb des Standard-PATH."""
    from_path = shutil.which("label-studio")
    if from_path:
        return from_path

    python_folder = Path(sys.executable).resolve().parent
    executable_name = "label-studio.exe" if sys.platform == "win32" else "label-studio"
    candidates = [
        python_folder / executable_name,
        python_folder / "Scripts" / executable_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise RuntimeError(
        "Label Studio is not installed for this Python installation. Run: "
        "py -m pip install -r requirements.txt"
    )


def start_label_studio():
    """Startet den Label-Studio-Hintergrundprozess und wartet, bis Port 8080 bereit ist."""
    if label_studio_is_responding():
        print(f"Label Studio is already running at {LABEL_STUDIO_URL}; using it.")
        return None

    command = find_label_studio_command()
    print(f"Starting Label Studio at {LABEL_STUDIO_URL}...")
    try:
        process = subprocess.Popen(
            [
                command,
                "start",
                "--port",
                str(LABEL_STUDIO_PORT),
                "--no-browser",
            ]
        )
    except OSError as error:
        raise RuntimeError(f"Could not start Label Studio: {error}") from error

    deadline = time.monotonic() + LABEL_STUDIO_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if label_studio_is_responding():
            print("Label Studio is ready.")
            return process
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Label Studio stopped during startup with exit code {return_code}."
            )
        time.sleep(0.5)

    process.terminate()
    raise RuntimeError(
        f"Label Studio did not become ready within "
        f"{LABEL_STUDIO_START_TIMEOUT_SECONDS} seconds."
    )


def stop_label_studio(process) -> None:
    """Beendet den Label-Studio-Hintergrundprozess sauber."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def create_client(api_key: str):
    """Initialisiert den SDK-Client fuer die Kommunikation mit der Label-Studio-API."""
    try:
        import httpx
        from label_studio_sdk import LabelStudio
    except ImportError as error:
        raise RuntimeError(
            "Missing Label Studio dependency. Run: "
            "py -m pip install -r requirements.txt"
        ) from error
    connection = httpx.Client(
        timeout=60,
        follow_redirects=True,
        trust_env=False,
    )
    return LabelStudio(
        base_url=LABEL_STUDIO_URL,
        api_key=api_key,
        httpx_client=connection,
    )


def recreate_project(client):
    """Loescht alle alten Projekte mit identischem Titel und erstellt ein frisches Projekt."""
    deleted = 0
    for project in client.projects.list(page_size=100):
        if project.title == PROJECT_NAME:
            client.projects.delete(id=project.id)
            deleted += 1
    project = client.projects.create(
        title=PROJECT_NAME,
        label_config=LABEL_CONFIG,
    )
    return project, deleted


# ==============================================================================
# 6. HAUPTABLAUF
# ==============================================================================

def main() -> None:
    """
    Gesamtablauf:
    1. CSVs und API-Token einlesen
    2. Server starten
    3. Label-Studio-Projekt erzeugen und Tasks importieren
    4. Webbrowser oeffnen und Server bis zum Beenden (Ctrl+C) aktiv halten
    """
    csvs = find_output_csvs()
    tasks, total_segments = build_tasks(csvs)
    api_key = read_api_key()
    label_studio_process = None
    server = None
    try:
        label_studio_process = start_label_studio()
        server = start_csv_server()
        client = create_client(api_key)
        project, deleted = recreate_project(client)
        client.projects.import_tasks(
            id=project.id,
            request=tasks,
            return_task_ids=True,
        )

        project_url = f"{LABEL_STUDIO_URL}/projects/{project.id}/data/"
        print(f"Deleted {deleted} old project(s) named {PROJECT_NAME!r}.")
        print(f"Created project {project.id}: {PROJECT_NAME}")
        print(f"Imported {len(tasks)} CSV task(s) from {OUTPUT_FOLDER_NAME}.")
        print(f"Added {total_segments} visible Exercising/Pause prediction regions.")
        print(f"Open: {project_url}")
        print("Keep this script running while you use Label Studio.")
        print("Press Ctrl+C here when finished.")
        webbrowser.open(project_url)
        Event().wait()
    except KeyboardInterrupt:
        print("\nStopping servers.")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        stop_label_studio(label_studio_process)


if __name__ == "__main__":
    main()
