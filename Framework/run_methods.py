"""
RUN METHODS - Zentrale Ausfuehrung der Erkennungsmethoden
=========================================================
Dieses Skript fuehrt ausschliesslich die hier registrierten Erkennungs-
algorithmen aus. Fuer jeden Algorithmus wird die entsprechende Output-CSV
im jeweiligen Ordner (<methode>_output/) generiert.

Es findet hier bewusst KEINE Benchmark-Auswertung statt (dafuer: run_benchmark.py).
"""

from __future__ import annotations

import time

# ==============================================================================
# 1. KONFIGURATION DER METHODEN (HIER EINBINDEN & AKTIVIEREN)
# ==============================================================================

# Import der gewuenschten Erkennungsmethoden aus ihren jeweiligen Modulen:
from method_fft_ac_hybrid import fft_ac_hybrid
from GreedyGaussianSegmentation import ggs_method
from fluss import fluss_method
from run_voting import voting_method

# Liste aller Methoden, die beim Start dieses Skripts sequenziell ausgefuehrt werden.
# - Methode aktivieren:   Einfach in die Liste aufnehmen.
# - Methode deaktivieren: Mit '#' auskommentieren.
# - Reihenfolge: Falls das Voting-System mitlaufen soll, muss es nach den
#   Einzeldetektoren aufgerufen werden, da es deren Outputs benoetigt.
METHODS_TO_RUN = [
    fft_ac_hybrid,
    ggs_method,
    fluss_method,
    # voting_method,
]

# ==============================================================================
# 2. HAUPTFUNKTION & AUSFUEHRUNGSLOGIK
# ==============================================================================

def main() -> None:
    """
    Iteriert ueber alle in METHODS_TO_RUN definierten Funktionen, fuehrt sie
    nacheinander aus und misst die jeweilige Laufzeit.
    """
    # Pruefen, ob mindestens eine Methode aktiv ist
    if not METHODS_TO_RUN:
        print("Keine Methoden in METHODS_TO_RUN konfiguriert. Skript beendet.")
        return

    print(f"Starte Ausfuehrung von {len(METHODS_TO_RUN)} Methode(n)...")
    total_start = time.time()

    # Sequenzielle Ausfuehrung jeder Methode mit Zeitmessung und Fehlerbehandlung
    for idx, method in enumerate(METHODS_TO_RUN, start=1):
        method_name = getattr(method, "__name__", str(method))
        print(f"\n[{idx}/{len(METHODS_TO_RUN)}] >>> Starte Methode: {method_name}")
        start_time = time.time()
        try:
            # Eigentlicher Aufruf der Erkennungsmethode
            method()
            elapsed = time.time() - start_time
            print(f">>> {method_name} erfolgreich beendet in {elapsed:.2f}s.")
        except Exception as err:
            # Fehler abfangen, damit nachfolgende Methoden trotzdem ausgefuehrt werden
            print(f">>> FEHLER in {method_name}: {err}")

    total_elapsed = time.time() - total_start
    print(f"\nAlle Methoden wurden abgearbeitet. (Gesamtdauer: {total_elapsed:.2f}s)\n")


if __name__ == "__main__":
    main()
