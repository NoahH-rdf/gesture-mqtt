#WICHTIG

NACH DEM CLONEN DURCH GITHUB FOLGENDE ANWEISUNGEN AUSFÜHREN 
## Setup
1. `uv sync` – installiert alle Abhängigkeiten
2. Modell herunterladen:
   `curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
3. `uv run main.py`