# Air Control — Touchless Computer Control

Control your computer with hand and body gestures through a webcam — touchless
mouse, air drawing, air-writing, and gesture-based game control. Air Control
tracks you in real time so you can operate your PC without touching the keyboard
or trackpad.

Built in Python with MediaPipe (hand & body tracking), OpenCV (camera), PyQt5
(GUI), PyAutoGUI (input control), and scikit-learn (handwriting recognition).

## Features

The app runs as a single sleek dark desktop window with a live camera preview.
One feature is active at a time; you switch between them with on-screen buttons.

### Touchless Mouse
Move the cursor with your index finger; pinch to click, pinch with the middle
finger curled to drag, two fingers to scroll.

### Air Canvas
A transparent full-screen overlay you draw on with your index finger, over your
desktop or any app; make a fist to clear. Press Esc to exit.

### Air Writing
Write characters in the air on the full-screen overlay; they are recognised and
typed into whatever app has focus. Open palm confirms a character, pinch adds a
space, two fingers backspace, fist clears.

### Game (two-hand control)
Control games with gestures. Right hand = pedals (palm = gas, fist = brake),
left hand = steering (two fingers = right, fist = left, palm = neutral). Each of
the four controls has an enable checkbox and a configurable key, so it works
with any simple game — e.g. Hill Climb Racing or a browser racer.

### Cricket (full-body)
Stand back and swing your arm down to bat in simple browser cricket games (the
swing sends a click to hit the ball). Uses full-body pose tracking.

Settings — smoothing, click sensitivity, scroll speed, pen thickness, swing
sensitivity, configurable game keys/controls, and remappable gestures — are
adjustable with live controls and persist between sessions.

---

## Requirements

- **Python 3.9–3.12** (MediaPipe does not support 3.13+)
- A webcam
- Windows recommended (the transparent overlay and input injection are most
  reliable there)

> **Important:** this project pins `mediapipe==0.10.21`. Newer versions (0.10.22+)
> removed the `mediapipe.solutions.hands` API this project relies on.

## Installation

```bash
# clone the repo
git clone https://github.com/<your-username>/air-control.git
cd air-control

# create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python air_control.py
```

The app window opens with a camera preview. Pick a feature; gesture hints are
shown in the window. Click **Stop All** or close the window to stop.

- **Air Canvas / Air Writing** open a transparent full-screen overlay. Press
  **Esc** or the on-screen button to exit back to the window.
- **Game / Cricket** send input to whatever window is focused, so click the game
  first, then switch the mode on.

Air Writing needs `char_model.joblib` in the project folder. It is included in
the repo; to retrain it, run `Phase8_Training_OpenML.ipynb`.

---

## How it was built

The project was developed in stages, each captured as a Jupyter notebook:

| Notebook | What it covers |
|----------|----------------|
| `Phase1_Setup.ipynb` | Environment setup and MediaPipe verification |
| `Phase2_HandTracking.ipynb` | Threaded camera capture + 21-point hand tracking |
| `Phase3_CursorControl.ipynb` | Mapping hand position to cursor movement (smoothing) |
| `Phase4_VirtualMouse.ipynb` | Click, drag, and scroll gestures |
| `Phase5_AirCanvas.ipynb` | Transparent drawing overlay |
| `Phase7_AirWriting.ipynb` | Digit recognition (MNIST + scikit-learn) |
| `Phase8_SentenceWriting.ipynb` | Letter + digit recognition, sentence building |
| `Phase8_Training_OpenML.ipynb` | Trains the character model (`char_model.joblib`) |

`air_control.py` unifies all of the above into one application.

---

## Changelog

Grouped by what each update added. (Add real dates if you'd like — these are in
the order features were built.)

### Update 5 — Cricket polish & racing fixes
- Reduced cricket-mode lag by clearing the camera buffer and draining stale
  frames (the swing now registers far closer to real time).
- Left-hand steering remapped: two fingers = right, fist = left, palm = neutral.
- Status line shows which hands are detected and the active controls.

### Update 4 — Two-hand racing
- Game mode now tracks **both hands**: one steers, one pedals, so you can hold
  gas and steer at the same time (real combos).
- Four independent controls (gas, brake, left, right), each with an enable
  checkbox and a configurable key.
- UI reorganised with a scroll area and proper spacing so panels never overlap.

### Update 3 — Game & Cricket modes
- Added **Game** mode: palm/fist hold configurable keys to control simple
  two-key games (e.g. Hill Climb Racing).
- Added **Cricket** mode: full-body pose tracking detects a downward arm swing
  and fires a click to bat in simple browser cricket games.

### Update 2 — Transparent overlay & UI refresh
- Air Canvas / Air Writing now draw on a **transparent full-screen overlay** over
  the desktop, instead of a black window.
- Fixed Air Writing disabling itself (the overlay no longer steals keyboard
  focus, so typed characters reach the real target app).
- Sleek dark theme and refined layout.

### Update 1 — Core app
- Unified Mouse, Canvas, and Writing into one PyQt5 app with a live preview,
  adjustable settings, remappable gestures, and persistent configuration.

---

## Tech notes

- **Why scikit-learn instead of TensorFlow?** TensorFlow's protobuf requirement
  conflicts with MediaPipe's. scikit-learn coexists cleanly, so the recognition
  models use an MLP classifier.
- **Handwriting accuracy** is inherently harder than digits; the confirm-each-
  character design avoids the much harder problem of segmenting joined letters.
- **Hands vs. Pose** — most modes use MediaPipe Hands (21 points); Cricket mode
  switches to MediaPipe Pose (33 body points) to read a full-body swing, running
  one model at a time to stay light.
- **Latency** — gesture and especially body-tracking modes add some delay from
  the camera-to-action pipeline, so fast-reaction games feel best when forgiving.
- **Why gesture control suits simple games** — body gestures are a low-bandwidth
  input (a few clear signals at a time), so they fit games needing few, forgiving
  inputs (Hill Climb, Doodle Cricket) and not games needing many precise
  simultaneous inputs (full cricket sims).

## License

Add a license of your choice (e.g. MIT) before publishing if you want others to
reuse the code.
