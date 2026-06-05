# Deaf and Dum - Sign-to-Text and Text-to-Sign Web App

This project is a Flask-based web application with:

- Real-time **Sign-to-Text** prediction from webcam frames.
- **Text-to-Sign** playback using sign video assets.
- A **dataset collector** for building training data.
- A reproducible **model training script** for the sign classifier.

## 1. What This Project Does

### Sign-to-Text

- Captures webcam frames in browser.
- Detects a hand using `cvzone` (or `mediapipe` fallback).
- Draws a centered 400x400 hand skeleton image.
- Runs the skeleton through a CNN model (`cnn8grps_rad1_model.h5`).
- Applies geometric post-rules to convert group-level prediction into final symbols.
- Updates sentence text and word suggestions in real-time.

### Text-to-Sign

- Takes user sentence input.
- Uses NLTK tokenization + light lemmatization + tense cues.
- Resolves each token to `.mp4` signs in `assets/`.
- Falls back to character-level spelling where word sign is unavailable.

### Collector

- Captures and saves labeled hand samples for dataset building.
- Supports multiple output modes:
  - `skeleton`
  - `gray`
  - `binary`
  - `gray_with_drawing`

## 2. Project Structure

```text
deaf_and_dum/
  app.py
  sign_engine.py
  collection_engine.py
  train_sign_model.py
  cnn8grps_rad1_model.h5
  data.jpg
  requirements.txt
  templates/
  static/
  assets/
  AtoZ_3.1/
```

Core files:

- `app.py`: Flask routes and API endpoints.
- `sign_engine.py`: Sign inference pipeline and gesture-rule logic.
- `collection_engine.py`: Dataset capture pipeline.
- `train_sign_model.py`: Model training script matching runtime architecture.

## 3. Architecture

### 3.1 Runtime Components

1. Browser (`static/js/sign-to-text.js`) sends frame -> `/api/predict`.
2. `app.py` decodes image and calls `SignLanguageEngine.process_frame`.
3. `sign_engine.py`:
   - finds hand landmarks,
   - builds normalized skeleton image,
   - predicts via CNN,
   - applies rule logic,
   - returns `symbol`, `sentence`, suggestions, and skeleton preview.
4. UI updates text and suggestions continuously.

### 3.2 Model Architecture (Current H5)

The model is a Sequential CNN:

1. `Conv2D(32, 3x3, relu)` + `MaxPool`
2. `Conv2D(32, 3x3, relu)` + `MaxPool`
3. `Conv2D(16, 3x3, relu)` + `MaxPool`
4. `Conv2D(16, 3x3, relu)` + `MaxPool`
5. `Flatten`
6. `Dense(128, relu)` + `Dropout(0.5)`
7. `Dense(96, relu)` + `Dropout(0.4)`
8. `Dense(64, relu)`
9. `Dense(8, softmax)`

Compile configuration:

- Optimizer: `Adam(learning_rate=0.001)`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`

### 3.3 8-Group Output -> Final Symbol

The CNN predicts 8 coarse groups, then rule logic maps to letters/special controls.

Group mapping used by training script:

- `0`: A, E, M, N, S, T
- `1`: B, D, F, I, K, R, U, V, W
- `2`: C, O
- `3`: G, H
- `4`: L
- `5`: P, Q, Z
- `6`: X
- `7`: J, Y

Special control symbols handled in `sign_engine.py`:

- `"next"`: commit last stable symbol to sentence
- `"Backspace"`: remove last character

## 4. Setup

## 4.1 Prerequisites

- Python 3.10+ (3.10 recommended with current dependencies)
- Webcam access (for live prediction/collection)

## 4.2 Install

```powershell
cd d:\deaf_and_dum
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4.3 Run App

```powershell
python app.py
```

Then open:

- `http://127.0.0.1:5000/` (Sign-to-Text)
- `http://127.0.0.1:5000/text-to-sign`
- `http://127.0.0.1:5000/collector`
- `http://127.0.0.1:5000/predictor`

## 5. API Summary

### `POST /api/predict`

Input JSON:

```json
{ "image": "data:image/jpeg;base64,..." }
```

Output (example keys):

- `has_hand`
- `symbol`
- `sentence`
- `word`
- `suggestions`
- `group_top3`
- `skeleton_image`

### `POST /api/clear`

- Clears current sentence buffer.

### `POST /api/suggestion`

Input:

```json
{ "index": 1 }
```

- Applies suggestion slot 1..4.

### `POST /api/collector/process`

Input (example):

```json
{
  "image": "data:image/jpeg;base64,...",
  "label": "A",
  "dataset_dir": "collected_data",
  "save": true,
  "modes": ["skeleton", "gray"]
}
```

## 6. Training the Model

Training script: `train_sign_model.py`

What it provides:

- Recreates the same runtime architecture/compile setup.
- Supports A-Z class folders (like `AtoZ_3.1`) with automatic 8-group mapping.
- Supports direct group folders (`0..7`) or custom map via `--mapping-json`.
- Uses memory-safe `tf.data` streaming.

### 6.1 Train Command

```powershell
python train_sign_model.py `
  --dataset-dir d:\deaf_and_dum\AtoZ_3.1 `
  --output d:\deaf_and_dum\cnn8grps_rad1_model.h5 `
  --epochs 25 `
  --batch-size 32 `
  --val-split 0.2
```

Artifacts:

- Model: `*.h5`
- Metadata: `*.meta.json`

## 7. `data.jpg` Note (`ch` -> Backspace)

You mentioned your reference image is `data.jpg` and in your use-case **`ch` means Backspace**.

Project note:

- Runtime control token used by engine is `"Backspace"`.
- If your labeling convention uses `"ch"` for delete/backspace, document it as:
  - `ch` (dataset/concept label) -> `Backspace` (runtime control token)

Important:

- `data.jpg` is a reference chart image with multiple signs.
- The live predictor expects camera frames with a detectable hand.
- For stable `Backspace` behavior in inference, keep lighting clear and hand centered.

## 8. Troubleshooting

### No hand detected

- Improve lighting and contrast.
- Keep one hand clearly inside camera frame.
- Check webcam permissions in browser.

### Import error for hand detector

Install one of:

- `cvzone`
- `mediapipe`

### Slow training

- CPU-only training is expected to be slower.
- Use smaller `--epochs` for quick iteration.
- Reduce image size only if you are also updating runtime preprocessing to match.

### Word suggestions missing

- Ensure `pyenchant` is installed and dictionary available on OS.

## 9. Known Behavior and Design Notes

- Sign-to-text is a hybrid approach:
  - CNN classification
  - geometric rule disambiguation
  - temporal/command logic (`next`, `Backspace`)
- This gives practical robustness for webcam usage but is sensitive to hand detection quality.

## 10. Recommended Workflow

1. Use `/collector` to gather cleaner samples.
2. Train with `train_sign_model.py`.
3. Replace model file used by app.
4. Restart Flask app and validate in `/sign-to-text`.

