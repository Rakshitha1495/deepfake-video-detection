# DeepGuard AI — Deepfake Video Detection with CNN-LSTM

*Detect. Analyze. Verify.*

A lightweight deepfake video detector built for a CPU-only Windows laptop. It trains **once**, saves a checkpoint, and from then on the backend just loads that checkpoint at startup and reuses it for every video.

```
Video ─► sample N frames ─► center-crop 224×224 ─► ResNet18 (ImageNet, frozen)
      ─► 512-d feature per frame ─► LSTM (hidden 128) ─► Linear ─► Softmax ─► Real / Fake
```

> **Important:** this is an experimental student model trained on a small dataset. Its output is a probability, not proof that a video was or was not manipulated.

---

## 1. Project structure

```
deepfake-video-detection/
├── dataset/raw/{real,fake}/     put your videos here
├── models/best_model.pt         created by training
├── artifacts/                   training_history.json, evaluation_metrics.json,
│                                confusion_matrix.png, benchmark.json
├── src/
│   ├── __init__.py
│   ├── config.py                paths + default hyper-parameters
│   ├── video_utils.py           fast OpenCV frame sampling + preprocessing
│   ├── data_split.py            video discovery + leak-free group split
│   ├── dataset.py               PyTorch datasets + one-time feature cache
│   ├── model.py                 CNN-LSTM model, checkpoint save/load
│   ├── metrics.py               accuracy / precision / recall / F1 / ROC-AUC
│   ├── train.py                 training (best epoch by validation F1)
│   ├── evaluate.py              final test-set evaluation
│   └── predict.py               Predictor class + command-line prediction
├── backend/
│   ├── __init__.py
│   ├── config.py  schemas.py  inference.py  main.py
├── frontend/                    React + Vite + TypeScript + Tailwind dashboard
├── scripts/benchmark.py         measures real timings on your laptop
├── tests/                       pytest suite
├── initialize.py                one-time setup + training script
├── requirements.txt  .env.example  .gitignore
```

(The brief's `init.py` files are Python's package marker `__init__.py`.)

---

## 2. First-time setup (Windows PowerShell)

### 2.1 Install Python packages (once)

You need **Python 3.11** and **Node.js 18+**.

```powershell
cd deepfake-video-detection
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

On Windows, `pip install torch torchvision` installs the CPU build. No NVIDIA GPU is needed.

### 2.2 Add videos

Put 20–50 real and 20–50 fake videos here:

```
dataset\raw\real\   (label 0)
dataset\raw\fake\   (label 1)
```

MP4, AVI, MOV and MKV are accepted (whatever OpenCV can decode). Good public sources, all of which need an access request or agreement:

* **Celeb-DF (v2)** — filenames like `id0_0003.mp4` and `id0_id16_0003.mp4`. DeepGuard links these automatically.
* **FaceForensics++** — filenames like `000.mp4` (real) and `000_003.mp4` (fake). Also linked automatically.
* **DFDC preview / Kaggle subsets** — use a `groups.csv` (see below).

**Why groups matter.** A fake video is usually made from a real one. If the original is in training and its fake copy is in testing, the test score looks better than it really is. So the split is done by *source group*, never by single file.

For any other naming scheme, create `dataset\raw\groups.csv`:

```csv
path,group
real/interview_01.mp4,person_a
fake/interview_01_swap.mp4,person_a
real/vlog_07.mp4,person_b
```

Without a groups file or a known naming pattern, each video becomes its own group and a note is recorded in the training history.

Tip for speed: short clips (under 20 seconds) and 720p or lower decode much faster.

### 2.3 Train once

```powershell
python initialize.py
```

This script:

1. checks the Python version and virtual environment,
2. checks every required package,
3. checks whether the ResNet18 weights are already cached in `.cache\torch`,
4. counts your real and fake videos,
5. checks for `models\best_model.pt` — **if it exists, training is skipped**,
6. otherwise trains, saves the checkpoint and history, and evaluates on the test split.

Useful options (forwarded to training):

```powershell
python initialize.py --epochs 3 --frames 6 --max-videos-per-class 30
python initialize.py --force          # retrain even though a checkpoint exists
python initialize.py --check          # checks only, never trains
```

What gets written:

| File | Content |
|---|---|
| `models\best_model.pt` | weights + config + the exact train/val/test file lists |
| `artifacts\training_history.json` | per-epoch loss/accuracy/F1, timings, split, warnings |
| `artifacts\evaluation_metrics.json` | test accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| `artifacts\confusion_matrix.png` | the confusion matrix as an image |

If training runs past the 20-minute target, the console prints which stage was slowest (decoding, ResNet18 features or LSTM epochs) and which settings to reduce. Training is **not** stopped silently. Add `--stop-at-budget` if you want it to stop after the epoch that crosses the limit; the status is then recorded as stopped, not completed.

### 2.4 Start the backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

API docs: <http://127.0.0.1:8000/docs>

### 2.5 Start the frontend (second terminal)

```powershell
cd frontend
npm install                 # first time only
Copy-Item .env.example .env # first time only
npm run dev
```

Open <http://127.0.0.1:5173>, go to **Analyze a video**, drop a clip and press **Analyze video**.

---

## 3. Every launch after that

No training, no downloads, no dataset processing:

```powershell
# Terminal 1
cd deepfake-video-detection
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2
cd deepfake-video-detection\frontend
npm run dev
```

## 4. Other commands

```powershell
python -m src.evaluate                                   # re-run test evaluation only
python -m src.train --epochs 5                           # train directly (overwrites the checkpoint)
python -m src.predict path\to\video.mp4                  # predict from the command line
python scripts\benchmark.py path\to\video.mp4 --repeats 5  # measure timings, saves artifacts\benchmark.json
pytest -q                                                # run tests
cd frontend; npm run build                               # type-check + production build
```

---

## 5. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | backend status, whether the model is loaded, device |
| GET | `/api/model-info` | architecture, frame count, training config, split, real metrics, training history |
| POST | `/api/predict` | multipart field `file`; returns prediction, probabilities, timings |
| GET | `/api/artifacts/confusion-matrix` | the confusion matrix PNG |

`/api/predict` checks the extension (400), size limit (413, default 100 MB), that the file really decodes as video (422), and returns 503 with the training command if no model is loaded. Uploads are streamed to a temp file and deleted in a `finally` block.

Example response shape (values come from the real model):

```json
{
  "filename": "clip.mp4",
  "prediction": "Fake",
  "probabilities": { "real": 0.2, "fake": 0.8 },
  "processing_time_seconds": 3.1,
  "frames_processed": 8,
  "model_loaded": true,
  "timing": { "upload_seconds": 0.1, "decode_seconds": 0.9, "preprocess_seconds": 0.01,
              "inference_seconds": 0.6, "model_seconds": 1.5 },
  "video": { "fps": 30.0, "total_frames": 450, "sampled_indices": [0, 64, "..."] },
  "warning": "Experimental model trained on a small dataset. ..."
}
```

Settings live in `.env` (copy `.env.example`): `MODEL_PATH`, `DEVICE`, `MAX_UPLOAD_MB`, `ALLOWED_ORIGINS`, `TORCH_THREADS`. The frontend reads `VITE_API_BASE_URL` from `frontend\.env`.

---

## 6. How the speed targets are met

| Requirement | How it is done |
|---|---|
| Pretrained weights downloaded once | `TORCH_HOME` points to `.cache\torch`; `initialize.py` reports whether they are cached |
| No retraining on launch | the backend never trains; missing checkpoint → clear message |
| Model loaded once | loaded in FastAPI's startup (`lifespan`) and kept in `ModelService` |
| No weight download at inference | the model is rebuilt with `weights=None`, then every weight comes from the checkpoint |
| Fast training | ResNet18 is frozen, so each video's features are computed **once** and cached in `.cache\features`; epochs then train only the small LSTM + linear layer |
| Re-running training is cheap | the feature cache is reused (keyed by file, size, date and frame count) |
| Fast decoding | only the sampled frames are converted; long videos use seeking, short ones `grab()` without decoding to images |
| Batched CNN | all N frames of a video go through ResNet18 as one batch |
| Inference mode | `model.eval()` + `torch.inference_mode()`; a warm-up pass at startup |
| No wasted memory | uploads stream to disk in 1 MB chunks; temp files always deleted |
| CPU contention | one inference at a time (a lock), so two uploads don't fight over the 4 cores |

**Timings.** I could not run PyTorch in the environment where this code was written, so no timings on your laptop are claimed here. Measure them yourself:

* training time is printed at the end of training and saved in `training_history.json`,
* each prediction returns its own measured timings (also shown in the UI),
* `scripts\benchmark.py` records import time, model loading, first and repeated predictions, per-stage timings and peak memory into `artifacts\benchmark.json`.

What to expect qualitatively: ResNet18 on 8 frames is a small amount of work for a modern laptop CPU, so for short clips **video decoding** is usually the largest part. If a target is missed, the benchmark shows which stage is slow; the fixes are in section 8.

---

## 7. Honest limitations

* 40–100 videos is enough to prove the pipeline works, not to build a reliable detector. Expect noisy metrics; a handful of test videos means every video moves accuracy by several percentage points (the evaluation file says this explicitly).
* Frames are center-cropped, not face-cropped. Adding a face detector (e.g. OpenCV DNN or MediaPipe) before the crop usually helps more than any other change.
* The model learns the artefacts of the datasets it saw. It will be less reliable on new generation methods, heavy compression, or very different videos.
* ROC-AUC is only reported when the test split contains both classes.
* Session history in the UI is session-only (browser `sessionStorage`); nothing is stored on the server.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| UI says "Backend offline" | start uvicorn; check `VITE_API_BASE_URL` in `frontend\.env` |
| UI says "Model not trained" | `python initialize.py`, then restart uvicorn |
| CORS error in browser console | add your frontend URL to `ALLOWED_ORIGINS` in `.env` |
| Training too slow | `--frames 5 --max-videos-per-class 20 --epochs 3`; trim clips; downscale to 480p |
| Prediction slow | use shorter clips; try `TORCH_THREADS=4` in `.env` |
| A video is skipped | it could not be decoded; re-encode with `ffmpeg -i in.avi -c:v libx264 out.mp4` |
| Preview black in browser | the browser cannot play that codec; analysis still works |

## 9. Next steps (after the baseline works)

1. Face cropping before the 224×224 crop.
2. More data, then try `--finetune-backbone` (trains ResNet18 too; much slower on CPU).
3. 5-fold group cross-validation for more stable metrics on small datasets.
