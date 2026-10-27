# Human Activity Recognition using Deep Learning & Computer Vision

An end-to-end Human Activity Recognition (HAR) system using a lightweight hybrid architecture:
**MobileNetV3-Small 2D-CNN** + **Temporal Multi-Head Self-Attention Transformer** + **MediaPipe BlazePose 33-Keypoint Guidance**.

---

## Supported Activity Classes

The system currently classifies **6 UCF101 activity classes**:

| Index | Class | Description |
|-------|-------|-------------|
| 0 | ApplyEyeMakeup | Person applying eye makeup |
| 1 | Biking | Person riding a bicycle |
| 2 | JumpingJack | Jumping jacks exercise |
| 3 | PushUps | Push-up exercise |
| 4 | Squats | Squat exercise |
| 5 | WalkingWithDog | Person walking a dog |

---

## System Architecture

```
[Video Frames: 1×16×3×224×224]
        │
        ▼
[MobileNetV3-Small]  → CNN Spatial Features (16×576 → projected → 16×256)
        │
        ▼
[Sinusoidal Positional Encoding] + [Temporal Transformer Encoder (2L, 8H, d=256)]
        │ mean-pool → (1×256)
        │
[Video Frames] → [MediaPipe BlazePose] → [Pose MLP Encoder] → (1×64)
        │                                        │
        └─────────────┬───────────────────────── ┘
                      ▼
           [Fusion: Concat → Linear (320→256)]
                      │
                      ▼
           [Classifier: 256→128→6 logits]
                      │
                      ▼
           [EMA Smoothing (α=0.7) for live predictions]
```

**Total Parameters:** 2,288,999 (~2.3M — 90% smaller than I3D/C3D)

---

## Quick Start

### Prerequisites
- Python 3.11 (embedded — already included at `../python/`)
- Node.js (embedded — already included at `../node/`)

### 1. Run Backend Server
```powershell
.\run_backend.bat
```
FastAPI backend starts at `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

### 2. Run Frontend Web App
```powershell
.\run_frontend.bat
```
React dashboard at `http://localhost:5173`

### 3. Run Tests
```powershell
.\run_tests.bat
```
Runs 15 pytest tests (unit + API integration).

### 4. Run Local Fine-Tuning
```powershell
.\run_finetune.bat
```
Fine-tunes the model on the local 24-video dataset (~15 min on CPU).

### 5. Train from Scratch (full UCF101 data)
Use the Google Colab notebook for GPU training on larger data:
`scripts/colab_training.ipynb`

---

## Installation (Clean Environment)

If running on a different machine:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For CPU-only PyTorch (no CUDA):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Directory Structure

```
har-project/
├── backend/
│   ├── app/main.py              # FastAPI server + all API endpoints
│   ├── inference/engine.py      # HARInferenceEngine (rolling window, EMA)
│   ├── models/
│   │   ├── cnn_encoder.py       # MobileNetV3/EfficientNet/ResNet backbones
│   │   ├── temporal_transformer.py  # Transformer + LSTM/GRU temporal modules
│   │   ├── pose_encoder.py      # BlazePose 33-point MLP encoder
│   │   ├── fusion_model.py      # Concat/add/attention multimodal fusion
│   │   ├── classifier.py        # MLP classification head
│   │   └── har_model.py         # Complete unified HARModel
│   ├── preprocessing/
│   │   ├── frame_processor.py   # Frame sampling, resizing, normalization
│   │   └── dataset_loader.py    # UCF101 dataset loading, splits, DataLoader
│   ├── pose/pose_estimator.py   # MediaPipe BlazePose wrapper
│   ├── training/
│   │   ├── trainer.py           # Training loop, AMP, Cosine LR, checkpointing
│   │   └── evaluator.py         # Metrics, confusion matrix, training curves
│   └── config.yaml              # All hyperparameters and paths
├── frontend/                    # React 18 + Vite + Tailwind CSS dashboard
│   └── dist/                    # Pre-built production bundle (ready to serve)
├── data/
│   ├── ucf101/                  # 24 UCF101 videos (6 classes × 4 videos)
│   └── class_names.json         # Ordered list of class names
├── checkpoints/
│   └── best_model.pth           # Best trained checkpoint
├── results/
│   ├── metrics.json             # Evaluation metrics (accuracy, F1, etc.)
│   ├── confusion_matrix.png     # Per-class confusion matrix
│   ├── training_curves.png      # Training loss curve
│   └── classification_report.txt
├── scripts/
│   ├── train.py                 # Full training script (requires full UCF101)
│   ├── finetune_local.py        # Fine-tune with train/val split
│   ├── finetune_alldata.py      # Fine-tune on all available data (recommended for demo)
│   ├── evaluate.py              # Standalone evaluation
│   ├── generate_sample_data.py  # Synthetic test video generator
│   └── colab_training.ipynb     # Google Colab GPU training notebook
├── tests/
│   ├── test_components.py       # 8 unit tests for model components
│   └── test_api.py              # 7 API integration tests
├── docs/
│   ├── architecture.md          # Technical architecture documentation
│   └── viva_notes.md            # Viva Q&A preparation guide
├── requirements.txt             # Python dependencies
├── run_backend.bat              # Start backend server
├── run_frontend.bat             # Start frontend
├── run_tests.bat                # Run all tests
├── run_train.bat                # Full training (requires more UCF101 data)
└── run_finetune.bat             # Local fine-tuning on included 24 videos
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + model state |
| GET | `/api/model-info` | Architecture details |
| GET | `/api/classes` | Supported activity classes |
| GET | `/api/metrics` | Evaluation metrics (from results/) |
| POST | `/api/predict/video` | Upload video → get prediction |
| POST | `/api/predict/frame-sequence` | Send frames → get prediction |
| GET | `/api/inference-stats` | Inference timing stats |
| GET | `/api/results/confusion-matrix` | Serve confusion matrix image |
| GET | `/api/results/training-curves` | Serve training curves image |
| GET | `/api/results/training-history` | Training history JSON |
| POST | `/api/model/load` | Reload model checkpoint |

---

## Evaluation Results

Current best checkpoint was trained on the 24-video local subset using:
- **Frozen MobileNetV3-Small backbone** (pretrained ImageNet weights)
- **Trainable:** Temporal Transformer, Fusion, and Classifier layers
- **Training data:** 24 UCF101 videos (4 per class × 6 classes)

> ⚠️ **Important limitation:** The current model was trained on only 24 short videos.
> Real-world performance requires significantly more training data.
> Use `scripts/colab_training.ipynb` on Google Colab with full UCF101 data for production-quality results.

For true accuracy metrics, see `results/metrics.json` and `results/classification_report.txt`.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel Core i5 / AMD Ryzen 5 | Intel Core Ultra 5+ |
| RAM | 8 GB | 16 GB |
| GPU | Not required | NVIDIA GPU (CUDA) for faster training |
| Storage | 2 GB free | 10 GB (for full UCF101 subset) |

Inference runs on CPU at ~220ms per clip (~4-5 FPS). GPU accelerates this to <10ms.

---

## Demonstration Guide

### Video Upload Demo
1. Start backend: `run_backend.bat`
2. Start frontend: `run_frontend.bat`
3. Open `http://localhost:5173` in browser
4. Click **"Video Analyzer"** tab
5. Drag and drop any of the UCF101 videos from `data/ucf101/<class>/`
6. Click **"Analyze Action"** — results appear with confidence scores

### Live Webcam Demo
1. Both backend and frontend must be running
2. Click **"Live Webcam"** tab
3. Click **"Start Webcam"** — grant camera permission
4. Perform one of the 6 supported activities in front of the camera
5. The system collects 16 frames, runs inference, displays the predicted activity
6. Click **"Stop Camera"** to cleanly release resources

### Evaluation Dashboard
- Click **"Evaluation"** tab to view confusion matrix, training curves, and metrics
- Click **"Architecture"** tab for a visual explanation of the model pipeline

---

## Known Limitations

1. **Small training dataset:** Only 24 videos available locally. True generalization requires 100+ videos per class.
2. **CPU-only inference:** ~220ms per 16-frame clip on Intel Core Ultra 5 (no GPU). Not true real-time.
3. **Webcam latency:** ~1-2 seconds per prediction cycle due to CPU inference time.
4. **6 fixed classes:** Cannot recognize activities outside the 6 trained classes.
5. **MediaPipe dependency:** If pose estimation fails, the model falls back to visual-only inference.

---

## Future Improvements

- Train on the full UCF101 dataset (101 classes, 13,320 videos) on Google Colab
- Unfreeze the backbone after initial convergence (stage-2 training)
- Add subject-independent cross-validation splits
- Enable ONNX export for optimized CPU inference
- Add webcam overlay with skeleton visualization
- Extend to more activity classes

---

## License & Credits

- **Dataset:** UCF101 — University of Central Florida Action Recognition Dataset
- **Backbone:** MobileNetV3-Small (pretrained on ImageNet via torchvision)
- **Pose Estimation:** MediaPipe Pose (Google)
- **Architecture References:** ViViT, TimeSformer (temporal attention for video)
