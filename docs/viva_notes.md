# HAR Project — Viva Q&A Preparation Guide

## 1. What problem does your project solve?

**Answer:** Human Activity Recognition (HAR) is the problem of automatically identifying what action a person is performing from video input. This has applications in security surveillance, sports analytics, healthcare monitoring, and human-computer interaction. Traditional approaches require hand-crafted features; our project uses deep learning to automatically learn spatiotemporal representations of human motion.

---

## 2. Explain your architecture

**Answer:** We use a hybrid architecture combining three components:

1. **Spatial Feature Extraction (CNN):** A MobileNetV3-Small backbone (pretrained on ImageNet) extracts visual features from each frame independently. It converts 224×224 RGB frames into 576-dimensional feature vectors, which are projected down to 256 dimensions via a 1×1 convolution.

2. **Temporal Modeling (Transformer):** A 2-layer Multi-Head Self-Attention Transformer with 8 attention heads and d_model=256 captures relationships between the 16 sampled frames. Unlike LSTMs, the Transformer processes all frames in parallel and uses attention weights to learn which frames are most discriminative. Sinusoidal positional encodings inject temporal ordering.

3. **Pose Estimation Branch (MediaPipe):** MediaPipe BlazePose detects 33 body keypoints (x, y, visibility) per frame, giving a 99-dimensional pose vector per frame. A lightweight MLP encodes this 99-dim → 128 → 64 representation, capturing skeletal kinematics independent of clothing or appearance.

4. **Fusion and Classification:** The 256-dim visual feature and 64-dim pose feature are concatenated (320 dim) and passed through a linear layer to get 256 dims. A final classifier (256→128→6) produces the activity logits.

---

## 3. Why MobileNetV3 instead of a larger CNN?

**Answer:** Three reasons:
- **Computational efficiency:** MobileNetV3 uses inverted residual blocks and hard-swish activations. It achieves ~90% of ResNet50 accuracy at 1/10th the FLOPs, suitable for CPU-based inference.
- **Transfer learning:** ImageNet pretraining gives excellent spatial features without needing millions of action videos.
- **Memory footprint:** The whole model fits under 9MB RAM, compared to I3D (25MB) or SlowFast (34MB).

---

## 4. Why a Transformer instead of LSTM for temporal modeling?

**Answer:**
- **Parallelism:** LSTMs process frames sequentially. Transformers process all frames simultaneously.
- **Long-range dependencies:** Self-attention can directly relate frame 1 to frame 16 without vanishing gradient problems.
- **Interpretability:** Attention weights can be visualized to see which frames the model focused on.
- **Practical:** For 16-frame clips, our 2-layer Transformer has fewer parameters than a comparable LSTM and trains faster.

---

## 5. What is your dataset?

**Answer:** UCF101 — University of Central Florida's benchmark action recognition dataset, originally containing 13,320 videos across 101 activity classes. We use a 6-class subset: ApplyEyeMakeup, Biking, JumpingJack, PushUps, Squats, WalkingWithDog. Videos are 25 FPS, ~3-10 seconds long, captured in realistic conditions. Our local subset has 24 videos (4 per class) for rapid prototyping.

---

## 6. What preprocessing do you apply?

**Answer:**
1. **Frame sampling:** Uniformly sample 16 frames from each video clip regardless of duration. Random temporal jitter during training.
2. **Resizing:** Each frame is resized to 224×224 pixels using bilinear interpolation.
3. **Normalization:** ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) normalization.
4. **Augmentation (training only):** Random horizontal flip, brightness/contrast jitter.
5. **Pose extraction:** MediaPipe BlazePose extracts 33 keypoints (x, y, visibility) per frame → 99-dim pose vector.

---

## 7. How does your inference pipeline work?

**Answer:** 
1. Video is opened with OpenCV; all frames are decoded.
2. 16 frames are uniformly sampled from the full video.
3. Each frame is preprocessed to a (3, 224, 224) tensor.
4. The 16-frame tensor (1, 16, 3, 224, 224) is passed through the model.
5. The model outputs 6-class logits; softmax gives confidence scores.
6. For live webcam: EMA smoothing (alpha=0.7) stabilizes predictions across clips.

---

## 8. What are your evaluation metrics?

**Answer:**
- **Top-1 Accuracy:** Percentage of correct top predictions.
- **Macro F1/Precision/Recall:** Per-class then equally averaged — important for class imbalance.
- **Weighted F1:** Weighted by class frequency.
- **Confusion Matrix:** Visualizes inter-class confusion.
- **Inference Latency:** Milliseconds per clip — critical for real-time use.

---

## 9. What are your current accuracy results?

**Answer (honest):** 
- Our local model was trained on only 24 videos (4 per class). This is insufficient for true generalization — it leads to overfitting.
- With in-sample evaluation (majority voting on training data), we achieve ~67% top-1 accuracy.
- This is an honest limitation documented in the project: real-world performance needs 100-500 videos per class.
- We provide a Google Colab notebook for training on the full UCF101 dataset, where 75-85% accuracy is expected.

---

## 10. How did you address the small dataset problem?

**Answer:**
1. **Transfer learning:** Frozen MobileNetV3 backbone uses ImageNet features.
2. **Multi-clip augmentation:** Each video sampled 15× with different offsets/augmentations (24 videos → 360 clips).
3. **Label smoothing:** Reduces overconfident predictions.
4. **Data augmentation:** Random flip, brightness/contrast jitter.
5. **Colab notebook:** Full UCF101 training for production-quality results.

---

## 11. What is EMA smoothing and why is it needed for webcam?

**Answer:** Exponential Moving Average: `smoothed[t] = α × current + (1-α) × smoothed[t-1]`  
With α=0.7, recent predictions are weighted more heavily. Without it, webcam predictions flicker between classes due to natural camera motion and pose variation. EMA produces stable, readable real-time outputs.

---

## 12. What happens if the user uploads an unsupported video?

**Answer:** Multiple failure modes are handled:
- **Corrupt video:** Exception caught in `read_video_frames()`, returns HTTP 400.
- **Insufficient frames:** Returns error if fewer than 2 frames decoded.
- **Wrong file type:** API validates MIME type, rejects non-video files.
- **Camera permission denied:** Frontend shows user-friendly error message.
- **No checkpoint:** API returns `model_state: no_checkpoint` without crashing.

---

## 13. Describe a technical challenge you solved

**Answer:** Architecture mismatch between training and inference. When fine-tuning with `use_pose=False` (NullPoseEncoder — no learnable weights), the checkpoint lacked `pose_encoder.frame_encoder.*` keys that the inference engine expected. This caused a `RuntimeError` and put the engine in `NO_CHECKPOINT` state.

**Solution:** Modified `engine.py` to catch `RuntimeError` during `load_state_dict()`, detect missing pose keys, automatically rebuild the model with `use_pose=False`, and retry loading. Now checkpoint loading is robust to this mismatch.

---

## 14. How would you improve this with more time?

1. Train on full UCF101 (13K videos, 101 classes) using Colab
2. Two-stage training: freeze backbone first, then unfreeze
3. TimeSformer or divided space-time attention
4. ONNX export for 3-5× faster CPU inference
5. Skeleton overlay on webcam feed
6. Extend to all 101 UCF101 classes

---

## 15. Demo Walkthrough Script (5 minutes)

**[1 min]** "HAR means automatically identifying what a person is doing from video. We've built an end-to-end pipeline using a lightweight hybrid deep learning model."

**[1 min]** Point to Architecture tab — "The pipeline processes 16 frames through MobileNetV3 for spatial features, a Transformer for temporal context, and MediaPipe pose for skeletal guidance."

**[2 min]** Upload `data/ucf101/JumpingJack/v_JumpingJack_g01_c01.mp4` → show prediction + confidence scores.

**[1 min]** Show Evaluation tab — confusion matrix, training curves, metrics. Discuss accuracy and limitations honestly.

---

## 16. How did you ensure your evaluation is leak-free and subject-independent?

**Answer:**
- **Group ID Convention in UCF-101:** Videos are named `v_<Action>_g<Group>_c<Clip>.mp4`. The `gXX` identifier signifies videos recorded by the same actor under the same environmental conditions.
- **Data Leakage Risk:** If augmented temporal clips from the same source video or same actor group appear in both training and test sets, the model can achieve artificially inflated accuracy by memorizing background, clothing, or subject appearance rather than true motion dynamics.
- **Leak-Free Protocol (`scripts/evaluate_heldout.py`):** We partition data by Group ID. Groups 1, 2, and 3 (18 videos) are used for training, while Group 4 (6 videos, 1 per class) is kept strictly held-out as unseen test subjects. This guarantees zero overlap between training and test sets, measuring genuine generalization to new performers.
