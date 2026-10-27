# Human Activity Recognition (HAR) — System Architecture & Theoretical Foundations

## 1. Executive Summary & Problem Formulation
Vision-based Human Activity Recognition (HAR) aims to classify human movements, actions, and interactions from continuous video streams. 
Formally, given a sequence of video frames:
$$\mathcal{V} = \{I_1, I_2, \dots, I_T\}, \quad I_t \in \mathbb{R}^{H \times W \times C}$$
the goal is to predict the categorical action label $\hat{y} \in \{1, \dots, K\}$ maximizing the posterior:
$$\hat{y} = \arg\max_{k} P(y = k \mid \mathcal{V})$$

Conventional approaches rely heavily on 3D Convolutional Neural Networks (3D-CNNs) such as C3D and I3D. However, computing 3D convolutions across space and time incurs cubic complexity $\mathcal{O}(T \cdot H \cdot W \cdot C \cdot K^3)$, consuming over 30M-45M parameters and requiring 100+ GFLOPs. Consequently, 3D-CNNs struggle to achieve real-time throughput on edge devices and consumer laptop GPUs.

To overcome these computational bottlenecks while retaining high discriminative capability, this project implements a **Lightweight Hybrid Architecture** combining:
1. **Efficient 2D CNN Spatial Encoder** (MobileNetV3-Small)
2. **Temporal Multi-Head Self-Attention Transformer**
3. **Auxiliary Skeletal Pose Guidance** (MediaPipe BlazePose 33 3D Keypoints)
4. **Multimodal Late/Cross-Attention Fusion**

---

## 2. End-to-End Pipeline Architecture

```
[Input Video Stream]
         │
         ├───> [Uniform / Sliding Frame Sampling (T=16, 224x224)]
         │                 │
         │                 ├───> [2D CNN Spatial Backbone (MobileNetV3-Small)]
         │                 │          │
         │                 │          └──> Frame Embeddings: (B, 16, 256)
         │                 │                       │
         │                 │                       v
         │                 │             [Positional Encoding]
         │                 │                       │
         │                 │                       v
         │                 │         [Temporal Multi-Head Attention]
         │                 │          (2 Layers, 8 Heads, Mean Pool)
         │                 │                       │
         │                 │                       v
         │                 │            Visual Embedding: (B, 256)
         │                 │                       │
         │                 └───> [MediaPipe BlazePose 33 Keypoints]
         │                                         │
         │                                         v
         │                               [Pose Encoder (MLP + Temporal Attn)]
         │                                         │
         │                                         v
         │                              Pose Embedding: (B, 64)
         │                                         │
         │                                         v
         └───────────────────────────────> [Multimodal Fusion Layer]
                                                   │
                                                   v
                                          [Classifier Head]
                                                   │
                                                   v
                                          [Softmax Action Logits]
```

---

## 3. Mathematical Subsystem Formulations

### 3.1 2D CNN Spatial Encoder (MobileNetV3-Small)
Each video frame $I_t \in \mathbb{R}^{3 \times 224 \times 224}$ is projected into a lower-dimensional feature representation using depthwise separable convolutions and squeeze-and-excitation blocks:
$$f_t = \text{Linear}(\text{GlobalAvgPool}(\text{Backbone}(I_t))) \in \mathbb{R}^{d_{model}}$$
where $d_{model} = 256$. For a clip of length $T=16$, the visual sequence matrix is:
$$F_V = [f_1, f_2, \dots, f_T]^\top \in \mathbb{R}^{T \times d_{model}}$$

### 3.2 Sinusoidal Positional Encoding & Temporal Attention
Because standard Self-Attention is permutation-invariant, temporal ordering must be explicitly injected:
$$PE(t, 2i) = \sin\left(\frac{t}{10000^{2i / d_{model}}}\right), \quad PE(t, 2i+1) = \cos\left(\frac{t}{10000^{2i / d_{model}}}\right)$$
$$Z_0 = F_V + PE \in \mathbb{R}^{T \times d_{model}}$$

The temporal transformer applies $L=2$ Multi-Head Self-Attention layers:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
Global temporal pooling produces the compact visual embedding:
$$v_{\text{temporal}} = \frac{1}{T} \sum_{t=1}^T Z_L[t] \in \mathbb{R}^{256}$$

### 3.3 Auxiliary Skeletal Pose-Guided Branch
For each frame $I_t$, MediaPipe BlazePose detects 33 body keypoints:
$$p_t = [x_1, y_1, v_1, x_2, y_2, v_2, \dots, x_{33}, y_{33}, v_{33}]^\top \in \mathbb{R}^{99}$$
The pose sequence $P = [p_1, \dots, p_T]^\top \in \mathbb{R}^{T \times 99}$ is encoded via a 2-layer MLP with temporal attention pooling:
$$e_t = \text{GELU}(\text{LayerNorm}(W_1 p_t + b_1))$$
$$u_{\text{pose}} = \sum_{t=1}^T \alpha_t (W_2 e_t + b_2) \in \mathbb{R}^{64}$$
where $\alpha_t = \text{softmax}(w_a^\top \tanh(W_s e_t))$.

### 3.4 Multimodal Fusion & Classification Head
The visual vector $v_{\text{temporal}} \in \mathbb{R}^{256}$ and pose vector $u_{\text{pose}} \in \mathbb{R}^{64}$ are concatenated:
$$h_{\text{fused}} = \text{GELU}(W_f [v_{\text{temporal}} \,\|\, u_{\text{pose}}] + b_f) \in \mathbb{R}^{256}$$
The classification head outputs class logits:
$$\hat{z} = W_c (\text{Dropout}(h_{\text{fused}})) + b_c \in \mathbb{R}^K$$
$$\hat{y}_k = \frac{\exp(\hat{z}_k)}{\sum_{j=1}^K \exp(\hat{z}_j)}$$

---

## 4. Hardware Optimization & Real-Time Performance
- **Model Parameters**: 2,288,999 (2.28 Million)
- **Trainable Parameters**: 2,288,999 (100% fine-tuned)
- **Inference Latency on RTX 2050 (4GB VRAM)**: ~9.2 ms per clip
- **Theoretical Throughput**: ~108.6 FPS (Well above 30 FPS real-time threshold)
- **CPU Fallback Latency**: ~32-38 ms per clip (~28 FPS)
