import React, { useState } from 'react';
import { HelpCircle, ChevronDown, ChevronUp, Search, BookCheck, ShieldAlert, Cpu, Sparkles } from 'lucide-react';

export default function VivaPrep() {
  const [search, setSearch] = useState('');
  const [openIndex, setOpenIndex] = useState(0);
  const [selectedCategory, setSelectedCategory] = useState('All');

  const questions = [
    {
      q: 'Why did you choose a lightweight hybrid 2D CNN + Temporal Attention model over 3D CNNs (like C3D or I3D)?',
      category: 'Architecture',
      answer: `Standard 3D CNNs perform spatio-temporal convolutions across time and space simultaneously. While effective, their computational cost scales cubically: O(T * H * W * C * K³).
1. Computational Complexity: C3D and I3D require 30M-45M parameters and over 100 GFLOPs per forward pass, resulting in frame rates under 15 FPS on consumer laptops.
2. Parameter Efficiency: Our hybrid architecture decouples spatial feature extraction (MobileNetV3: ~1.5M params) from temporal modeling (Temporal Transformer: ~0.7M params), totaling just 2.28M parameters.
3. Latency & Throughput: We achieve sub-10ms latency (~108 FPS on RTX 2050 GPU, and ~35 FPS on CPU), enabling true real-time webcam operation without frame drops.`,
    },
    {
      q: 'What specific advantage does the auxiliary pose branch provide over pure RGB video networks?',
      category: 'Computer Vision',
      answer: `Pure RGB networks frequently suffer from scene bias and shortcut learning—the model predicts "Swimming" simply because it detects blue water pixels, or predicts "Walking" because of outdoor sidewalk textures.
1. Background Invariance: MediaPipe BlazePose extracts 33 body landmarks (99 coordinates) that capture pure skeletal kinematics independent of lighting conditions, clothing, or camera background clutter.
2. Occlusion Robustness: Even if the person's torso or legs are partially shadowed or obscured, the temporal movement of key joints preserves the semantic activity signature.
3. Fallback Grace: If pose estimation fails or has low confidence, our multimodal fusion layer gracefully handles missing coordinates while relying on the visual 2D CNN stream.`,
    },
    {
      q: 'How does the Self-Attention mechanism in the Temporal Transformer outperform an LSTM or GRU?',
      category: 'Deep Learning',
      answer: `Recurrent Neural Networks (LSTMs/GRUs) process video frames sequentially: h_t = f(h_{t-1}, x_t).
1. Sequential Bottleneck: LSTMs cannot parallelize across the time dimension T during training, whereas Multi-Head Self-Attention computes pairwise attention weights A = softmax(QK^T / sqrt(d)) across all frames concurrently in O(1) sequential steps.
2. Long-Range Dependency: Recurrent cells suffer from vanishing gradients and memory decay over longer sequences. Self-Attention provides a direct connection between frame 1 and frame 16 regardless of temporal distance.
3. Interpretability: Attention weights can be extracted and visualized to discover which specific frames (e.g. the peak of a jump or extension of a pushup) influenced the final classification decision.`,
    },
    {
      q: 'Why is Positional Encoding necessary in the Temporal Transformer?',
      category: 'Deep Learning',
      answer: `Self-Attention is permutation invariant: swapping the order of video frames would result in the exact same set of unpooled attention outputs without positional information.
In activity recognition, temporal order is fundamental—"Standing Up" and "Sitting Down" contain identical individual frames, but in reverse chronological order.
We inject sinusoidal positional encodings:
PE(pos, 2i) = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
This gives each temporal token an explicit continuous representation of its relative and absolute position in the clip.`,
    },
    {
      q: 'How does the Real-Time Rolling Window Buffer work without causing stream lag?',
      category: 'Engineering',
      answer: `Our inference engine implements a double-ended queue (deque) with maxlen=16:
1. As webcam frames arrive, they are resized and normalized into memory.
2. Once the buffer holds 16 frames, inference is triggered.
3. With window_stride=1 (or stride=2), only the newest frame is added while the oldest is popped, avoiding redundant full-video decoding.
4. Exponential Moving Average (EMA) smoothing is applied: P_t = alpha * P_{t-1} + (1 - alpha) * P_new (alpha=0.7). This eliminates flickering predictions between adjacent frames and produces silky smooth confidence transitions.`,
    },
    {
      q: 'Why choose MobileNetV3-Small over ResNet-50 or a pure Vision Transformer (ViT)?',
      category: 'Architecture',
      answer: `MobileNetV3-Small is designed specifically through hardware-aware Neural Architecture Search (NAS) and NetAdapt:
1. Inverted Residuals with Linear Bottlenecks: Expands to a higher-dimensional space, applies depthwise separable convolution (reducing MACs by ~8-9x compared to standard convs), and projects back with a linear layer.
2. Squeeze-and-Excitation (SE): Selectively emphasizes informative spatial feature channels at negligible compute cost.
3. Hard-Swish Activation: h-swish(x) = x * ReLU6(x + 3) / 6 replaces expensive sigmoid/tanh transcendental calculations with fast piecewise linear operations, ideal for ARM and mobile instruction sets.`,
    },
    {
      q: 'What evaluation metrics did you use, and why is Top-1 accuracy alone insufficient?',
      category: 'Evaluation',
      answer: `In Human Activity Recognition:
1. Class Imbalance: Video datasets often have uneven sample counts per action. High overall accuracy can hide 0% recall on minority classes.
2. Macro vs Weighted F1: Macro F1 calculates the unweighted mean across all classes: F1_macro = (1/C) * sum(F1_c). This treats every action equally, penalizing failures on rare classes.
3. Confusion Matrix: Pinpoints specific semantic ambiguities (e.g. distinguishing between "PushUps" and "Plank", or "Squats" and "Lunges").
4. Latency & FPS: Academic papers often report accuracy while ignoring that a model takes 200ms per clip. For clinical gait monitoring or sports robotics, sub-30ms latency is an essential constraint.`,
    },
    {
      q: 'What training optimizations and regularization techniques did you implement?',
      category: 'Optimization',
      answer: `1. Mixed Precision (AMP): Automatic Mixed Precision with torch.cuda.amp.autocast and GradScaler halves memory bandwidth and doubles tensor core utilization on RTX GPUs.
2. Gradient Clipping: torch.nn.utils.clip_grad_norm_(max_norm=1.0) prevents exploding gradients during transformer backpropagation.
3. Label Smoothing: Loss = (1 - eps) * CE + eps * Uniform(1/C) with eps=0.1 prevents the softmax layer from becoming overconfident, reducing overfitting on small datasets.
4. Cosine Annealing LR: Gradually lowers the learning rate following a cosine curve to explore sharp minima early and settle into wide, generalizable minima.`,
    },
  ];

  const categories = ['All', 'Architecture', 'Computer Vision', 'Deep Learning', 'Engineering', 'Evaluation', 'Optimization'];

  const filtered = questions.filter((item) => {
    const matchesCat = selectedCategory === 'All' || item.category === selectedCategory;
    const matchesSearch =
      item.q.toLowerCase().includes(search.toLowerCase()) ||
      item.answer.toLowerCase().includes(search.toLowerCase());
    return matchesCat && matchesSearch;
  });

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="bg-white/[0.03] p-5 rounded-2xl border border-white/5 space-y-3">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-bold uppercase tracking-wider">
          <BookCheck className="w-4 h-4" /> Comprehensive Academic Defense & Viva Q&A Guide
        </div>
        <h2 className="text-xl font-bold text-white">
          Theoretical Justifications, Architectural Trade-offs & Deep Learning Insights
        </h2>
        <p className="text-xs text-gray-400 leading-relaxed max-w-3xl">
          Curated answers to the most rigorous questions professors and examiners ask during project viva defenses,
          covering model efficiency, mathematical complexity, and computer vision theory.
        </p>

        {/* Filter bar */}
        <div className="flex flex-col md:flex-row gap-3 pt-2">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-gray-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search concepts, mathematical formulas, or keywords..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2 rounded-xl bg-white/5 border border-white/10 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500 transition"
            />
          </div>

          <div className="flex flex-wrap gap-1.5">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                  selectedCategory === cat
                    ? 'bg-indigo-600 text-white'
                    : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Accordion Questions */}
      <div className="space-y-3">
        {filtered.map((item, idx) => {
          const isOpen = openIndex === idx;
          return (
            <div
              key={idx}
              className={`glass-card transition duration-200 overflow-hidden border ${
                isOpen ? 'border-indigo-500/40 bg-white/[0.05]' : 'border-white/5'
              }`}
            >
              <button
                onClick={() => setOpenIndex(isOpen ? -1 : idx)}
                className="w-full p-4 text-left flex items-start justify-between gap-4"
              >
                <div className="flex items-start gap-3">
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 mt-0.5">
                    Q{idx + 1}
                  </span>
                  <div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-purple-400">
                      {item.category}
                    </span>
                    <h3 className="text-sm font-bold text-white mt-0.5">{item.q}</h3>
                  </div>
                </div>

                <div className="p-1 rounded-lg bg-white/5 text-gray-400">
                  {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </button>

              {isOpen && (
                <div className="px-5 pb-5 pt-1 text-xs text-gray-300 leading-relaxed border-t border-white/5 whitespace-pre-line font-normal space-y-2">
                  <div className="bg-black/30 p-4 rounded-xl border border-white/5 text-gray-200">
                    {item.answer}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
