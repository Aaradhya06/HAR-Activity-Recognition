import React, { useState } from 'react';
import { Layers, Cpu, Eye, Zap, Shield, Sparkles, Check, ArrowRight, BookOpen } from 'lucide-react';

export default function ArchitectureExplorer() {
  const [selectedBlock, setSelectedBlock] = useState('transformer');

  const architectureBlocks = [
    {
      id: 'backbone',
      title: '2D CNN Feature Extractor',
      tech: 'MobileNetV3-Small / EfficientNet-B0',
      desc: 'Processes each frame independently using inverted residual blocks and squeeze-and-excitation attention to extract spatial visual representations with sub-10ms latency.',
      specs: ['Input: (B, 16, 3, 224, 224)', 'Native Dim: 576', 'Projected Feature: 256-d', 'Parameters: ~1.5M'],
      advantage: 'Avoids cubic compute cost of 3D convolutions while maintaining high spatial discriminability.',
    },
    {
      id: 'transformer',
      title: 'Temporal Transformer Module',
      tech: 'Multi-Head Self-Attention + Sinusoidal PE',
      desc: 'Captures long-range temporal dependencies across the 16-frame sequence without recurrent vanishing gradients or sequential bottlenecks.',
      specs: ['Heads: 8', 'Transformer Layers: 2', 'FeedForward Dim: 512', 'Attention Pooling: Mean / CLS'],
      advantage: 'Direct pairwise token interactions across frames model non-local temporal dynamics far better than LSTM/GRU.',
    },
    {
      id: 'pose',
      title: 'Auxiliary Pose-Guided Branch',
      tech: 'MediaPipe BlazePose + MLP Encoder',
      desc: 'Extracts 33 high-precision 3D landmarks (99 coordinates) per frame. Encodes skeletal articulation patterns that are invariant to clothing, lighting, and camera backgrounds.',
      specs: ['Landmarks: 33 keypoints', 'Coords: (x, y, visibility)', 'Input: (B, 16, 99)', 'Encoded Dim: 64-d'],
      advantage: 'Prevents overfitting to scene backgrounds and preserves discriminative activity signals even in severe occlusion.',
    },
    {
      id: 'fusion',
      title: 'Multimodal Fusion & Classification',
      tech: 'Cross-Modal Concat & MLP Head',
      desc: 'Fuses spatial-temporal visual embeddings (256-d) with skeletal pose dynamics (64-d) into a unified 320-d representation before final classification.',
      specs: ['Visual: 256-d', 'Pose: 64-d', 'Fused Vector: 320-d -> 256-d', 'Dropout: 0.3'],
      advantage: 'Joint feature representation ensures the network can smoothly fall back to purely visual signals if pose is undetected.',
    },
  ];

  const literatureTable = [
    {
      method: '2D CNN + Pooling',
      examples: 'ResNet-50, VGG-16',
      temporal: 'Average pooling across frames',
      params: '25.6 M',
      latency: 'Fast (~15 ms)',
      pros: 'Simple, fast, low computational complexity',
      cons: 'Completely ignores temporal ordering and action speed dynamics',
    },
    {
      method: '3D CNNs',
      examples: 'C3D, I3D, ResNet3D',
      temporal: '3D Spatio-Temporal Convolutions',
      params: '33.0 - 45.0 M',
      latency: 'Very High (~90-180 ms)',
      pros: 'Joint spatial and temporal representation learning',
      cons: 'Huge memory footprint, high FLOPs, unsuitable for edge devices without discrete GPU',
    },
    {
      method: 'Two-Stream Networks',
      examples: 'Simonyan & Zisserman (2014)',
      temporal: 'Dense Optical Flow Stream + RGB Stream',
      params: '48.0 M',
      latency: 'Extremely Slow (>300 ms)',
      pros: 'Strong explicit motion representation',
      cons: 'Computing TV-L1 optical flow in real-time is computationally prohibitive on CPUs/laptops',
    },
    {
      method: 'CNN + RNN / LSTM',
      examples: 'LRCN (Donahue et al.)',
      temporal: 'Recurrent Memory Cell (LSTM / GRU)',
      params: '12.0 - 28.0 M',
      latency: 'Moderate (~40-60 ms)',
      pros: 'Explicit sequential modeling with modest compute',
      cons: 'Sequential dependency prevents parallel execution; suffers from vanishing gradients over long clips',
    },
    {
      method: 'SlowFast Networks',
      examples: 'Feichtenhofer et al. (ICCV 2019)',
      temporal: 'Dual pathway (Low frame-rate + High frame-rate)',
      params: '34.0 M',
      latency: 'High (~75 ms)',
      pros: 'Decoupled spatial semantics and rapid motion dynamics',
      cons: 'Heavy 3D convolutions require substantial hardware budget',
    },
    {
      method: 'Graph Convolution (GCN)',
      examples: 'ST-GCN (Yan et al.)',
      temporal: 'Spatio-Temporal Graph Convolutions',
      params: '3.1 M',
      latency: 'Low (~20 ms)',
      pros: 'Excellent skeleton topology modeling, compact model size',
      cons: 'Fails when pose estimation fails or when interacting with external objects (e.g. makeup, dog)',
    },
    {
      method: 'Pure Video Transformers',
      examples: 'TimeSformer, ViViT',
      temporal: 'Divided space-time self-attention',
      params: '121.0 M',
      latency: 'Very High (~120-250 ms)',
      pros: 'State-of-the-art accuracy on large datasets (Kinetics-400)',
      cons: 'Prohibitive memory footprint, quadratic complexity O(T²), requires massive pre-training datasets',
    },
    {
      method: 'Lightweight Hybrid + Pose (Proposed)',
      examples: 'MobileNetV3 + Transformer + BlazePose',
      temporal: 'Multi-Head Self-Attention + Pose Guidance',
      params: '2.28 M',
      latency: 'Real-time (<10 ms on GPU, ~35 ms CPU)',
      pros: 'High accuracy, lightweight (2.28M params), background-invariant, real-time edge execution',
      cons: 'Slight dependency on pose estimator clarity for subtle finger-level gestures',
    },
  ];

  return (
    <div className="space-y-8">
      {/* Overview Card */}
      <div className="bg-white/[0.03] p-6 rounded-2xl border border-white/5 space-y-3">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-bold uppercase tracking-wider">
          <BookOpen className="w-4 h-4" /> Comprehensive Technical Architecture & Literature Review
        </div>
        <h2 className="text-2xl font-black text-white tracking-tight">
          Lightweight Hybrid 2D CNN + Temporal Attention + Pose Guidance
        </h2>
        <p className="text-sm text-gray-300 leading-relaxed max-w-4xl">
          Based on the comparative analysis in human action recognition literature, purely 3D convolutions (e.g., C3D, I3D)
          and pure video transformers (e.g., ViViT) suffer from quadratic complexity and extreme parameter counts. Our proposed architecture
          achieves the optimal Pareto frontier between accuracy and latency by decoupling spatial representation learning (MobileNetV3),
          global sequence modeling (Temporal Self-Attention), and skeletal topology (MediaPipe BlazePose).
        </p>
      </div>

      {/* Interactive Architecture Flow Diagram */}
      <div className="glass-card p-6 space-y-6">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" /> End-to-End Multimodal Pipeline Flow
        </h3>

        {/* Visual Pipeline Blocks */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {architectureBlocks.map((block) => (
            <div
              key={block.id}
              onClick={() => setSelectedBlock(block.id)}
              className={`p-4 rounded-xl border cursor-pointer transition-all duration-300 flex flex-col justify-between ${
                selectedBlock === block.id
                  ? 'bg-indigo-600/15 border-indigo-500 shadow-lg shadow-indigo-500/20'
                  : 'bg-white/[0.02] border-white/5 hover:bg-white/[0.05] hover:border-white/10'
              }`}
            >
              <div>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-white/10 text-gray-300">
                  {block.tech.split(' ')[0]}
                </span>
                <h4 className="text-sm font-bold text-white mt-2">{block.title}</h4>
                <p className="text-xs text-gray-400 mt-1 line-clamp-2">{block.tech}</p>
              </div>

              <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between text-xs text-indigo-400 font-semibold">
                <span>View Details</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </div>
            </div>
          ))}
        </div>

        {/* Detail Inspector for Selected Block */}
        {(() => {
          const block = architectureBlocks.find((b) => b.id === selectedBlock) || architectureBlocks[0];
          return (
            <div className="p-5 rounded-2xl bg-gradient-to-r from-indigo-500/10 via-purple-500/5 to-transparent border border-indigo-500/20 space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-white/10 pb-3">
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">
                    Selected Subsystem
                  </span>
                  <h4 className="text-lg font-bold text-white">{block.title} ({block.tech})</h4>
                </div>
                <div className="text-xs font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full">
                  Design Advantage: {block.advantage}
                </div>
              </div>

              <p className="text-sm text-gray-300">{block.desc}</p>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
                {block.specs.map((spec, i) => (
                  <div key={i} className="p-2.5 rounded-lg bg-white/5 border border-white/5 text-xs font-mono text-gray-300">
                    {spec}
                  </div>
                ))}
              </div>
            </div>
          );
        })()}
      </div>

      {/* Literature Review Comparison Matrix */}
      <div className="glass-card p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white">
              Comparative Analysis: Video HAR Paradigms
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              Synthesis of architectures analyzed in the literature review report.
            </p>
          </div>
          <span className="text-xs font-mono text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2.5 py-1 rounded-lg">
            8 Architectures Evaluated
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-[10px]">
                <th className="py-3 px-3">Architecture Paradigm</th>
                <th className="py-3 px-3">Notable Examples</th>
                <th className="py-3 px-3">Temporal Modeling Strategy</th>
                <th className="py-3 px-3">Params</th>
                <th className="py-3 px-3">Latency</th>
                <th className="py-3 px-3">Key Advantage</th>
                <th className="py-3 px-3">Primary Limitation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {literatureTable.map((row, idx) => {
                const isProposed = row.method.includes('Proposed');
                return (
                  <tr
                    key={idx}
                    className={`transition ${
                      isProposed
                        ? 'bg-gradient-to-r from-indigo-500/15 via-purple-500/10 to-emerald-500/10 font-medium'
                        : 'hover:bg-white/[0.02]'
                    }`}
                  >
                    <td className="py-3.5 px-3 font-bold text-white flex items-center gap-1.5">
                      {isProposed && <Sparkles className="w-3.5 h-3.5 text-emerald-400" />}
                      {row.method}
                    </td>
                    <td className="py-3.5 px-3 font-mono text-gray-300">{row.examples}</td>
                    <td className="py-3.5 px-3 text-gray-400">{row.temporal}</td>
                    <td className="py-3.5 px-3 font-mono text-gray-200">{row.params}</td>
                    <td className="py-3.5 px-3 font-mono text-indigo-300">{row.latency}</td>
                    <td className="py-3.5 px-3 text-emerald-400/90">{row.pros}</td>
                    <td className="py-3.5 px-3 text-rose-300/80">{row.cons}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
