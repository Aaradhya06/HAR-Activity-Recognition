import React, { useState, useRef } from 'react';
import { UploadCloud, Play, Film, CheckCircle, RefreshCw, Sparkles, UserCheck, Clock, Gauge } from 'lucide-react';
import { predictVideo } from '../services/api';

export default function VideoAnalyzer() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [usePose, setUsePose] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const sampleVideos = [
    { name: 'JumpingJack', file: 'v_JumpingJack_g01_c01.mp4', label: 'Jumping Jack' },
    { name: 'PushUps', file: 'v_PushUps_g01_c01.mp4', label: 'Push Ups' },
    { name: 'Squats', file: 'v_Squats_g01_c01.mp4', label: 'Squats' },
    { name: 'WalkingWithDog', file: 'v_WalkingWithDog_g01_c01.mp4', label: 'Walking With Dog' },
    { name: 'Biking', file: 'v_Biking_g01_c01.mp4', label: 'Biking' },
  ];

  const handleFileChange = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const loadSample = async (sample) => {
    try {
      setLoading(true);
      setError(null);
      // Fetch sample from public/data or backend
      const res = await fetch(`/api/results/sample-video/${sample.name}/${sample.file}`);
      if (!res.ok) {
        // Fallback: create mock blob from generated files
        throw new Error('Sample not directly downloadable via browser URL');
      }
      const blob = await res.blob();
      const file = new File([blob], sample.file, { type: 'video/mp4' });
      handleFileChange(file);
    } catch {
      setError(`To test "${sample.label}", please select or drag and drop any test video file.`);
    } finally {
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictVideo(selectedFile, usePose);
      setResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to analyze video');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white/[0.03] p-5 rounded-2xl border border-white/5">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Film className="w-5 h-5 text-indigo-400" /> Video File Activity Recognition
          </h2>
          <p className="text-xs text-gray-400 mt-1">
            Upload any standard action clip to extract uniform temporal sequences and compute deep multi-modal classification.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer bg-white/5 px-3 py-2 rounded-xl border border-white/10 hover:bg-white/10 transition">
            <input
              type="checkbox"
              checked={usePose}
              onChange={(e) => setUsePose(e.target.checked)}
              className="rounded border-gray-600 text-indigo-600 focus:ring-indigo-500 bg-transparent"
            />
            <span className="text-xs font-medium text-gray-300 flex items-center gap-1.5">
              <UserCheck className="w-3.5 h-3.5 text-emerald-400" /> Pose-Guided Branch
            </span>
          </label>

          <button
            onClick={runAnalysis}
            disabled={!selectedFile || loading}
            className="btn-primary flex items-center gap-2 text-xs font-semibold disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {loading ? 'Analyzing...' : 'Analyze Action'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
          {error}
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Upload and Video Preview */}
        <div className="lg:col-span-6 space-y-4">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="upload-zone p-8 text-center flex flex-col items-center justify-center cursor-pointer min-h-[220px]"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
            />
            <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-3">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-sm font-semibold text-gray-200">
              {selectedFile ? selectedFile.name : 'Click or drag and drop action video'}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Supports MP4, AVI, MOV, WEBM (Max 100MB)
            </p>
          </div>

          {previewUrl && (
            <div className="glass-card p-4 space-y-2">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Video Scrubber</p>
              <video
                src={previewUrl}
                controls
                className="w-full rounded-lg aspect-video bg-black/50 object-contain"
              />
            </div>
          )}

          {/* Quick Sample Selector */}
          <div className="glass-card p-4 space-y-2">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Quick Test Actions from Generated Dataset:
            </p>
            <div className="flex flex-wrap gap-2 pt-1">
              {sampleVideos.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => loadSample(s)}
                  className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-medium text-gray-300 transition"
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Analysis Results */}
        <div className="lg:col-span-6 glass-card p-6 flex flex-col justify-between space-y-6">
          <div>
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-gray-300">
                Action Classification Results
              </h3>
              {result && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30 text-emerald-300">
                  Processed Successfully
                </span>
              )}
            </div>

            {result ? (
              <div className="mt-6 space-y-6">
                {/* Highlight Winner */}
                <div className="p-5 rounded-2xl bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-emerald-500/10 border border-indigo-500/30">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
                      Predicted Activity
                    </span>
                    <span className="text-xs font-bold font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      {(result.confidence * 100).toFixed(1)}% Confidence
                    </span>
                  </div>
                  <h4 className="text-3xl font-black text-white mt-2 tracking-tight">
                    {result.activity}
                  </h4>

                  {/* Progress bar */}
                  <div className="mt-3 w-full bg-white/10 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 rounded-full"
                      style={{ width: `${Math.round(result.confidence * 100)}%` }}
                    />
                  </div>
                </div>

                {/* Top-5 Predictions */}
                <div className="space-y-3">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Prediction Rankings (Top-5)
                  </p>
                  <div className="space-y-2">
                    {result.top_predictions?.map((item, idx) => (
                      <div key={idx} className="p-2.5 rounded-lg bg-white/[0.02] border border-white/5 space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="font-semibold text-gray-200">
                            #{item.rank || idx + 1} {item.class_name || item.class}
                          </span>
                          <span className="font-mono text-gray-400 font-medium">
                            {((item.probability ?? item.confidence ?? 0) * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="h-full bg-indigo-500 rounded-full transition-all duration-300"
                            style={{ width: `${Math.round((item.probability ?? item.confidence ?? 0) * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Benchmarks */}
                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="metric-card flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                      <Clock className="w-4 h-4" />
                    </div>
                    <div>
                      <p className="text-[10px] uppercase font-semibold text-gray-400">Processing Latency</p>
                      <p className="text-sm font-bold text-white font-mono">{result.inference_time_ms} ms</p>
                    </div>
                  </div>

                  <div className="metric-card flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
                      <Gauge className="w-4 h-4" />
                    </div>
                    <div>
                      <p className="text-[10px] uppercase font-semibold text-gray-400">Throughput</p>
                      <p className="text-sm font-bold text-white font-mono">{result.fps} FPS</p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-20 text-center text-gray-500 space-y-2">
                <p className="text-sm">No analysis performed yet</p>
                <p className="text-xs">Upload a video or choose a quick test action from the left panel.</p>
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-white/5 text-[11px] text-gray-500 flex items-center justify-between">
            <span>Sampling: Uniform 16 Frames (224x224)</span>
            <span>Feature Fusion: Multimodal Concatenation</span>
          </div>
        </div>
      </div>
    </div>
  );
}
