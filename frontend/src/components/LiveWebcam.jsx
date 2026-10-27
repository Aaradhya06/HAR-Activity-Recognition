import React, { useRef, useState, useEffect } from 'react';
import { Camera, CameraOff, Sparkles, RefreshCw, Zap, Gauge, UserCheck } from 'lucide-react';
import { predictFrameSequence } from '../services/api';

export default function LiveWebcam() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [usePose, setUsePose] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);

  const bufferRef = useRef([]);
  const frameIntervalRef = useRef(null);
  const lastTimeRef = useRef(performance.now());

  // Start webcam
  const startCamera = async () => {
    try {
      setErrorMsg(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setStreaming(true);
      }
    } catch (err) {
      setErrorMsg(`Webcam access denied or unavailable: ${err.message}`);
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setStreaming(false);
    clearInterval(frameIntervalRef.current);
  };

  // Capture loop
  useEffect(() => {
    if (!streaming) return;

    const captureInterval = 120; // ms per frame (~8 fps sampling)
    frameIntervalRef.current = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (video.videoWidth === 0 || video.videoHeight === 0) return;

      canvas.width = 224;
      canvas.height = 224;
      ctx.drawImage(video, 0, 0, 224, 224);

      // Convert canvas to blob
      canvas.toBlob(async (blob) => {
        if (!blob) return;
        bufferRef.current.push(new File([blob], `frame_${Date.now()}.jpg`, { type: 'image/jpeg' }));

        // Keep rolling buffer of 16 frames
        if (bufferRef.current.length > 16) {
          bufferRef.current.shift();
        }

        // Run prediction when buffer is full (16 frames) and not already analyzing
        if (bufferRef.current.length === 16 && !analyzing) {
          setAnalyzing(true);
          const t0 = performance.now();
          try {
            const framesToSend = [...bufferRef.current];
            const res = await predictFrameSequence(framesToSend, usePose);
            const t1 = performance.now();
            setLatency(Math.round(t1 - t0));
            setFps(Math.round(1000 / (t1 - t0)));
            setPrediction(res);
          } catch (e) {
            console.error('Frame prediction error:', e);
          } finally {
            setAnalyzing(false);
          }
        }
      }, 'image/jpeg', 0.85);
    }, captureInterval);

    return () => clearInterval(frameIntervalRef.current);
  }, [streaming, analyzing, usePose]);

  return (
    <div className="space-y-6">
      {/* Title & Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white/[0.03] p-5 rounded-2xl border border-white/5">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Camera className="w-5 h-5 text-indigo-400" /> Real-Time Webcam Stream Analysis
          </h2>
          <p className="text-xs text-gray-400 mt-1">
            Captures dynamic 16-frame sliding video clips with real-time pose extraction and temporal attention.
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

          {!streaming ? (
            <button
              onClick={startCamera}
              className="btn-primary flex items-center gap-2 text-xs font-semibold"
            >
              <Camera className="w-4 h-4" /> Start Webcam
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="bg-rose-500/20 text-rose-300 border border-rose-500/30 px-4 py-2 rounded-xl text-xs font-semibold hover:bg-rose-500/30 transition flex items-center gap-2"
            >
              <CameraOff className="w-4 h-4" /> Stop Camera
            </button>
          )}
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
          {errorMsg}
        </div>
      )}

      {/* Main Stream & Predictions Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Stream Canvas */}
        <div className="lg:col-span-7 glass-card p-5 space-y-4">
          <div className="relative aspect-video bg-black/60 rounded-xl overflow-hidden border border-white/10 flex items-center justify-center">
            <video
              ref={videoRef}
              playsInline
              muted
              className={`w-full h-full object-cover ${streaming ? 'block' : 'hidden'}`}
            />
            <canvas ref={canvasRef} className="hidden" />

            {!streaming && (
              <div className="text-center p-8 space-y-3">
                <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto text-indigo-400">
                  <Camera className="w-7 h-7" />
                </div>
                <p className="text-sm font-medium text-gray-300">Camera feed is inactive</p>
                <p className="text-xs text-gray-500 max-w-sm">
                  Click 'Start Webcam' above to begin capturing video frames for real-time action classification.
                </p>
              </div>
            )}

            {streaming && (
              <div className="absolute top-3 left-3 flex items-center gap-2">
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-[10px] text-white">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                  <span>LIVE</span>
                </div>
                {analyzing && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-500/60 backdrop-blur-md text-[10px] text-white">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    <span>Processing Window</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Telemetry Bar */}
          <div className="grid grid-cols-3 gap-3">
            <div className="metric-card flex items-center gap-3">
              <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                <Gauge className="w-4 h-4" />
              </div>
              <div>
                <p className="text-[10px] uppercase font-semibold text-gray-400">Inference Latency</p>
                <p className="text-sm font-bold text-white font-mono">{latency > 0 ? `${latency} ms` : '--'}</p>
              </div>
            </div>

            <div className="metric-card flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
                <Zap className="w-4 h-4" />
              </div>
              <div>
                <p className="text-[10px] uppercase font-semibold text-gray-400">Estimated FPS</p>
                <p className="text-sm font-bold text-white font-mono">{fps > 0 ? `${fps} FPS` : '--'}</p>
              </div>
            </div>

            <div className="metric-card flex items-center gap-3">
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <p className="text-[10px] uppercase font-semibold text-gray-400">Sliding Window</p>
                <p className="text-sm font-bold text-white font-mono">16 Frames (T=16)</p>
              </div>
            </div>
          </div>
        </div>

        {/* Prediction Results Card */}
        <div className="lg:col-span-5 glass-card p-6 flex flex-col justify-between space-y-6">
          <div>
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-gray-300">
                Live Activity Classification
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 border border-white/10 text-gray-400">
                EMA-Smoothed
              </span>
            </div>

            {prediction ? (
              <div className="mt-6 space-y-5">
                {/* Main Label */}
                <div className="p-4 rounded-xl bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-transparent border border-indigo-500/20">
                  <p className="text-xs uppercase font-semibold text-indigo-400 tracking-wider">Top Detected Action</p>
                  <p className="text-2xl font-black text-white mt-1 tracking-tight">
                    {prediction.activity}
                  </p>
                  <div className="mt-2 flex items-center gap-2">
                    <div className="flex-1 bg-white/10 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-300"
                        style={{ width: `${Math.round(prediction.confidence * 100)}%` }}
                      />
                    </div>
                    <span className="text-xs font-bold font-mono text-emerald-400">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Top-5 breakdown */}
                <div className="space-y-3 pt-2">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Probability Distribution (Top-5)
                  </p>
                  {prediction.top_predictions?.map((item, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs font-medium">
                        <span className="text-gray-300">
                          {idx + 1}. {item.class_name || item.class}
                        </span>
                        <span className="font-mono text-gray-400">
                          {((item.probability ?? item.confidence ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${Math.round((item.probability ?? item.confidence ?? 0) * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-gray-500 space-y-2">
                <p className="text-sm">Awaiting video window frames...</p>
                <p className="text-xs">Start the camera to begin streaming real-time action predictions.</p>
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-white/5 text-[11px] text-gray-500 flex items-center justify-between">
            <span>Model: MobileNetV3 + Transformer</span>
            <span>Auxiliary: MediaPipe 33-pt Pose</span>
          </div>
        </div>
      </div>
    </div>
  );
}
