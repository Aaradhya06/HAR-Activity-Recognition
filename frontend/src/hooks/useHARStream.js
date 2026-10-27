/**
 * useHARStream — Custom React hook for live HAR webcam streaming.
 * Manages frame buffer, inference loop, and EMA-smoothed predictions.
 */

import { useRef, useState, useCallback } from 'react';
import { predictFrameSequence } from '../services/api';

const FRAME_COUNT = 16;
const CAPTURE_INTERVAL_MS = 120; // ~8 fps capture rate

export function useHARStream({ usePose = true } = {}) {
  const [streaming, setStreaming]     = useState(false);
  const [prediction, setPrediction]   = useState(null);
  const [analyzing, setAnalyzing]     = useState(false);
  const [latencyMs, setLatencyMs]     = useState(0);
  const [fps, setFps]                 = useState(0);
  const [error, setError]             = useState(null);

  const videoRef      = useRef(null);
  const canvasRef     = useRef(null);
  const bufferRef     = useRef([]);
  const intervalRef   = useRef(null);
  const analyzingRef  = useRef(false);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setStreaming(true);

        intervalRef.current = setInterval(() => {
          const video  = videoRef.current;
          const canvas = canvasRef.current;
          if (!video || !canvas || video.videoWidth === 0) return;
          const ctx = canvas.getContext('2d');
          canvas.width  = 224;
          canvas.height = 224;
          ctx.drawImage(video, 0, 0, 224, 224);

          canvas.toBlob(async (blob) => {
            if (!blob) return;
            bufferRef.current.push(new File([blob], `frame_${Date.now()}.jpg`, { type: 'image/jpeg' }));
            if (bufferRef.current.length > FRAME_COUNT) bufferRef.current.shift();

            if (bufferRef.current.length === FRAME_COUNT && !analyzingRef.current) {
              analyzingRef.current = true;
              setAnalyzing(true);
              const t0 = performance.now();
              try {
                const res = await predictFrameSequence([...bufferRef.current], usePose);
                const dt  = performance.now() - t0;
                setLatencyMs(Math.round(dt));
                setFps(Math.round(1000 / dt));
                setPrediction(res);
              } catch (e) {
                console.error('HAR stream error:', e);
              } finally {
                analyzingRef.current = false;
                setAnalyzing(false);
              }
            }
          }, 'image/jpeg', 0.85);
        }, CAPTURE_INTERVAL_MS);
      }
    } catch (err) {
      setError(`Camera unavailable: ${err.message}`);
    }
  }, [usePose]);

  const stopCamera = useCallback(() => {
    clearInterval(intervalRef.current);
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    bufferRef.current = [];
    analyzingRef.current = false;
    setStreaming(false);
    setAnalyzing(false);
    setPrediction(null);
  }, []);

  return {
    videoRef,
    canvasRef,
    streaming,
    prediction,
    analyzing,
    latencyMs,
    fps,
    error,
    startCamera,
    stopCamera,
  };
}

export default useHARStream;
