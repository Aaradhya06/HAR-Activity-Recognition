import React, { useEffect, useState } from 'react';
import { Activity, Cpu, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import { getHealth, getModelInfo } from '../services/api';

export default function Navbar({ activeTab, setActiveTab }) {
  const [health, setHealth] = useState({ status: 'connecting', device: 'unknown' });
  const [modelInfo, setModelInfo] = useState(null);

  const checkStatus = async () => {
    try {
      const h = await getHealth();
      setHealth(h);
      const m = await getModelInfo();
      setModelInfo(m);
    } catch {
      setHealth({ status: 'offline', device: 'disconnected' });
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 8000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'webcam', label: 'Live Webcam' },
    { id: 'video', label: 'Video Analyzer' },
    { id: 'architecture', label: 'Architecture & Theory' },
    { id: 'evaluation', label: 'Evaluation & Metrics' },
    { id: 'viva', label: 'Viva Defense Q&A' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#0a0a1a]/80 backdrop-blur-xl border-b border-white/10 px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-emerald-400 p-[2px] shadow-lg shadow-indigo-500/30">
            <div className="w-full h-full bg-[#0a0a1a] rounded-[10px] flex items-center justify-center">
              <Activity className="w-5 h-5 text-indigo-400 animate-pulse" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-extrabold text-base tracking-tight text-white">
                HAR <span className="gradient-text">Deep Vision</span>
              </h1>
              <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Hybrid CNN+Attn+Pose
              </span>
            </div>
            <p className="text-[11px] text-gray-400 font-medium">
              Human Activity Recognition via Deep Learning & Computer Vision
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1 bg-white/[0.03] p-1 rounded-xl border border-white/5">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                activeTab === item.id
                  ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-md shadow-indigo-500/25'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {/* Hardware Status Badge */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/10 text-xs">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-mono text-[11px] text-gray-300 capitalize">
              {health.device.includes('cuda') ? 'NVIDIA RTX 2050 (CUDA)' : health.device}
            </span>
          </div>

          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border ${
              health.status === 'ok'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
            }`}
          >
            {health.status === 'ok' ? (
              <>
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                <span>Backend Online</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3 h-3 text-rose-400" />
                <span>Offline</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
