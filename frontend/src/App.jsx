import React, { useState } from 'react';
import Navbar from './components/Navbar';
import LiveWebcam from './components/LiveWebcam';
import VideoAnalyzer from './components/VideoAnalyzer';
import ArchitectureExplorer from './components/ArchitectureExplorer';
import EvaluationDashboard from './components/EvaluationDashboard';
import VivaPrep from './components/VivaPrep';
import { Sparkles, Terminal, Code, Cpu } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('webcam');

  return (
    <div className="min-h-screen bg-[#0a0a1a] text-gray-100 flex flex-col relative selection:bg-indigo-500 selection:text-white">
      {/* Ambient background glows */}
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 relative z-10">
        {activeTab === 'webcam' && <LiveWebcam />}
        {activeTab === 'video' && <VideoAnalyzer />}
        {activeTab === 'architecture' && <ArchitectureExplorer />}
        {activeTab === 'evaluation' && <EvaluationDashboard />}
        {activeTab === 'viva' && <VivaPrep />}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-[#070714] py-6 px-6 relative z-10 text-xs text-gray-500">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-gray-300">Human Activity Recognition</span>
            <span>—</span>
            <span>Deep Learning & Computer Vision Capstone Project</span>
          </div>

          <div className="flex items-center gap-4 text-[11px] font-mono">
            <span className="flex items-center gap-1 text-gray-400">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" /> PyTorch 2.5 + CUDA 12.1
            </span>
            <span className="flex items-center gap-1 text-gray-400">
              <Terminal className="w-3.5 h-3.5 text-emerald-400" /> FastAPI + React/Vite
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
