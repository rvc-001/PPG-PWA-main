'use client';

import { useEffect, useRef, useState } from 'react';
import { RPPGAcquisition } from '@/lib/camera-utils'; 
import { SignalStorage, RecordingSession } from '@/lib/signal-processing';
import SignalVisualizer from '@/components/visualization/signal-visualizer';
import { Pause, Play, Save, Zap, ZapOff, Timer } from 'lucide-react';

export default function RecordingTab() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [visRaw, setVisRaw] = useState<number[]>([]);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isTorchOn, setIsTorchOn] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Ready");
  
  const recordedSamplesRef = useRef<{ timestamp: number; value: number }[]>([]);
  const rpPgRef = useRef<RPPGAcquisition | null>(null);

  useEffect(() => {
    initCamera();
    return () => stopCamera();
  }, []);

  const initCamera = async () => {
    try {
      rpPgRef.current = new RPPGAcquisition(30);
      const stream = await rpPgRef.current.requestCameraPermission();
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(e => console.error("Play error", e));
        
        // Wait a moment for camera to warm up then toggle torch
        setTimeout(() => {
            if(rpPgRef.current) {
                rpPgRef.current.toggleTorch(true).then(() => setIsTorchOn(true)).catch(() => {});
            }
        }, 1000);
      }
    } catch (e) {
      console.error("Camera init failed", e);
      setStatusMsg("Camera Error");
    }
  };

  const stopCamera = () => {
    if (recordingIntervalRef.current) clearInterval(recordingIntervalRef.current);
    rpPgRef.current?.stop();
  };

  const toggleRecording = () => {
    if (isRecording) {
      // STOP
      setIsRecording(false);
      if(recordingIntervalRef.current) clearInterval(recordingIntervalRef.current);
      setStatusMsg("Recording Stopped");
    } else {
      // START
      if (!videoRef.current || videoRef.current.readyState < 2) {
          alert("Wait for camera to load...");
          return;
      }

      recordedSamplesRef.current = [];
      setRecordingTime(0);
      setVisRaw([]);
      setIsRecording(true);
      setStatusMsg("Recording...");
      
      const startTime = Date.now();

      recordingIntervalRef.current = setInterval(() => {
        if (!rpPgRef.current || !videoRef.current) return;

        // Extract value
        const val = rpPgRef.current.extractSignal(videoRef.current);
        const now = Date.now();
        
        // Only push non-zero values if possible (or accept noise)
        recordedSamplesRef.current.push({ timestamp: now, value: val });
        
        // Update Visualizer
        setVisRaw(prev => {
            const next = [...prev, val];
            if (next.length > 300) return next.slice(next.length - 300);
            return next;
        });

        setRecordingTime((now - startTime) / 1000);

      }, 1000 / 30); 
    }
  };

  const saveRecording = async () => {
    // Notebook uses TRIM_SEC = 3 (Total 6s removed).
    // We need at least 10s data to have meaningful features
    if (recordingTime < 10) {
        alert("Recording too short! Please record at least 12 seconds for accurate analysis.");
        return;
    }

    const storage = new SignalStorage();
    const session: RecordingSession = {
      id: Date.now().toString(),
      createdAt: new Date(),
      startTime: recordedSamplesRef.current[0]?.timestamp || Date.now(),
      endTime: Date.now(),
      samplingRate: 30,
      rawSignal: recordedSamplesRef.current, 
      features: [],
      patientName: "New Session"
    };
    
    await storage.saveSession(session);
    alert("Recording Saved! Go to History/Model tab to analyze.");
    
    // Reset
    setVisRaw([]); 
    setRecordingTime(0);
    recordedSamplesRef.current = [];
    setStatusMsg("Saved");
  };

  return (
    <div className="space-y-4 p-4 pb-24">
      {/* Video Preview */}
      <div className="relative h-48 bg-black rounded-lg overflow-hidden shadow-md">
        <video ref={videoRef} muted playsInline className="w-full h-full object-cover" />
        <div className="absolute top-2 left-2 flex gap-2">
            <button 
            onClick={() => rpPgRef.current?.toggleTorch(!isTorchOn).then(() => setIsTorchOn(!isTorchOn))} 
            className="p-2 bg-black/40 backdrop-blur rounded-full">
            {isTorchOn ? <Zap className="text-yellow-400 w-5 h-5" /> : <ZapOff className="text-white w-5 h-5" />}
            </button>
            <div className="px-3 py-1 bg-black/40 backdrop-blur rounded-full text-white text-xs flex items-center">
                {statusMsg}
            </div>
        </div>
      </div>

      {/* Timer & Visualizer */}
      <div className="bg-card border rounded-lg p-3">
          <div className="text-center font-mono text-3xl font-bold mb-2 flex justify-center items-center gap-2">
            <Timer className="w-6 h-6 text-muted-foreground"/> 
            <span className={recordingTime < 10 && isRecording ? "text-red-500" : "text-primary"}>
                {recordingTime.toFixed(1)}s
            </span>
          </div>
          {recordingTime < 10 && isRecording && <p className="text-center text-xs text-red-500 animate-pulse">Keep recording... (min 10s)</p>}
          
          <div className="h-32 bg-slate-950 rounded border border-slate-800 p-1 mt-2">
            <SignalVisualizer rawSignal={visRaw} filteredSignal={[]} color="#10b981" />
          </div>
      </div>

      {/* Controls */}
      <div className="flex gap-4 justify-center pt-2">
        <button 
            onClick={toggleRecording} 
            className={`h-16 w-16 flex items-center justify-center rounded-full shadow-lg transition-all ${isRecording ? 'bg-red-500 hover:bg-red-600 scale-110' : 'bg-green-500 hover:bg-green-600'} text-white`}>
          {isRecording ? <Pause className="w-8 h-8" /> : <Play className="w-8 h-8 ml-1" />}
        </button>
        
        {!isRecording && recordedSamplesRef.current.length > 0 && (
           <button 
             onClick={saveRecording} 
             className="h-16 w-16 flex items-center justify-center rounded-full bg-blue-500 hover:bg-blue-600 text-white shadow-lg transition-all animate-in fade-in zoom-in">
             <Save className="w-8 h-8" />
           </button>
        )}
      </div>
    </div>
  );
}