'use client';

import { useEffect, useRef, useState } from 'react';
import { RPPGAcquisition } from '@/lib/camera-utils'; 
import { SignalStorage, RecordingSession, preprocessPPG, extractFeatures, performMathEstimation } from '@/lib/signal-processing';
import SignalVisualizer from '@/components/visualization/signal-visualizer';
import { Pause, Play, Save, Zap, ZapOff, Timer, User, X } from 'lucide-react';

export default function RecordingTab() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // -- APP STATE --
  const [isRecording, setIsRecording] = useState(false);
  const [visRaw, setVisRaw] = useState<number[]>([]);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isTorchOn, setIsTorchOn] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Ready");
  
  // -- USER FORM STATE --
  const [showUserForm, setShowUserForm] = useState(false);
  const [userDetails, setUserDetails] = useState({
      name: '',
      age: '',
      height: '',
      weight: ''
  });

  // -- RESULTS STATE --
  const [showResults, setShowResults] = useState(false);
  const [estimation, setEstimation] = useState<{sbp:number, dbp:number, glucose:number} | null>(null);
  const [pendingSession, setPendingSession] = useState<RecordingSession | null>(null);

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

  // 1. CLICK START -> OPEN FORM
  const handleStartClick = () => {
    if (!videoRef.current || videoRef.current.readyState < 2) {
        alert("Wait for camera to load...");
        return;
    }
    // Reset inputs
    setUserDetails({ name: '', age: '', height: '', weight: '' });
    setShowUserForm(true);
  };

  // 2. FORM CONFIRM -> START RECORDING
  const startRecording = () => {
    if(!userDetails.name || !userDetails.age || !userDetails.height || !userDetails.weight) {
        alert("Please fill all details");
        return;
    }

    setShowUserForm(false);
    recordedSamplesRef.current = [];
    setRecordingTime(0);
    setVisRaw([]);
    setIsRecording(true);
    setStatusMsg("Recording...");
    
    const startTime = Date.now();

    recordingIntervalRef.current = setInterval(() => {
      if (!rpPgRef.current || !videoRef.current) return;
      const val = rpPgRef.current.extractSignal(videoRef.current);
      const now = Date.now();
      recordedSamplesRef.current.push({ timestamp: now, value: val });
      setVisRaw(prev => {
          const next = [...prev, val];
          if (next.length > 300) return next.slice(next.length - 300);
          return next;
      });
      setRecordingTime((now - startTime) / 1000);
    }, 1000 / 30); 
  };

  // 3. STOP RECORDING -> PROCESS & SHOW RESULTS
  const stopAndAnalyze = async () => {
    // Stop loop
    setIsRecording(false);
    if(recordingIntervalRef.current) clearInterval(recordingIntervalRef.current);
    setStatusMsg("Analyzing...");

    if (recordingTime < 10) {
        alert("Recording too short! Need at least 10 seconds.");
        setStatusMsg("Ready");
        return;
    }

    try {
        // Prepare Data
        const raw = recordedSamplesRef.current.map(s => s.value);
        const filtered = preprocessPPG(raw);
        const features = extractFeatures(filtered);
        
        // Mathematical Estimation
        const age = parseInt(userDetails.age) || 30;
        const height = parseInt(userDetails.height) || 170;
        const weight = parseInt(userDetails.weight) || 70;

        const results = performMathEstimation(features, age, height, weight);
        setEstimation(results);

        // Prepare Session Object (But don't save yet)
        const storage = new SignalStorage();
        const newId = await storage.generateNextId();
        
        const session: RecordingSession = {
            id: newId,
            createdAt: new Date(),
            startTime: recordedSamplesRef.current[0]?.timestamp || Date.now(),
            endTime: Date.now(),
            samplingRate: 30,
            rawSignal: recordedSamplesRef.current,
            patientName: userDetails.name,
            age: age,
            height: height,
            weight: weight,
            features: features,
            sbp: results.sbp,
            dbp: results.dbp,
            glucose: results.glucose,
            quality: 'Good' // Simplified
        };

        setPendingSession(session);
        setShowResults(true);

    } catch (e) {
        console.error(e);
        alert("Analysis failed. Try again with a clearer signal.");
        setStatusMsg("Error");
    }
  };

  // 4. SAVE
  const handleSave = async () => {
    if(pendingSession) {
        const storage = new SignalStorage();
        await storage.saveSession(pendingSession);
        alert(`Session ${pendingSession.id} Saved!`);
    }
    resetFlow();
  };

  // 5. DISCARD
  const handleDiscard = () => {
    if(confirm("Discard this recording?")) {
        resetFlow();
    }
  };

  const resetFlow = () => {
    setShowResults(false);
    setPendingSession(null);
    setVisRaw([]);
    setRecordingTime(0);
    setStatusMsg("Ready");
    recordedSamplesRef.current = [];
  };

  return (
    <div className="space-y-4 p-4 pb-24 relative">
      
      {/* --- MODAL: USER FORM --- */}
      {showUserForm && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-card w-full max-w-sm rounded-xl border shadow-2xl p-6 space-y-4 animate-in fade-in zoom-in-95">
                <div className="flex justify-between items-center">
                    <h2 className="text-xl font-bold flex items-center gap-2"><User className="w-5 h-5"/> Patient Details</h2>
                    <button onClick={()=>setShowUserForm(false)}><X className="w-5 h-5"/></button>
                </div>
                <div className="space-y-3">
                    <div>
                        <label className="text-sm font-medium">Name</label>
                        <input type="text" className="w-full bg-background border rounded p-2" 
                            value={userDetails.name} onChange={e => setUserDetails({...userDetails, name: e.target.value})} placeholder="Your Name"/>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                        <div>
                            <label className="text-sm font-medium">Age</label>
                            <input type="number" className="w-full bg-background border rounded p-2" 
                                value={userDetails.age} onChange={e => setUserDetails({...userDetails, age: e.target.value})} placeholder="30"/>
                        </div>
                        <div>
                            <label className="text-sm font-medium">Ht (cm)</label>
                            <input type="number" className="w-full bg-background border rounded p-2" 
                                value={userDetails.height} onChange={e => setUserDetails({...userDetails, height: e.target.value})} placeholder="175"/>
                        </div>
                        <div>
                            <label className="text-sm font-medium">Wt (kg)</label>
                            <input type="number" className="w-full bg-background border rounded p-2" 
                                value={userDetails.weight} onChange={e => setUserDetails({...userDetails, weight: e.target.value})} placeholder="70"/>
                        </div>
                    </div>
                </div>
                <button onClick={startRecording} className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-bold text-lg shadow hover:opacity-90">
                    Confirm & Start
                </button>
            </div>
        </div>
      )}

      {/* --- MODAL: RESULTS --- */}
      {showResults && estimation && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
            <div className="bg-card w-full max-w-sm rounded-xl border shadow-2xl overflow-hidden animate-in slide-in-from-bottom-10">
                <div className="bg-primary/10 p-4 text-center border-b">
                    <h2 className="text-2xl font-bold text-primary">Estimation Results</h2>
                    <p className="text-xs text-muted-foreground">Session ID: {pendingSession?.id}</p>
                </div>
                <div className="p-6 space-y-6">
                    <div className="grid grid-cols-3 gap-4 text-center">
                        <div className="space-y-1">
                            <p className="text-xs text-muted-foreground uppercase">Glucose</p>
                            <p className="text-2xl font-mono font-bold">{estimation.glucose}</p>
                            <p className="text-[10px] text-muted-foreground">mg/dL</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-xs text-muted-foreground uppercase">Systolic</p>
                            <p className="text-2xl font-mono font-bold">{estimation.sbp}</p>
                            <p className="text-xs text-muted-foreground">mmHg</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-xs text-muted-foreground uppercase">Diastolic</p>
                            <p className="text-2xl font-mono font-bold">{estimation.dbp}</p>
                            <p className="text-xs text-muted-foreground">mmHg</p>
                        </div>
                    </div>
                    
                    <div className="bg-yellow-500/10 border border-yellow-500/20 p-3 rounded text-xs text-yellow-600 dark:text-yellow-400">
                        ⚠️ These are mathematical estimations based on PPG waveforms. NOT a medical diagnosis.
                    </div>
                </div>
                <div className="grid grid-cols-2 gap-1 p-2 bg-muted/50">
                    <button onClick={handleDiscard} className="py-3 rounded bg-background border shadow-sm hover:bg-destructive/10 hover:text-destructive transition-colors font-medium">
                        Discard
                    </button>
                    <button onClick={handleSave} className="py-3 rounded bg-green-600 text-white shadow-sm hover:bg-green-700 transition-colors font-bold flex justify-center items-center gap-2">
                        <Save className="w-4 h-4"/> Save
                    </button>
                </div>
            </div>
        </div>
      )}


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
        {!isRecording ? (
             <button 
             onClick={handleStartClick} 
             className="h-16 w-16 flex items-center justify-center rounded-full shadow-lg bg-green-500 hover:bg-green-600 text-white transition-all hover:scale-105">
                <Play className="w-8 h-8 ml-1" />
            </button>
        ) : (
            <button 
            onClick={stopAndAnalyze} 
            className="h-16 w-16 flex items-center justify-center rounded-full shadow-lg bg-red-500 hover:bg-red-600 text-white animate-pulse">
                <Pause className="w-8 h-8" />
            </button>
        )}
      </div>
    </div>
  );
}