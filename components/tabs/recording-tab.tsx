'use client';

import { useEffect, useRef, useState } from 'react';
import { RPPGAcquisition } from '@/lib/camera-utils'; 
import { SignalStorage, RecordingSession, preprocessPPG, extractFeatures } from '@/lib/signal-processing';
import SignalVisualizer from '@/components/visualization/signal-visualizer';
import { Pause, Play, Save, Zap, ZapOff, Timer, User, X, Activity } from 'lucide-react';
import * as ort from 'onnxruntime-web';

// Initialize ONNX wasm paths
if (typeof window !== 'undefined') {
    ort.env.wasm.wasmPaths = "/";
    ort.env.wasm.numThreads = 1; 
}

export default function RecordingTab() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // -- APP STATE --
  const [isRecording, setIsRecording] = useState(false);
  const [visRaw, setVisRaw] = useState<number[]>([]);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isTorchOn, setIsTorchOn] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Ready");
  
  // -- ONNX MODEL STATE --
  const [onnxSession, setOnnxSession] = useState<ort.InferenceSession | null>(null);

  // -- USER FORM STATE --
  const [showUserForm, setShowUserForm] = useState(false);
  const [userDetails, setUserDetails] = useState({
      name: '',
      age: '30',
      height: '170',
      weight: '70'
  });

  // -- RESULTS STATE --
  const [showResults, setShowResults] = useState(false);
  const [pendingSession, setPendingSession] = useState<RecordingSession | null>(null);
  const [modelResults, setModelResults] = useState<{sbp:number, dbp:number, glucose:number, hr:number, hrv:number} | null>(null);

  const recordedSamplesRef = useRef<{ timestamp: number; value: number }[]>([]);
  const rpPgRef = useRef<RPPGAcquisition | null>(null);

  useEffect(() => {
    // 1. Remember the user details from local storage
    const savedDetails = localStorage.getItem('ppg_user_details');
    if (savedDetails) {
        try { setUserDetails(JSON.parse(savedDetails)); } catch(e) {}
    }

    // 2. Pre-load the ONNX model so it's ready immediately after recording
    ort.InferenceSession.create("/Ok_ppg_bp_glucose_final.onnx", { 
        executionProviders: ['wasm'],
    }).then(s => setOnnxSession(s)).catch(e => console.error("ONNX Load Error", e));

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

  const handleStartClick = () => {
    if (!videoRef.current || videoRef.current.readyState < 2) {
        alert("Wait for camera to load...");
        return;
    }
    setShowUserForm(true);
  };

  const startRecording = () => {
    if(!userDetails.name || !userDetails.age || !userDetails.height || !userDetails.weight) {
        alert("Please fill all details");
        return;
    }

    // Save details so the user doesn't have to retype them
    localStorage.setItem('ppg_user_details', JSON.stringify(userDetails));

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

  const stopAndAnalyze = async () => {
    setIsRecording(false);
    if(recordingIntervalRef.current) clearInterval(recordingIntervalRef.current);
    setStatusMsg("Analyzing...");

    if (recordingTime < 10) {
        alert("Recording too short! Need at least 10 seconds.");
        setStatusMsg("Ready");
        return;
    }

    try {
        const raw = recordedSamplesRef.current.map(s => s.value);
        const filtered = preprocessPPG(raw);
        
        // 1. Extract exactly the 18 model features
        const features = extractFeatures(filtered);
        
        const age = parseFloat(userDetails.age) || 30;
        const height = parseFloat(userDetails.height) || 170;
        const weight = parseFloat(userDetails.weight) || 70;

        let estSbp = 0, estDbp = 0, estGlucose = 0;

        // 2. Run Inference Immediately
        if (onnxSession) {
            const inputData = [...features, age, height, weight];
            const tensor = new ort.Tensor('float32', Float32Array.from(inputData), [1, 21]);

            const feeds: any = {};
            feeds[onnxSession.inputNames[0]] = tensor;
            
            const out = await onnxSession.run(feeds);
            const outputData = out[onnxSession.outputNames[0]].data as Float32Array; 
            
            // Apply saved calibration
            const savedCalib = localStorage.getItem('calibration_offsets');
            const offsets = savedCalib ? JSON.parse(savedCalib) : { sbp: 0, dbp: 0, glu: 0 };

            estSbp = outputData[0] + offsets.sbp;
            estDbp = outputData[1] + offsets.dbp;
            estGlucose = outputData[2] + offsets.glu;
        } else {
            console.warn("ONNX Model was not loaded yet.");
        }

        const hr = Math.round(features[6]);
        const hrv = parseFloat(features[7].toFixed(2));

        setModelResults({
            sbp: estSbp,
            dbp: estDbp,
            glucose: estGlucose,
            hr: hr,
            hrv: hrv
        });

        // 3. Prepare Session payload
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
            sbp: estSbp,
            dbp: estDbp,
            glucose: estGlucose,
            quality: 'Good'
        };

        setPendingSession(session);
        setShowResults(true);

    } catch (e) {
        console.error(e);
        alert("Analysis failed. Try again with a clearer signal and ensure finger fully covers the lens.");
        setStatusMsg("Error");
    }
  };

  const handleSave = async () => {
    if(pendingSession) {
        const storage = new SignalStorage();
        await storage.saveSession(pendingSession);
        alert(`Session Saved Successfully!`);
    }
    resetFlow();
  };

  const handleDiscard = () => {
    if(confirm("Discard this recording?")) {
        resetFlow();
    }
  };

  const resetFlow = () => {
    setShowResults(false);
    setPendingSession(null);
    setModelResults(null);
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

      {/* --- MODAL: RESULTS & INFERENCE --- */}
      {showResults && modelResults && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
            <div className="bg-card w-full max-w-sm rounded-xl border shadow-2xl overflow-hidden animate-in slide-in-from-bottom-10">
                <div className="bg-primary/10 p-4 text-center border-b">
                    <h2 className="text-xl font-bold text-primary flex justify-center items-center gap-2">
                        <Activity className="w-5 h-5" /> Vitals & Analysis
                    </h2>
                </div>

                <div className="p-6 space-y-6">
                    {/* HR & HRV (Calculated from features) */}
                    <div className="grid grid-cols-2 gap-4 text-center border-b pb-4 border-muted">
                        <div className="space-y-1">
                            <p className="text-xs text-muted-foreground uppercase">Heart Rate</p>
                            <p className="text-3xl font-mono font-bold">{modelResults.hr}</p>
                            <p className="text-[10px] text-muted-foreground">BPM</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-xs text-muted-foreground uppercase">HRV</p>
                            <p className="text-3xl font-mono font-bold">{modelResults.hrv}</p>
                            <p className="text-[10px] text-muted-foreground">ms (SDNN)</p>
                        </div>
                    </div>

                    {/* SBP & DBP (Predicted by Model) */}
                    <div className="grid grid-cols-2 gap-4 text-center border-b pb-4 border-muted">
                        <div className="space-y-1">
                            <p className="text-xs text-muted-foreground uppercase">Systolic</p>
                            <p className="text-3xl font-mono font-bold text-blue-600 dark:text-blue-400">{modelResults.sbp.toFixed(0)}</p>
                            <p className="text-xs text-muted-foreground">mmHg</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-xs text-muted-foreground uppercase">Diastolic</p>
                            <p className="text-3xl font-mono font-bold text-green-600 dark:text-green-400">{modelResults.dbp.toFixed(0)}</p>
                            <p className="text-xs text-muted-foreground">mmHg</p>
                        </div>
                    </div>
                    
                    {/* Glucose (Predicted by Model) */}
                    <div className="text-center">
                        <div className="space-y-1">
                            <p className="text-xs text-muted-foreground uppercase">Glucose</p>
                            <p className="text-3xl font-mono font-bold text-orange-600 dark:text-orange-400">{modelResults.glucose.toFixed(1)}</p>
                            <p className="text-[10px] text-muted-foreground">mg/dL</p>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-1 p-2 bg-muted/50">
                    <button onClick={handleDiscard} className="py-3 rounded bg-background border shadow-sm hover:bg-destructive/10 hover:text-destructive transition-colors font-medium">
                        Discard
                    </button>
                    <button onClick={handleSave} className="py-3 rounded bg-blue-600 text-white shadow-sm hover:bg-blue-700 transition-colors font-bold flex justify-center items-center gap-2">
                        <Save className="w-4 h-4"/> Save Session
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