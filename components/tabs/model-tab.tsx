'use client';

import React, { useEffect, useState } from 'react';
import { RecordingSession, SignalStorage, preprocessPPG, extractFeatures, performMathEstimation } from '@/lib/signal-processing';
import * as ort from 'onnxruntime-web';
import { Settings, RefreshCw, CheckCircle, Activity, Loader2, AlertTriangle } from 'lucide-react';

if (typeof window !== 'undefined') {
    ort.env.wasm.wasmPaths = "/";
    ort.env.wasm.numThreads = 1; 
}

const MODEL_PATH = "/Ok_ppg_bp_glucose_final.onnx";

export default function ModelTab() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [recordings, setRecordings] = useState<RecordingSession[]>([]);
  const [selectedRecId, setSelectedRecId] = useState<string>('');
  
  const [age, setAge] = useState(30);
  const [height, setHeight] = useState(170);
  const [weight, setWeight] = useState(70);
  
  const [refSBP, setRefSBP] = useState(120);
  const [refDBP, setRefDBP] = useState(80);
  const [refGlu, setRefGlu] = useState(100);
  const [offsets, setOffsets] = useState({ sbp: 0, dbp: 0, glu: 0 });

  const [result, setResult] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const log = (m: string) => setLogs(p => [m, ...p].slice(0, 50));

  useEffect(() => {
    const init = async () => {
        const recs = await new SignalStorage().getSessions();
        const sortedRecs = recs.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
        setRecordings(sortedRecs);

        const saved = localStorage.getItem('calibration_offsets');
        if (saved) setOffsets(JSON.parse(saved));

        try {
            setLoadingStatus("loading");
            log(`Auto-loading Model: ${MODEL_PATH}...`);
            const s = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] });
            setSession(s);
            setLoadingStatus("success");
            log("System Ready: Model Loaded Successfully");
        } catch (err: any) {
            console.error(err);
            setLoadingStatus("error");
            log(`Model Load Failed: ${err.message}`);
        }
    };
    init();
  }, []);

  const runModel = async () => {
    if (!session) return alert("Model is not loaded.");
    if (!selectedRecId) return alert("Please select a recording first.");
    const rec = recordings.find(r => r.id === selectedRecId);
    if (!rec) return;

    try {
        log("Starting Analysis...");
        const rawValues = rec.rawSignal.map(x => x.value);
        const processed = preprocessPPG(rawValues);
        
        let feats: number[];
        try {
            feats = extractFeatures(processed);
            log("Features Extracted");
        } catch (e: any) {
            log(`Feature Error: ${e.message}`);
            return;
        }
        
        // Math Estimation (Estimated)
        const mathEst = performMathEstimation(feats, age, height, weight);

        // ONNX Model (Predicted)
        const inputData = [...feats, age, height, weight];
        const tensor = new ort.Tensor('float32', Float32Array.from(inputData), [1, 21]);
        const feeds: any = {};
        feeds[session.inputNames[0]] = tensor;
        
        const out = await session.run(feeds);
        const outputData = out[session.outputNames[0]].data as Float32Array; 
        
        const final = {
            math: { sbp: mathEst.sbp, dbp: mathEst.dbp, glu: mathEst.glucose },
            pred: { 
                sbp: outputData[0] + offsets.sbp, 
                dbp: outputData[1] + offsets.dbp, 
                glu: outputData[2] + offsets.glu 
            },
            raw: { sbp: outputData[0], dbp: outputData[1], glu: outputData[2] }
        };

        setResult(final);
        log(`Done: AI SBP=${final.pred.sbp.toFixed(0)}`);

    } catch (err: any) {
        log(`System Error: ${err.message}`);
    }
  };

  const calibrate = () => {
    if(!result) return;
    const newOffsets = {
        sbp: refSBP - result.raw.sbp,
        dbp: refDBP - result.raw.dbp,
        glu: refGlu - result.raw.glu
    };
    setOffsets(newOffsets);
    localStorage.setItem('calibration_offsets', JSON.stringify(newOffsets));
    setResult({ ...result, pred: { sbp: refSBP, dbp: refDBP, glu: refGlu } });
    log(`✅ Calibrated using reference values.`);
  };

  return (
    <div className="p-4 space-y-6 pb-24">
       <div className={`bg-card border p-4 rounded-lg flex items-center justify-between ${loadingStatus === 'error' ? 'border-red-500 bg-red-50 dark:bg-red-950/30' : ''}`}>
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-full ${loadingStatus === 'loading' ? 'bg-blue-100 dark:bg-blue-900' : loadingStatus === 'success' ? 'bg-green-100 dark:bg-green-900' : 'bg-red-100 dark:bg-red-900'}`}>
                {loadingStatus === 'loading' && <Loader2 className="w-5 h-5 animate-spin text-blue-600"/>}
                {loadingStatus === 'success' && <CheckCircle className="w-5 h-5 text-green-600" />}
                {loadingStatus === 'error' && <AlertTriangle className="w-5 h-5 text-red-600" />}
                {loadingStatus === 'idle' && <Activity className="w-5 h-5 text-gray-400"/>}
            </div>
            <div>
                <h3 className="font-bold text-sm">Vital Analysis AI</h3>
                <p className="text-xs text-muted-foreground">
                    {loadingStatus === 'loading' && "Loading Model..."}
                    {loadingStatus === 'success' && "Model Ready"}
                    {loadingStatus === 'error' && "Model Not Found"}
                    {loadingStatus === 'idle' && "Initializing..."}
                </p>
            </div>
          </div>
       </div>

       <div className="grid grid-cols-3 gap-2">
         <div className="space-y-1"><label className="text-xs text-muted-foreground ml-1">Age</label><input type="number" value={age} onChange={e=>setAge(+e.target.value)} className="w-full border p-2 rounded bg-background text-center"/></div>
         <div className="space-y-1"><label className="text-xs text-muted-foreground ml-1">Height (cm)</label><input type="number" value={height} onChange={e=>setHeight(+e.target.value)} className="w-full border p-2 rounded bg-background text-center"/></div>
         <div className="space-y-1"><label className="text-xs text-muted-foreground ml-1">Weight (kg)</label><input type="number" value={weight} onChange={e=>setWeight(+e.target.value)} className="w-full border p-2 rounded bg-background text-center"/></div>
       </div>

       <div className="space-y-1">
           <label className="text-xs text-muted-foreground ml-1">Select Measurement Session</label>
           <select onChange={e=>{
               setSelectedRecId(e.target.value);
               const rec = recordings.find(r => r.id === e.target.value);
               if(rec) {
                   if(rec.age) setAge(rec.age);
                   if(rec.height) setHeight(rec.height);
                   if(rec.weight) setWeight(rec.weight);
               }
           }} className="w-full border p-3 rounded bg-background">
             <option value="">-- Select a Recording --</option>
             {recordings.map(r => <option key={r.id} value={r.id}>{r.id} - {r.patientName} ({new Date(r.startTime).toLocaleTimeString()})</option>)}
           </select>
       </div>

       <button onClick={runModel} disabled={!session || !selectedRecId} className="w-full bg-blue-600 hover:bg-blue-700 text-white p-3 rounded font-bold disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg">
          ANALYZE VITALS
       </button>

       {result && (
         <div className="bg-slate-100 dark:bg-slate-900 border p-4 rounded-lg space-y-4 animate-in fade-in slide-in-from-bottom-2">
            <h3 className="text-center text-xs font-bold uppercase text-muted-foreground tracking-widest bg-muted/50 p-1 rounded">Estimated (Math) vs Predicted (AI)</h3>
            
            <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-background p-2 rounded shadow-sm border space-y-2">
                    <div className="text-[10px] font-bold text-muted-foreground uppercase">Glucose</div>
                    <div className="flex flex-col items-center bg-muted/30 py-1 rounded">
                        <span className="text-xs line-through text-muted-foreground">{result.math.glu}</span>
                        <span className="text-xl font-bold text-orange-600">{result.pred.glu.toFixed(1)}</span>
                    </div>
                </div>
                <div className="bg-background p-2 rounded shadow-sm border space-y-2">
                    <div className="text-[10px] font-bold text-muted-foreground uppercase">SBP</div>
                    <div className="flex flex-col items-center bg-muted/30 py-1 rounded">
                        <span className="text-xs line-through text-muted-foreground">{result.math.sbp}</span>
                        <span className="text-xl font-bold text-blue-600">{result.pred.sbp.toFixed(0)}</span>
                    </div>
                </div>
                <div className="bg-background p-2 rounded shadow-sm border space-y-2">
                    <div className="text-[10px] font-bold text-muted-foreground uppercase">DBP</div>
                    <div className="flex flex-col items-center bg-muted/30 py-1 rounded">
                        <span className="text-xs line-through text-muted-foreground">{result.math.dbp}</span>
                        <span className="text-xl font-bold text-green-600">{result.pred.dbp.toFixed(0)}</span>
                    </div>
                </div>
            </div>
            
            <div className="border-t pt-4">
                <h4 className="text-sm font-bold flex gap-2 items-center mb-3"><Settings className="w-4 h-4"/> Calibration (Reference)</h4>
                <div className="flex gap-2 items-center">
                    <div className="grid grid-cols-3 gap-2 flex-1">
                        <input type="number" value={refSBP} onChange={e=>setRefSBP(+e.target.value)} className="w-full border p-2 rounded text-sm text-center bg-background" placeholder="SBP"/>
                        <input type="number" value={refDBP} onChange={e=>setRefDBP(+e.target.value)} className="w-full border p-2 rounded text-sm text-center bg-background" placeholder="DBP"/>
                        <input type="number" value={refGlu} onChange={e=>setRefGlu(+e.target.value)} className="w-full border p-2 rounded text-sm text-center bg-background" placeholder="Glu"/>
                    </div>
                    <button onClick={calibrate} className="bg-slate-800 text-white p-2 rounded hover:bg-slate-700"><RefreshCw className="w-5 h-5"/></button>
                </div>
            </div>
         </div>
       )}

       <div className="bg-black/90 text-green-400 p-3 rounded-lg h-32 overflow-auto text-[10px] font-mono border border-slate-700 shadow-inner">
         {logs.length === 0 && <div className="text-slate-500 italic">System logs will appear here...</div>}
         {logs.map((l,i) => <div key={i} className="border-b border-white/5 py-0.5">{l}</div>)}
       </div>
    </div>
  );
}