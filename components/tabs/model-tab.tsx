'use client';

import React, { useEffect, useState } from 'react';
import { RecordingSession, SignalStorage, preprocessPPG, extractFeatures } from '@/lib/signal-processing';
import * as ort from 'onnxruntime-web';
import { Settings, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';

if (typeof window !== 'undefined') {
    ort.env.wasm.wasmPaths = "/";
    ort.env.wasm.numThreads = 1; 
}

export default function ModelTab() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [recordings, setRecordings] = useState<RecordingSession[]>([]);
  const [selectedRecId, setSelectedRecId] = useState<string>('');
  
  // Inputs
  const [age, setAge] = useState(30);
  const [height, setHeight] = useState(170);
  const [weight, setWeight] = useState(70);
  
  // Calibration
  const [refSBP, setRefSBP] = useState(120);
  const [refDBP, setRefDBP] = useState(80);
  const [refGlu, setRefGlu] = useState(100);
  const [offsets, setOffsets] = useState({ sbp: 0, dbp: 0, glu: 0 });

  const [result, setResult] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    new SignalStorage().getSessions().then(setRecordings);
    const saved = localStorage.getItem('calibration_offsets');
    if (saved) setOffsets(JSON.parse(saved));
  }, []);

  const log = (m: string) => setLogs(p => [m, ...p].slice(0, 50));

  const loadModel = async (e: any) => {
    try {
        const file = e.target.files[0];
        if(!file) return;
        log("📂 Loading Model...");
        const buf = await file.arrayBuffer();
        const s = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
        setSession(s);
        log("✅ Model Loaded!");
    } catch(err: any) {
        log(`❌ Model Error: ${err.message}`);
    }
  };

  const runModel = async () => {
    if (!session) { alert("Load Model first"); return; }
    if (!selectedRecId) { alert("Select Recording"); return; }
    const rec = recordings.find(r => r.id === selectedRecId);
    if (!rec) return;

    try {
        log("🔄 Starting Analysis...");
        
        // 1. Get Raw Data
        const rawValues = rec.rawSignal.map(x => x.value);
        log(`📊 Signal Length: ${rawValues.length} samples`);

        // 2. Preprocess
        const processed = preprocessPPG(rawValues);
        
        // 3. Extract Features (Now throws errors if bad)
        let feats: number[];
        try {
            feats = extractFeatures(processed);
            log("✅ Features Extracted");
        } catch (e: any) {
            log(`❌ Feature Error: ${e.message}`);
            log("👉 TIP: Record again with finger covering camera + flash.");
            return;
        }
        
        // 4. Inference
        const inputData = [...feats, age, height, weight];
        const tensor = new ort.Tensor('float32', Float32Array.from(inputData), [1, 21]);

        const feeds: any = {};
        feeds[session.inputNames[0]] = tensor;
        
        const out = await session.run(feeds);
        const outputData = out[session.outputNames[0]].data as Float32Array; 
        
        // 5. Apply Calibration
        const final = {
            sbp: outputData[0] + offsets.sbp,
            dbp: outputData[1] + offsets.dbp,
            glu: outputData[2] + offsets.glu,
            raw: { sbp: outputData[0], dbp: outputData[1], glu: outputData[2] }
        };

        setResult(final);
        log(`🎉 Done: SBP=${final.sbp.toFixed(0)}`);

    } catch (err: any) {
        log(`❌ System Error: ${err.message}`);
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
    setResult({ ...result, sbp: refSBP, dbp: refDBP, glu: refGlu });
    log(`✅ Calibrated using reference values.`);
  };

  return (
    <div className="p-4 space-y-6 pb-24">
       {/* LOAD MODEL */}
       <div className="bg-card border p-4 rounded-lg flex items-center justify-between">
          <div className="text-sm font-bold flex items-center gap-2">
            <CheckCircle className={`w-5 h-5 ${session ? 'text-green-500' : 'text-gray-300'}`} />
            Load Model
          </div>
          <input type="file" onChange={loadModel} accept=".onnx" className="text-xs"/>
       </div>

       {/* DEMOGRAPHICS */}
       <div className="grid grid-cols-3 gap-2">
         <input type="number" placeholder="Age" value={age} onChange={e=>setAge(+e.target.value)} className="border p-2 rounded bg-background" />
         <input type="number" placeholder="Height" value={height} onChange={e=>setHeight(+e.target.value)} className="border p-2 rounded bg-background" />
         <input type="number" placeholder="Weight" value={weight} onChange={e=>setWeight(+e.target.value)} className="border p-2 rounded bg-background" />
       </div>

       {/* SELECT RECORDING */}
       <select onChange={e=>setSelectedRecId(e.target.value)} className="w-full border p-3 rounded bg-background">
         <option value="">Select Recording...</option>
         {recordings.map(r => <option key={r.id} value={r.id}>{new Date(r.startTime).toLocaleTimeString()} ({((r.rawSignal?.length||0)/30).toFixed(1)}s)</option>)}
       </select>

       <button onClick={runModel} disabled={!session} className="w-full bg-blue-600 hover:bg-blue-700 text-white p-3 rounded font-bold disabled:opacity-50">
          ANALYZE SIGNAL
       </button>

       {/* RESULTS */}
       {result && (
         <div className="bg-slate-100 dark:bg-slate-900 border p-4 rounded-lg space-y-4">
            <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-background p-2 rounded shadow">
                    <div className="text-xs text-muted-foreground">SBP</div>
                    <div className="text-xl font-bold text-blue-600">{result.sbp.toFixed(0)}</div>
                </div>
                <div className="bg-background p-2 rounded shadow">
                    <div className="text-xs text-muted-foreground">DBP</div>
                    <div className="text-xl font-bold text-green-600">{result.dbp.toFixed(0)}</div>
                </div>
                <div className="bg-background p-2 rounded shadow">
                    <div className="text-xs text-muted-foreground">GLU</div>
                    <div className="text-xl font-bold text-orange-600">{result.glu.toFixed(1)}</div>
                </div>
            </div>
            
            <div className="border-t pt-4">
                <h4 className="text-sm font-bold flex gap-2 items-center mb-2"><Settings className="w-4 h-4"/> Calibration</h4>
                <div className="flex gap-2">
                    <input type="number" value={refSBP} onChange={e=>setRefSBP(+e.target.value)} className="w-full border p-1 rounded text-sm text-center" placeholder="SBP"/>
                    <input type="number" value={refDBP} onChange={e=>setRefDBP(+e.target.value)} className="w-full border p-1 rounded text-sm text-center" placeholder="DBP"/>
                    <input type="number" value={refGlu} onChange={e=>setRefGlu(+e.target.value)} className="w-full border p-1 rounded text-sm text-center" placeholder="Glu"/>
                    <button onClick={calibrate} className="bg-slate-800 text-white px-3 rounded"><RefreshCw className="w-4 h-4"/></button>
                </div>
            </div>
         </div>
       )}

       {/* LOGS */}
       <div className="bg-black text-green-400 p-2 rounded h-40 overflow-auto text-xs font-mono border border-slate-700">
         {logs.map((l,i) => <div key={i} className="border-b border-white/10 py-1">{l}</div>)}
       </div>
    </div>
  );
}