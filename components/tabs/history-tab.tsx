'use client';

import { useEffect, useState, useMemo } from 'react';
import { RecordingSession, SignalStorage, generateMIMICCSV, calculateSignalStats, applyFilterToArray, performMathEstimation } from '@/lib/signal-processing';
import SignalVisualizer from '@/components/visualization/signal-visualizer';
import { Trash2, Download, ChevronLeft, Scissors, Activity, User } from 'lucide-react';
type ViewMode = 'list' | 'detail';

export default function HistoryTab() {
  const [sessions, setSessions] = useState<RecordingSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [selectedSession, setSelectedSession] = useState<RecordingSession | null>(null);
  
  const [startMin, setStartMin] = useState<string>('0');
  const [startSec, setStartSec] = useState<string>('0');
  const [endMin, setEndMin] = useState<string>('0');
  const [endSec, setEndSec] = useState<string>('0');

  useEffect(() => { loadSessions(); }, []);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const data = await new SignalStorage().getSessions();
      setSessions(data.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime()));
    } catch (e) { console.error(e); } finally { setLoading(false); }
  };

  const handleSelectSession = (s: RecordingSession) => {
    setSelectedSession(s);
    setStartMin('0'); 
    setStartSec('0');
    let dur = s.endTime && s.startTime ? Math.floor((s.endTime - s.startTime) / 1000) : Math.floor(s.rawSignal.length / 30);
    setEndMin(Math.floor(dur / 60).toString()); 
    setEndSec((dur % 60).toString());
    setViewMode('detail');
  };

  const handleDelete = async (id: string) => {
    if(!confirm('Delete recording?')) return;
    await new SignalStorage().deleteSession(id);
    setSessions(prev => prev.filter(s => s.id !== id));
    if(selectedSession?.id === id) { setViewMode('list'); setSelectedSession(null); }
  };

  const getTimestamps = () => {
    if(!selectedSession) return {start:0, end:0};
    const sOff = ((parseInt(startMin)||0) * 60 + (parseInt(startSec)||0)) * 1000;
    const eOff = ((parseInt(endMin)||0) * 60 + (parseInt(endSec)||0)) * 1000;
    return { start: selectedSession.startTime + sOff, end: selectedSession.startTime + eOff };
  };

  const handleExport = () => {
    if(!selectedSession) return;
    const {start, end} = getTimestamps();
    const csv = generateMIMICCSV(selectedSession, start, end);
    const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
    const pid = selectedSession.patientId ? selectedSession.patientId.replace(/[^a-z0-9]/gi, '') : 'anon';
    const pname = selectedSession.patientName ? selectedSession.patientName.replace(/[^a-z0-9]/gi, '') : 'user';
    const a = document.createElement('a');
    a.href = url;
    a.download = `${pid}_${pname}.csv`; 
    a.click();
  };

  const getMathVitals = (s: RecordingSession) => {
    if (!s.features || s.features.length < 18) return { hr: '-', hrv: '-', sbp: '-', dbp: '-' };
    const est = performMathEstimation(s.features, s.age || 30, s.height || 170, s.weight || 70);
    return {
        hr: Math.round(s.features[6]),
        hrv: s.features[7].toFixed(1),
        sbp: est.sbp,
        dbp: est.dbp
    };
  };

  const { rawSlice, filteredSlice } = useMemo(() => {
    if (!selectedSession) return { rawSlice: [], filteredSlice: [] };
    const { start, end } = getTimestamps();
    if (!selectedSession.rawSignal || selectedSession.rawSignal.length === 0) return { rawSlice: [], filteredSlice: [] };
    const raw = selectedSession.rawSignal.filter(s => s.timestamp >= start && s.timestamp <= end).map(s => s.value);
    const filtered = raw.length > 30 ? applyFilterToArray(raw) : raw;
    return { rawSlice: raw, filteredSlice: filtered };
  }, [selectedSession, startMin, startSec, endMin, endSec]);

  const stats = useMemo(() => calculateSignalStats(filteredSlice), [filteredSlice]);

  if(loading) return <div className="p-8 text-center">Loading...</div>;

  if (viewMode === 'detail' && selectedSession) {
    const v = getMathVitals(selectedSession);
    return (
      <div className="w-full flex flex-col bg-background min-h-screen pb-32">
        <div className="p-4 border-b bg-card sticky top-0 z-30">
          <button onClick={() => setViewMode('list')} className="flex items-center gap-2 text-primary mb-2"><ChevronLeft className="w-4 h-4"/> Back</button>
          <div className="flex justify-between items-center">
            <div>
                <h2 className="font-bold">{selectedSession.patientName || 'Unknown'}</h2>
                <p className="text-xs text-muted-foreground">{new Date(selectedSession.startTime).toLocaleString()}</p>
            </div>
          </div>
          <div className="grid grid-cols-4 gap-2 mt-4 text-center">
              <div className="bg-muted p-2 rounded border"><p className="text-[10px] uppercase text-muted-foreground">HR</p><p className="font-bold text-sm">{v.hr}</p></div>
              <div className="bg-muted p-2 rounded border"><p className="text-[10px] uppercase text-muted-foreground">HRV</p><p className="font-bold text-sm">{v.hrv}</p></div>
              <div className="bg-muted p-2 rounded border"><p className="text-[10px] uppercase text-muted-foreground">Est SBP</p><p className="font-bold text-sm text-blue-500">{v.sbp}</p></div>
              <div className="bg-muted p-2 rounded border"><p className="text-[10px] uppercase text-muted-foreground">Est DBP</p><p className="font-bold text-sm text-green-500">{v.dbp}</p></div>
          </div>
        </div>

        <div className="p-4 space-y-4">
           <div className="bg-card border rounded p-4">
             <div className="flex items-center gap-2 mb-2 font-bold text-sm"><Scissors className="w-4 h-4"/> Clip Range</div>
             <div className="flex items-center gap-4">
               <div>
                 <label className="text-xs">Start</label>
                 <div className="flex gap-1">
                    <input type="text" inputMode="numeric" value={startMin} onChange={e=>setStartMin(e.target.value)} className="w-12 bg-background border rounded text-center"/>:
                    <input type="text" inputMode="numeric" value={startSec} onChange={e=>setStartSec(e.target.value)} className="w-12 bg-background border rounded text-center"/>
                 </div>
               </div>
               <div>
                 <label className="text-xs">End</label>
                 <div className="flex gap-1">
                    <input type="text" inputMode="numeric" value={endMin} onChange={e=>setEndMin(e.target.value)} className="w-12 bg-background border rounded text-center"/>:
                    <input type="text" inputMode="numeric" value={endSec} onChange={e=>setEndSec(e.target.value)} className="w-12 bg-background border rounded text-center"/>
                 </div>
               </div>
             </div>
           </div>

           <SignalVisualizer rawSignal={[]} filteredSignal={filteredSlice} title="Filtered Signal (Pulse)" color="emerald" height={140}/>
           <SignalVisualizer rawSignal={rawSlice} filteredSignal={[]} title="Raw Signal" color="cyan" height={100}/>

           <div className="bg-card border rounded p-4 grid grid-cols-4 gap-2 text-center text-xs">
              <div className="bg-background p-2 rounded border"><p className="text-muted-foreground">Mean</p><p className="font-bold">{stats.mean.toFixed(2)}</p></div>
              <div className="bg-background p-2 rounded border"><p className="text-muted-foreground">Std</p><p className="font-bold">{stats.std.toFixed(2)}</p></div>
              <div className="bg-background p-2 rounded border"><p className="text-muted-foreground">Min</p><p className="font-bold">{stats.min.toFixed(2)}</p></div>
              <div className="bg-background p-2 rounded border"><p className="text-muted-foreground">Max</p><p className="font-bold">{stats.max.toFixed(2)}</p></div>
           </div>
        </div>

        <div className="fixed bottom-20 left-0 right-0 p-4 bg-card border-t border-border flex gap-2 z-50 shadow-[0_-5px_10px_rgba(0,0,0,0.1)]">
           <button onClick={handleExport} className="flex-1 flex items-center justify-center gap-2 bg-primary text-primary-foreground py-3 rounded-lg font-bold shadow-lg hover:brightness-110">
             <Download className="w-4 h-4" /> Export CSV
           </button>
           <button onClick={()=>handleDelete(selectedSession.id)} className="px-4 bg-destructive/10 text-destructive rounded-lg border border-destructive/20 hover:bg-destructive/20"><Trash2 className="w-5 h-5"/></button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full flex flex-col bg-background min-h-screen">
      <div className="p-4 border-b"><h1 className="text-2xl font-bold">History</h1><p className="text-sm text-muted-foreground">{sessions.length} sessions</p></div>
      <div className="flex-1 overflow-auto p-4 space-y-3 pb-24">
        {sessions.map(s => {
          const v = getMathVitals(s);
          return (
            <div key={s.id} onClick={()=>handleSelectSession(s)} className="bg-card border p-4 rounded-lg cursor-pointer hover:border-primary transition-colors flex flex-col gap-3">
                <div className="flex justify-between items-start">
                    <div>
                        <span className="font-bold flex items-center gap-1"><User className="w-4 h-4 text-muted-foreground"/> {s.patientName || 'Unknown'}</span>
                        <span className="text-xs font-normal text-muted-foreground block mt-1">{new Date(s.startTime).toLocaleDateString()} at {new Date(s.startTime).toLocaleTimeString()}</span>
                    </div>
                    <div className="text-right text-xs">
                        <span className="bg-secondary px-2 py-1 rounded-full text-secondary-foreground font-medium flex items-center gap-1"><Activity className="w-3 h-3"/> {(s.rawSignal.length / 30).toFixed(1)}s</span>
                    </div>
                </div>
                
                <div className="grid grid-cols-4 gap-2 text-center bg-muted/30 p-2 rounded border border-muted">
                    <div><p className="text-[9px] uppercase text-muted-foreground">HR</p><p className="font-medium text-xs">{v.hr}</p></div>
                    <div><p className="text-[9px] uppercase text-muted-foreground">HRV</p><p className="font-medium text-xs">{v.hrv}</p></div>
                    <div><p className="text-[9px] uppercase text-blue-500/70">Est SBP</p><p className="font-medium text-xs text-blue-500">{v.sbp}</p></div>
                    <div><p className="text-[9px] uppercase text-green-500/70">Est DBP</p><p className="font-medium text-xs text-green-500">{v.dbp}</p></div>
                </div>
            </div>
          )
        })}
      </div>
    </div>
  );
}