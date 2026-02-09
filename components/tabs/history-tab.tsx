'use client';

import { useEffect, useState, useMemo } from 'react';
import { RecordingSession, SignalStorage, generateMIMICCSV, calculateSignalStats, applyFilterToArray } from '@/lib/signal-processing';
import SignalVisualizer from '@/components/visualization/signal-visualizer';
import { Trash2, Download, ChevronLeft, Scissors } from 'lucide-react';

type ViewMode = 'list' | 'detail';

export default function HistoryTab() {
  const [sessions, setSessions] = useState<RecordingSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [selectedSession, setSelectedSession] = useState<RecordingSession | null>(null);
  
  // Changed to strings to handle empty inputs without NaN warnings
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
    
    // Calculate duration
    let dur = 0;
    if (s.endTime && s.startTime) {
        dur = Math.floor((s.endTime - s.startTime) / 1000);
    } else {
        dur = Math.floor(s.rawSignal.length / 30); // Fallback estimate
    }
    
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
    // Safe parsing
    const sMin = parseInt(startMin) || 0;
    const sSec = parseInt(startSec) || 0;
    const eMin = parseInt(endMin) || 0;
    const eSec = parseInt(endSec) || 0;

    const sOff = (sMin * 60 + sSec) * 1000;
    const eOff = (eMin * 60 + eSec) * 1000;
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

  const { rawSlice, filteredSlice } = useMemo(() => {
    if (!selectedSession) return { rawSlice: [], filteredSlice: [] };
    const { start, end } = getTimestamps();
    
    // Safety check for empty signal
    if (!selectedSession.rawSignal || selectedSession.rawSignal.length === 0) {
        return { rawSlice: [], filteredSlice: [] };
    }

    const raw = selectedSession.rawSignal
        .filter(s => s.timestamp >= start && s.timestamp <= end)
        .map(s => s.value);
    
    // Only filter if we have enough data points (e.g. > 1 second)
    const filtered = raw.length > 30 ? applyFilterToArray(raw) : raw;
    
    return { rawSlice: raw, filteredSlice: filtered };
  }, [selectedSession, startMin, startSec, endMin, endSec]);

  const stats = useMemo(() => calculateSignalStats(filteredSlice), [filteredSlice]);

  if(loading) return <div className="p-8 text-center">Loading...</div>;

  if (viewMode === 'detail' && selectedSession) {
    return (
      <div className="w-full flex flex-col bg-background min-h-screen pb-32">
        <div className="p-4 border-b bg-card sticky top-0 z-30">
          <button onClick={() => setViewMode('list')} className="flex items-center gap-2 text-primary mb-2"><ChevronLeft className="w-4 h-4"/> Back</button>
          <div className="flex justify-between">
            <h2 className="font-bold">{selectedSession.patientName || 'Unknown'}</h2>
            <div className="flex gap-2">
                <span className="text-xs border px-2 py-1 rounded bg-secondary/50">BP: {selectedSession.sbp || '?'}/{selectedSession.dbp || '?'}</span>
                <span className="text-xs border px-2 py-1 rounded">{selectedSession.quality || 'N/A'}</span>
            </div>
          </div>
        </div>

        <div className="p-4 space-y-4">
           {/* CLIP CONTROLS - Inputs fixed to use strings */}
           <div className="bg-card border rounded p-4">
             <div className="flex items-center gap-2 mb-2 font-bold text-sm"><Scissors className="w-4 h-4"/> Clip Range</div>
             <div className="flex items-center gap-4">
               <div>
                 <label className="text-xs">Start</label>
                 <div className="flex gap-1">
                    <input type="text" inputMode="numeric" value={startMin} onChange={e=>setStartMin(e.target.value)} className="w-12 bg-background border rounded text-center"/>
                    :
                    <input type="text" inputMode="numeric" value={startSec} onChange={e=>setStartSec(e.target.value)} className="w-12 bg-background border rounded text-center"/>
                 </div>
               </div>
               <div>
                 <label className="text-xs">End</label>
                 <div className="flex gap-1">
                    <input type="text" inputMode="numeric" value={endMin} onChange={e=>setEndMin(e.target.value)} className="w-12 bg-background border rounded text-center"/>
                    :
                    <input type="text" inputMode="numeric" value={endSec} onChange={e=>setEndSec(e.target.value)} className="w-12 bg-background border rounded text-center"/>
                 </div>
               </div>
             </div>
           </div>

           {/* GRAPHS */}
           <SignalVisualizer rawSignal={[]} filteredSignal={filteredSlice} title="Filtered Signal (Pulse)" color="emerald" height={140}/>
           <SignalVisualizer rawSignal={rawSlice} filteredSignal={[]} title="Raw Signal" color="cyan" height={100}/>

           {/* STATS */}
           <div className="bg-card border rounded p-4 grid grid-cols-4 gap-2 text-center text-xs">
              <div className="bg-background p-2 rounded"><p className="text-muted-foreground">Mean</p><p className="font-bold">{stats.mean.toFixed(2)}</p></div>
              <div className="bg-background p-2 rounded"><p className="text-muted-foreground">Std</p><p className="font-bold">{stats.std.toFixed(2)}</p></div>
              <div className="bg-background p-2 rounded"><p className="text-muted-foreground">Min</p><p className="font-bold">{stats.min.toFixed(2)}</p></div>
              <div className="bg-background p-2 rounded"><p className="text-muted-foreground">Max</p><p className="font-bold">{stats.max.toFixed(2)}</p></div>
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
        {sessions.map(s => (
          <div key={s.id} onClick={()=>handleSelectSession(s)} className="bg-card border p-4 rounded-lg cursor-pointer hover:border-primary transition-colors">
            <div className="flex justify-between font-bold"><span>{s.patientName || 'Unknown'}</span><span className="text-xs font-normal text-muted-foreground">{new Date(s.startTime).toLocaleDateString()}</span></div>
            <div className="text-xs mt-1 flex gap-2">
                <span className="bg-secondary px-2 rounded">{s.rawSignal.length} pts</span>
                <span className="bg-secondary px-2 rounded">{(s.rawSignal.length / 30).toFixed(1)}s</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}