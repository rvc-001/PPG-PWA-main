/**
 * lib/signal-processing.ts
 * ROBUST VERSION WITH DETAILED ERROR REPORTING
 */

const FS = 30;
const TRIM_SEC = 3; 

export interface FilterConfig {
  lowCutoff: number;     // Hz
  highCutoff: number;    // Hz
  order: number;         // Butterworth order
  samplingRate: number;  // Hz
}



export interface SignalSample {
  timestamp: number;
  value: number;
}

export interface RecordingSession {
  id: string;
  startTime: number;
  endTime?: number;
  samplingRate: number;
  rawSignal: SignalSample[];
  createdAt: Date;
  patientName?: string;
  patientId?: string;
  features?: number[];
  sbp?: number;
  dbp?: number;
  quality?: string;
  prediction?: { sbp: number; dbp: number; glu: number };
}

// ============================================================================
// PART 2: MATH HELPERS (Zero Dependency)
// ============================================================================

const _mean = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
const _min = (arr: number[]) => arr.length ? Math.min(...arr) : 0;
const _max = (arr: number[]) => arr.length ? Math.max(...arr) : 0;

function _std(arr: number[]) {
    if (arr.length <= 1) return 0;
    const m = _mean(arr);
    return Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / arr.length);
}

function gradient(data: number[]): number[] {
    const n = data.length;
    if (n < 2) return new Array(n).fill(0);
    const out = new Array(n).fill(0);
    out[0] = data[1] - data[0];
    out[n - 1] = data[n - 1] - data[n - 2];
    for (let i = 1; i < n - 1; i++) {
        out[i] = (data[i + 1] - data[i - 1]) / 2;
    }
    return out;
}

function trapz(y: number[]): number {
    let sum = 0;
    for (let i = 0; i < y.length - 1; i++) {
        sum += 0.5 * (y[i] + y[i+1]);
    }
    return sum;
}

function getWelchLF(signal: number[], fs: number): number {
    const nperseg = Math.min(256, signal.length); // Adapt to length
    if (nperseg < 32) return 0; // Too short for spectral analysis

    const step = Math.floor(nperseg / 2);
    const nWindows = Math.floor((signal.length - nperseg) / step) + 1;
    
    if (nWindows < 1) return 0; 

    const psdAccumulator = new Array(Math.floor(nperseg/2) + 1).fill(0);
    const window = new Array(nperseg).fill(0).map((_, i) => 
        0.5 * (1 - Math.cos((2 * Math.PI * i) / (nperseg - 1)))
    );

    for (let w = 0; w < nWindows; w++) {
        const start = w * step;
        const segment = signal.slice(start, start + nperseg);
        // Simple DFT
        const fftSize = nperseg;
        const spectrum = new Array(Math.floor(fftSize/2) + 1).fill(0);
        
        for (let k = 0; k < spectrum.length; k++) {
            let real = 0, imag = 0;
            const angularFreq = -2 * Math.PI * k / fftSize;
            for (let n = 0; n < fftSize; n++) {
                const wVal = segment[n] * window[n];
                const angle = angularFreq * n;
                real += wVal * Math.cos(angle);
                imag += wVal * Math.sin(angle);
            }
            spectrum[k] = (real*real + imag*imag);
        }
        for (let k = 0; k < spectrum.length; k++) psdAccumulator[k] += spectrum[k];
    }

    const winSumSq = window.reduce((a, b) => a + b*b, 0);
    const scale = 1.0 / (fs * winSumSq || 1);
    const avgPSD = psdAccumulator.map(v => (v / nWindows) * scale);
    
    for (let k = 1; k < avgPSD.length - 1; k++) avgPSD[k] *= 2;

    let lfSum = 0;
    const freqRes = fs / nperseg;
    for (let k = 0; k < avgPSD.length; k++) {
        const f = k * freqRes;
        if (f >= 0.01 && f <= 0.15) lfSum += avgPSD[k];
    }
    return lfSum;
}

// ============================================================================
// PART 3: PREPROCESSING & FEATURES
// ============================================================================

export function preprocessPPG(raw: number[]): number[] {
    if (!raw || raw.length === 0) return [];

    // Check for NaNs in input
    if (raw.some(isNaN)) return raw.map(v => isNaN(v) ? 0 : v);

    // 1. Bandpass (Butterworth 4th order, 0.5-5.0Hz, fs=30)
    const b = [0.032623, 0.0, -0.130493, 0.0, 0.195740, 0.0, -0.130493, 0.0, 0.032623];
    const a = [1.0, -4.8465, 10.6666, -14.1672, 12.0620, -6.6582, 2.3387, -0.4746, 0.0433];
    
    let signal = filtfilt(b, a, raw);
    
    // Safety check: if filter exploded (values > 1e6 or NaN), revert to raw
    const maxVal = _max(signal.map(Math.abs));
    if (isNaN(maxVal) || maxVal > 1e9) {
        console.warn("Filter unstable. Using raw signal.");
        signal = [...raw]; // Fallback
    }

    // 2. Gaussian Filter (sigma=2)
    signal = gaussianFilter1d(signal, 2);

    // 3. Trim (3 seconds from start/end) ONLY if signal is long enough
    const trimSamples = TRIM_SEC * FS;
    if (signal.length > 2.5 * trimSamples) {
        signal = signal.slice(trimSamples, signal.length - trimSamples);
    } 
    
    return signal;
}

export function extractFeatures(ppg: number[]): number[] {
    // 1. Basic Checks
    if (!ppg || ppg.length < 30) throw new Error(`Signal too short (${ppg?.length || 0} pts). Need > 30.`);
    
    const stdVal = _std(ppg);
    if (stdVal < 1e-4) throw new Error(`Signal flatline (Std=${stdVal.toFixed(5)}). Is camera covered?`);

    // 2. Peak Detection (Relaxed: dist=10 samples ~ 0.33s -> Max 180 BPM)
    const peaks = findPeaks(ppg, 10);
    const valleys = findPeaks(ppg.map(x => -x), 10);

    if (peaks.length < 2) throw new Error(`Not enough peaks (${peaks.length}). Hold finger still.`);
    if (valleys.length < 1) throw new Error("No valleys found.");

    // Ensure matched pairs (roughly) for slope calc
    // We need at least 2 peaks and 1 valley in between or before
    
    const peakValues = peaks.map(i => ppg[i]);
    const peak_val = _mean(peakValues);
    const eps = 1e-6; 

    // Safe Access Utils
    const safeP = (idx: number) => ppg[peaks[Math.min(idx, peaks.length-1)]];
    const safeV = (idx: number) => ppg[valleys[Math.min(idx, valleys.length-1)]];
    const safePi = (idx: number) => peaks[Math.min(idx, peaks.length-1)];
    const safeVi = (idx: number) => valleys[Math.min(idx, valleys.length-1)];

    // 1. RI
    const RI = safeV(0) / (peak_val + eps);
    // 2. AIx
    const AIx = (_max(ppg) - _min(ppg)) / (peak_val + eps);
    // 3. sys_slope
    const sys_num = safeP(0) - safeV(0);
    const sys_den = Math.abs(safePi(0) - safeVi(0)) + eps;
    const sys_slope = sys_num / sys_den;
    // 4. dia_slope
    const dia_num = safeP(1) - safeV(0);
    const dia_den = Math.abs(safePi(1) - safeVi(0)) + eps;
    const dia_slope = dia_num / dia_den;
    // 5-6. PW50, PW75
    const PW50 = ppg.filter(v => v > 0.5 * peak_val).length / FS;
    const PW75 = ppg.filter(v => v > 0.75 * peak_val).length / FS;
    // 7-8. HR, HRV
    const HR = peaks.length * (60 / (ppg.length / FS));
    const RR_intervals = [];
    for(let i=1; i<peaks.length; i++) RR_intervals.push((peaks[i] - peaks[i-1])/FS);
    const HRV = RR_intervals.length > 0 ? _std(RR_intervals) : 0;
    // 9. AUC
    const AUC = trapz(ppg);

    // 10-14. Stiffness (SDPPG)
    const d1 = gradient(ppg);
    const d2 = gradient(d1);
    const meanD2 = _mean(d2);
    const stdD2 = _std(d2);
    const d2_norm = d2.map(x => (x - meanD2) / (stdD2 + eps));
    const sp = findPeaks(d2_norm, 8); // distance=8

    let BA=0, CA=0, DA=0, EA=0, STIFF=0;
    // If not enough d2 peaks, fill 0 (don't crash)
    if (sp.length >= 5) {
        const [ia, ib, ic, id, ie] = sp.slice(0, 5);
        const a = d2_norm[ia];
        const b = d2_norm[ib];
        const c = d2_norm[ic];
        const d = d2_norm[id];
        const e = d2_norm[ie];
        BA = b/(a+eps); CA = c/(a+eps); DA = d/(a+eps); EA = e/(a+eps);
        STIFF = Math.abs(a - b);
    }

    // 15-18. Statistical & Spectral
    const mean_ppg = _mean(ppg);
    const std_ppg = _std(ppg);
    
    // Trend
    const smooth5 = gaussianFilter1d(ppg, 5);
    const trendGrad = gradient(smooth5);
    const baseline_trend = _mean(trendGrad);

    const LF = getWelchLF(ppg, FS);

    return [
        RI, AIx, sys_slope, dia_slope, PW50, PW75,
        HR, HRV, AUC, BA, CA, DA, EA, STIFF,
        mean_ppg, std_ppg, baseline_trend, LF
    ];
}

// ============================================================================
// PART 4: APP UTILITIES
// ============================================================================

export function applyFilterToArray(data: number[]): number[] {
    return preprocessPPG(data);
}

export function calculateSignalStats(data: number[]) {
    if (!data || data.length === 0) return { mean: 0, std: 0, min: 0, max: 0 };
    return { 
        mean: _mean(data), 
        std: _std(data), 
        min: _min(data), 
        max: _max(data) 
    };
}

export function generateMIMICCSV(session: RecordingSession, startTime: number, endTime: number): string {
    const rows = [['Timestamp', 'PPG']];
    const data = session.rawSignal.filter(s => s.timestamp >= startTime && s.timestamp <= endTime);
    data.forEach(s => rows.push([new Date(s.timestamp).toISOString(), s.value.toString()]));
    return rows.map(r => r.join(',')).join('\n');
}

// ============================================================================
// PART 5: STORAGE UTILS
// ============================================================================

export class SignalStorage {
  async saveSession(session: RecordingSession) {
    const sessions = await this.getSessions();
    sessions.push(session);
    if (sessions.length > 50) sessions.shift();
    localStorage.setItem('ppg_sessions', JSON.stringify(sessions));
  }

  async getSessions(): Promise<RecordingSession[]> {
    if (typeof window === 'undefined') return [];
    const s = localStorage.getItem('ppg_sessions');
    if(!s) return [];
    try {
        return JSON.parse(s).map((x:any) => ({
            ...x,
            createdAt: new Date(x.createdAt)
        }));
    } catch (e) {
        console.error("Failed to parse sessions", e);
        return [];
    }
  }

  async deleteSession(id: string) {
    const sessions = await this.getSessions();
    const filtered = sessions.filter(s => s.id !== id);
    localStorage.setItem('ppg_sessions', JSON.stringify(filtered));
  }

  // ADDED THIS METHOD TO FIX THE BUILD ERROR
  async clearAll() {
    localStorage.removeItem('ppg_sessions');
  }
}

// ============================================================================
// PART 6: LOW LEVEL MATH UTILS
// ============================================================================

function filtfilt(b: number[], a: number[], x: number[]): number[] {
    // Basic LFILTER implementation (Direct Form I)
    const forward = lfilter(b, a, x);
    // Reverse and filter again to cancel phase delay
    const backward = lfilter(b, a, forward.reverse());
    return backward.reverse();
}

function lfilter(b: number[], a: number[], x: number[]): number[] {
    const y = new Array(x.length).fill(0);
    // Helper to prevent NaN propagation
    const safe = (v: number) => isNaN(v) ? 0 : v;

    for (let n = 0; n < x.length; n++) {
        // FIR part
        for (let i = 0; i < b.length; i++) {
            if (n - i >= 0) y[n] += b[i] * safe(x[n - i]);
        }
        // IIR part
        for (let j = 1; j < a.length; j++) {
            if (n - j >= 0) y[n] -= a[j] * safe(y[n - j]);
        }
        y[n] /= a[0];
        
        // Clamp to prevent infinity explosion
        if(Math.abs(y[n]) > 1e9) y[n] = 0;
    }
    return y;
}

function gaussianFilter1d(data: number[], sigma: number): number[] {
    const radius = Math.ceil(4 * sigma);
    const kernel: number[] = [];
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
        const val = Math.exp(-(i * i) / (2 * sigma * sigma));
        kernel.push(val);
        sum += val;
    }
    const normKernel = kernel.map(k => k / sum);
    
    const result = new Array(data.length).fill(0);
    for (let i = 0; i < data.length; i++) {
        let val = 0;
        for (let j = 0; j < normKernel.length; j++) {
            const idx = Math.min(Math.max(i + (j - radius), 0), data.length - 1);
            val += data[idx] * normKernel[j];
        }
        result[i] = val;
    }
    return result;
}

function findPeaks(data: number[], distance: number): number[] {
    const candidates: number[] = [];
    for(let i=1; i<data.length-1; i++){
        // Simple local maxima check
        if(data[i] > data[i-1] && data[i] > data[i+1]){
            candidates.push(i);
        }
    }
    candidates.sort((a,b) => data[b] - data[a]); 
    
    const kept: number[] = [];
    for(const c of candidates){
        if(!kept.some(k => Math.abs(k - c) < distance)){
            kept.push(c);
        }
    }
    return kept.sort((a,b) => a - b);
}