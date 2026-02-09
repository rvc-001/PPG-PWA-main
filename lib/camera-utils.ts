export class RPPGAcquisition {
    private frameInterval: number;
    private stream: MediaStream | null = null;
    private track: MediaStreamTrack | null = null;
  
    constructor(targetFps: number = 30) {
      this.frameInterval = 1000 / targetFps;
    }
  
    /**
     * Robust Camera Request
     */
    async requestCameraPermission(): Promise<MediaStream> {
      if (typeof window !== 'undefined' && 
          window.location.protocol !== 'https:' && 
          window.location.hostname !== 'localhost') {
        throw new Error("Camera access requires HTTPS. Please deploy with SSL.");
      }
  
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Browser API not supported");
      }
  
      try {
        this.stop();
  
        const constraints: MediaStreamConstraints = {
            audio: false,
            video: {
                facingMode: 'environment',
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            }
        };
  
        this.stream = await navigator.mediaDevices.getUserMedia(constraints);
        this.track = this.stream.getVideoTracks()[0];
  
        // Torch activation delay for Android stability
        setTimeout(async () => {
            await this.toggleTorch(true);
        }, 500);
  
        return this.stream;
      } catch (err) {
        console.error("Camera Error:", err);
        throw err;
      }
    }
  
    async toggleTorch(on: boolean): Promise<boolean> {
        if (!this.track) return false;
        try {
            const capabilities = this.track.getCapabilities() as any;
            if (!capabilities.torch) {
                console.warn("Device does not support Torch.");
                return false;
            }
            await this.track.applyConstraints({
                advanced: [{ torch: on } as any]
            });
            return true;
        } catch (e) {
            console.warn("Failed to toggle torch:", e);
            return false;
        }
    }
  
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
            this.track = null;
        }
    }
  
    /**
     * Extracts avg red intensity from the center of the video frame.
     * Removed internal throttling to allow caller (setInterval) to control rate.
     */
    extractSignal(video: HTMLVideoElement): number {
      const canvas = document.createElement('canvas');
      // Low resolution is sufficient for avg color
      canvas.width = 40; 
      canvas.height = 40;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      
      if (!ctx) return 0;
  
      // Draw center crop (50% width/height)
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw === 0 || vh === 0) return 0;

      ctx.drawImage(video, vw/4, vh/4, vw/2, vh/2, 0, 0, canvas.width, canvas.height);
  
      const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = frame.data;
      
      let sumRed = 0;
      let count = 0;
  
      for (let i = 0; i < data.length; i += 4) {
        sumRed += data[i]; // Red channel
        count++;
      }
  
      return count > 0 ? sumRed / count : 0;
    }
}
  
export function generateSimulatedSignal(baseHeartRate: number, samplingRate: number, seconds: number): number[] {
    const samples = samplingRate * seconds;
    const signal: number[] = [];
    for (let i = 0; i < samples; i++) {
      const t = i / samplingRate;
      const pulse = -Math.cos(2 * Math.PI * (baseHeartRate / 60) * t);
      const dicrotic = 0.5 * Math.cos(2 * Math.PI * (baseHeartRate / 60) * 2 * t + 0.5);
      const resp = 0.2 * Math.sin(2 * Math.PI * 0.25 * t);
      const noise = (Math.random() - 0.5) * 0.1;
      
      signal.push(100 + 10 * (pulse + dicrotic + resp + noise));
    }
    return signal;
}