/**
 * iTantra Audio Player (web_app/frontend/audio/audio-player.js)
 * Low-latency Web Audio API streaming PCM player with jitter buffer.
 */
class StreamAudioPlayer {
  constructor(sampleRate = 22050) {
    this.sampleRate = sampleRate;
    this.audioCtx = null;
    this.nextStartTime = 0;
    this.isPlaying = false;
    this.totalSamplesQueued = 0;
    this.onProgress = null;
    this.onComplete = null;
  }

  init(sr = null) {
    if (sr && sr !== this.sampleRate) {
      this.reset();
      this.sampleRate = sr;
    }
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContext({ sampleRate: this.sampleRate });
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  reset() {
    if (this.audioCtx) {
      try {
        this.audioCtx.close();
      } catch (e) {}
      this.audioCtx = null;
    }
    this.nextStartTime = 0;
    this.isPlaying = false;
    this.totalSamplesQueued = 0;
  }

  /**
   * Enqueues a base64 encoded signed 16-bit PCM chunk.
   */
  queueBase64PCM(b64Data, isFinal = false) {
    this.init();

    // Decode base64 to byte array
    const binaryStr = atob(b64Data);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }

    // Convert 16-bit PCM to float32
    const int16View = new Int16Array(bytes.buffer);
    const float32Data = new Float32Array(int16View.length);
    for (let i = 0; i < int16View.length; i++) {
      float32Data[i] = int16View[i] / 32768.0;
    }

    if (float32Data.length === 0) return;

    // Create AudioBuffer
    const audioBuffer = this.audioCtx.createBuffer(1, float32Data.length, this.sampleRate);
    audioBuffer.getChannelData(0).set(float32Data);

    const source = this.audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioCtx.destination);

    const currentTime = this.audioCtx.currentTime;
    // Small jitter buffer: schedule next chunk immediately or seamlessly at tail
    if (this.nextStartTime < currentTime) {
      this.nextStartTime = currentTime + 0.02; // 20ms jitter buffer
    }

    source.start(this.nextStartTime);
    this.isPlaying = true;

    const chunkDuration = float32Data.length / this.sampleRate;
    this.nextStartTime += chunkDuration;
    this.totalSamplesQueued += float32Data.length;

    if (this.onProgress) {
      this.onProgress(this.totalSamplesQueued / this.sampleRate);
    }

    if (isFinal) {
      source.onended = () => {
        this.isPlaying = false;
        if (this.onComplete) this.onComplete();
      };
    }
  }
}

window.StreamAudioPlayer = StreamAudioPlayer;
