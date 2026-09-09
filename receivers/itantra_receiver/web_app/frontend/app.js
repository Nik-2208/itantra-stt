/**
 * iTantra Receiver Front-end Application (web_app/frontend/app.js)
 * Connects to local WebSocket server, drives immediate UI rendering,
 * plays streaming audio chunks, and visualizes benchmark milestones.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const srcLangSelect = document.getElementById('srcLangSelect');
  const tgtLangSelect = document.getElementById('tgtLangSelect');
  const voiceProfileSelect = document.getElementById('voiceProfileSelect');
  const styleModeSelect = document.getElementById('styleModeSelect');
  const pipelineModeSelect = document.getElementById('pipelineModeSelect');
  const chunkSizeSelect = document.getElementById('chunkSizeSelect');
  const textInput = document.getElementById('textInput');
  const btnSubmit = document.getElementById('btnSubmit');
  const btnStop = document.getElementById('btnStop');
  const btnClear = document.getElementById('btnClear');
  const btnExportJson = document.getElementById('btnExportJson');
  const btnExportCsv = document.getElementById('btnExportCsv');

  // Badges & Monitors
  const wsStatusBadge = document.getElementById('wsStatusBadge');
  const activeModeBadge = document.getElementById('activeModeBadge');
  const emergencyBanner = document.getElementById('emergencyBanner');
  const emergencyTitle = document.getElementById('emergencyTitle');
  const emergencyDetails = document.getElementById('emergencyDetails');
  const emergencyPriorityBadge = document.getElementById('emergencyPriorityBadge');

  // Text & Audio cards
  const originalTextDisplay = document.getElementById('originalTextDisplay');
  const originalDisplayTime = document.getElementById('originalDisplayTime');
  const translatedTextDisplay = document.getElementById('translatedTextDisplay');
  const translatedDisplayTime = document.getElementById('translatedDisplayTime');
  const playbackStatusBadge = document.getElementById('playbackStatusBadge');
  const audioProgressFill = document.getElementById('audioProgressFill');
  const speechTokensCount = document.getElementById('speechTokensCount');
  const audioChunksCount = document.getElementById('audioChunksCount');
  const audioDuration = document.getElementById('audioDuration');
  const eventTimeline = document.getElementById('eventTimeline');

  // Metrics
  const metricTTFA = document.getElementById('metricTTFA');
  const metricE2E = document.getElementById('metricE2E');
  const metricRTF = document.getElementById('metricRTF');
  const metricTtsTotal = document.getElementById('metricTtsTotal');
  const metricPeakRAM = document.getElementById('metricPeakRAM');
  const metricCurrentRAM = document.getElementById('metricCurrentRAM');
  const metricCPU = document.getElementById('metricCPU');
  const metricFailures = document.getElementById('metricFailures');
  const metricUnderruns = document.getElementById('metricUnderruns');

  // Comparison
  const compBaseTTFA = document.getElementById('compBaseTTFA');
  const compOptTTFA = document.getElementById('compOptTTFA');
  const compBaseE2E = document.getElementById('compBaseE2E');
  const compOptE2E = document.getElementById('compOptE2E');
  const compBaseRTF = document.getElementById('compBaseRTF');
  const compOptRTF = document.getElementById('compOptRTF');
  const compBaseRAM = document.getElementById('compBaseRAM');
  const compOptRAM = document.getElementById('compOptRAM');

  let ws = null;
  let audioPlayer = new window.StreamAudioPlayer(22050);
  let totalTokens = 0;
  let totalChunks = 0;
  let sessionStart = 0;

  audioPlayer.onProgress = (dur) => {
    audioDuration.textContent = `${dur.toFixed(1)}s`;
    audioProgressFill.style.width = `${Math.min(100, (dur / 4.0) * 100)}%`;
  };

  audioPlayer.onComplete = () => {
    playbackStatusBadge.textContent = 'COMPLETED';
    playbackStatusBadge.className = 'status-pill pill-idle';
    audioProgressFill.style.width = '100%';
  };

  function addTimelineEvent(timeOffsetSec, eventName) {
    if (eventTimeline.querySelector('.timeline-empty')) {
      eventTimeline.innerHTML = '';
    }
    const row = document.createElement('div');
    row.className = 'timeline-row';
    row.innerHTML = `
      <span class="timeline-time">${timeOffsetSec.toFixed(3)}s</span>
      <span class="timeline-event">${eventName}</span>
    `;
    eventTimeline.appendChild(row);
    eventTimeline.scrollTop = eventTimeline.scrollHeight;
  }

  function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/ws/receiver`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsStatusBadge.textContent = '● CONNECTED (LOCAL)';
      wsStatusBadge.className = 'badge badge-connected';
    };

    ws.onclose = () => {
      wsStatusBadge.textContent = 'DISCONNECTED';
      wsStatusBadge.className = 'badge badge-status';
      setTimeout(connectWebSocket, 2000);
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      const now = performance.now();
      const offset = sessionStart > 0 ? (now - sessionStart) / 1000.0 : 0.0;

      switch (msg.event_type) {
        case 'ORIGINAL_DISPLAYED':
          originalTextDisplay.textContent = msg.data.text;
          originalTextDisplay.classList.remove('placeholder-text');
          originalDisplayTime.textContent = `${msg.data.latency_ms.toFixed(2)} ms`;
          addTimelineEvent(offset, `TEXT_RECEIVED & DISPLAYED [${msg.data.source_language}]`);
          break;

        case 'EMERGENCY_DETECTED':
          addTimelineEvent(offset, `EMERGENCY_CHECK: ${msg.data.is_emergency ? 'ALERT' : 'NORMAL'} (${msg.data.latency_ms.toFixed(2)} ms)`);
          if (msg.data.is_emergency) {
            emergencyBanner.classList.remove('hidden');
            emergencyTitle.textContent = `EMERGENCY: ${msg.data.priority} DETECTED (${msg.data.categories.join(', ')})`;
            emergencyDetails.textContent = `Matched terms: ${msg.data.matched_terms.join(', ')} • Confidence: ${(msg.data.confidence * 100).toFixed(0)}%`;
            emergencyPriorityBadge.textContent = msg.data.priority;
          } else {
            emergencyBanner.classList.add('hidden');
          }
          break;

        case 'TRANSLATION_STARTED':
          addTimelineEvent(offset, `TRANSLATION_START [${msg.data.source_language} → ${msg.data.target_language}]`);
          break;

        case 'TRANSLATION_CHUNK':
          translatedTextDisplay.textContent = msg.data.chunk;
          translatedTextDisplay.classList.remove('placeholder-text');
          translatedDisplayTime.textContent = `${msg.data.chunk_latency_ms.toFixed(2)} ms`;
          addTimelineEvent(offset, `TRANSLATION_CHUNK (${msg.data.chunk_latency_ms.toFixed(2)} ms)`);
          break;

        case 'TRANSLATION_COMPLETE':
          translatedTextDisplay.textContent = msg.data.translated_text;
          translatedTextDisplay.classList.remove('placeholder-text');
          translatedDisplayTime.textContent = `${msg.data.total_translation_ms.toFixed(2)} ms`;
          addTimelineEvent(offset, `TRANSLATION_COMPLETE (${msg.data.total_translation_ms.toFixed(2)} ms)`);
          break;

        case 'TTS_STARTED':
          playbackStatusBadge.textContent = 'GENERATING';
          playbackStatusBadge.className = 'status-pill pill-playing';
          addTimelineEvent(offset, `TTS_START [${msg.data.target_language} • Chunk: ${msg.data.chunk_size}]`);
          break;

        case 'SPEECH_TOKEN_CHUNK':
          totalTokens += msg.data.token_count;
          speechTokensCount.textContent = totalTokens;
          addTimelineEvent(offset, `SPEECH_TOKEN_CHUNK (+${msg.data.token_count} tokens in ${msg.data.generation_ms.toFixed(2)} ms)`);
          break;

        case 'TTS_FIRST_AUDIO':
          metricTTFA.textContent = `${msg.data.ttfa_ms.toFixed(1)} ms`;
          addTimelineEvent(offset, `FIRST_AUDIO_READY (TTFA: ${msg.data.ttfa_ms.toFixed(1)} ms)`);
          playbackStatusBadge.textContent = 'PLAYING';
          break;

        case 'AUDIO_CHUNK':
          totalChunks += 1;
          audioChunksCount.textContent = totalChunks;
          if (msg.data.sample_rate) {
            audioPlayer.init(msg.data.sample_rate);
          }
          audioPlayer.queueBase64PCM(msg.data.pcm_b64, msg.data.is_final);
          break;

        case 'TTS_COMPLETE':
          addTimelineEvent(offset, `TTS_COMPLETE (Total: ${msg.data.tts_total_ms.toFixed(1)} ms)`);
          break;

        case 'BENCHMARK_RESULT':
          renderBenchmarkResult(msg.data);
          updateComparison();
          break;

        case 'ERROR':
          alert(`Pipeline error: ${msg.data.error}`);
          break;
      }
    };
  }

  function renderBenchmarkResult(metrics) {
    metricTTFA.textContent = `${metrics.ttfa_ms} ms`;
    metricE2E.textContent = `${metrics.e2e_latency_ms} ms`;
    metricRTF.textContent = `${metrics.rtf}`;
    metricTtsTotal.textContent = `${metrics.tts_total_ms} ms`;
    metricPeakRAM.textContent = `${metrics.peak_ram_mb} MB`;
    metricCurrentRAM.textContent = `${metrics.current_ram_mb} MB`;
    metricCPU.textContent = `${metrics.cpu_percent} %`;
    metricFailures.textContent = metrics.codec_failures;
    metricUnderruns.textContent = metrics.underruns;
  }

  async function updateComparison() {
    try {
      const res = await fetch('/api/benchmark/comparison');
      const data = await res.json();
      if (data.baseline.count > 0) {
        compBaseTTFA.textContent = `${data.baseline.avg_ttfa_ms} ms`;
        compBaseE2E.textContent = `${data.baseline.avg_e2e_ms} ms`;
        compBaseRTF.textContent = `${data.baseline.avg_rtf}`;
        compBaseRAM.textContent = `${data.baseline.avg_peak_ram_mb} MB`;
      }
      if (data.optimized.count > 0) {
        compOptTTFA.textContent = `${data.optimized.avg_ttfa_ms} ms`;
        compOptE2E.textContent = `${data.optimized.avg_e2e_ms} ms`;
        compOptRTF.textContent = `${data.optimized.avg_rtf}`;
        compOptRAM.textContent = `${data.optimized.avg_peak_ram_mb} MB`;
      }
    } catch (e) {}
  }

  // Quick Scenario Buttons
  document.querySelectorAll('.btn-scenario').forEach((btn) => {
    btn.addEventListener('click', () => {
      const src = btn.getAttribute('data-src');
      const tgt = btn.getAttribute('data-tgt');
      const text = btn.getAttribute('data-text');
      const mode = btn.getAttribute('data-mode') || 'NORMAL';

      if (src) srcLangSelect.value = src;
      if (tgt) tgtLangSelect.value = tgt;
      if (text) textInput.value = text;
      styleModeSelect.value = mode;
    });
  });

  // Submit button
  btnSubmit.addEventListener('click', () => {
    const text = textInput.value.trim();
    if (!text) {
      alert('Please enter text to transmit.');
      return;
    }
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      alert('WebSocket is connecting. Please wait a moment.');
      return;
    }

    // Reset session
    audioPlayer.reset();
    totalTokens = 0;
    totalChunks = 0;
    speechTokensCount.textContent = '0';
    audioChunksCount.textContent = '0';
    audioDuration.textContent = '0.0s';
    audioProgressFill.style.width = '0%';
    eventTimeline.innerHTML = '';
    sessionStart = performance.now();

    const mode = pipelineModeSelect.value;
    activeModeBadge.textContent = `${mode} MODE`;
    activeModeBadge.className = mode === 'OPTIMIZED' ? 'panel-tag tag-optimized' : 'panel-tag';

    const payload = {
      text: text,
      source_language: srcLangSelect.value,
      target_language: tgtLangSelect.value,
      accent_label: voiceProfileSelect.value,
      style_tag: styleModeSelect.value,
      mode: mode,
      chunk_size: parseInt(chunkSizeSelect.value, 10),
    };

    ws.send(JSON.stringify(payload));
  });

  btnStop.addEventListener('click', () => {
    audioPlayer.reset();
    playbackStatusBadge.textContent = 'STOPPED';
    playbackStatusBadge.className = 'status-pill pill-idle';
  });

  btnClear.addEventListener('click', () => {
    textInput.value = '';
    originalTextDisplay.textContent = 'Awaiting incoming message...';
    originalTextDisplay.classList.add('placeholder-text');
    translatedTextDisplay.textContent = 'Translation output will appear here...';
    translatedTextDisplay.classList.add('placeholder-text');
    emergencyBanner.classList.add('hidden');
    eventTimeline.innerHTML = '<div class="timeline-empty">Submit text to view live stage milestones...</div>';
  });

  btnExportJson.addEventListener('click', () => {
    window.open('/api/benchmark/export/json', '_blank');
  });

  btnExportCsv.addEventListener('click', () => {
    window.open('/api/benchmark/export/csv', '_blank');
  });

  connectWebSocket();
  updateComparison();
});
