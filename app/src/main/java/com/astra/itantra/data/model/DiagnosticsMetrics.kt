package com.astra.itantra.data.model

data class DiagnosticsMetrics(
    val realTimeFactor: Float = 0.42f,       // RTF (e.g. 0.42 = processing 1s audio in 420ms)
    val sttLatencyMs: Long = 180L,            // STT inference latency
    val ttsLatencyMs: Long = 140L,            // TTS synthesis latency
    val endToEndDelayMs: Long = 390L,         // Total round-trip latency
    val ramUsageMb: Int = 185,                // App RAM footprint
    val cpuIdleUsagePercent: Float = 12.5f    // CPU utilization
)
