"""
iTantra Receiver Web App - Master Receiver Pipeline (web_app/pipeline/receiver_pipeline.py)
===========================================================================================
Master async pipeline orchestrating:
Incoming Text -> Validate -> Display Immediately -> Emergency Check ->
Language Routing -> IndicTrans2 Translation -> Indic-Mio TTS ->
MioCodec Decode -> Streaming Audio Chunks -> Real-Time Milestones.

Supports both OPTIMIZED streaming mode and BASELINE full-file generation mode.
"""

import asyncio
import base64
import math
import time
from typing import AsyncGenerator, Optional
from web_app.config.settings import LanguageCode, MessageType, SAMPLE_RATE, DEFAULT_TOKEN_CHUNK_SIZE
from web_app.pipeline.emergency_detector import EmergencyDetector
from web_app.pipeline.translation_engine import TranslationEngine
from web_app.pipeline.tts_engine import IndicMioAutoregressiveEngine, MioCodecStreamDecoder
from web_app.benchmark.timer import PipelineBenchmarkTimer
from web_app.benchmark.metrics import benchmark_store
from web_app.schemas.message import ReceivedMessage, EmergencyResult
from web_app.schemas.events import WebSocketEvent, EventType

class ReceiverPipeline:
    def __init__(self):
        self.emergency_detector = EmergencyDetector()
        self.translation_engine = TranslationEngine()
        self.tts_engine = IndicMioAutoregressiveEngine()

    async def execute_stream(
        self,
        message: ReceivedMessage,
        mode: str = "OPTIMIZED",
        chunk_size: int = DEFAULT_TOKEN_CHUNK_SIZE,
    ) -> AsyncGenerator[WebSocketEvent, None]:
        """
        Executes pipeline emitting typed WebSocket events asynchronously.
        """
        timer = PipelineBenchmarkTimer(
            message_id=message.message_id,
            source_lang=message.source_language.value,
            target_lang=message.target_language.value,
            message_type=message.message_type.value,
            mode=mode,
            chunk_size=chunk_size,
        )

        # 1. Immediate Text Display Event
        timer.mark_text_displayed()
        yield WebSocketEvent(
            event_type=EventType.ORIGINAL_DISPLAYED,
            message_id=message.message_id,
            data={
                "text": message.text,
                "source_language": message.source_language.value,
                "timestamp": timer.t1,
                "latency_ms": round((timer.t1 - timer.t0) * 1000.0, 2),
            },
        )
        await asyncio.sleep(0.001)

        # 2. Emergency Detection (Pass 1: Source)
        timer.mark_emergency_start()
        emergency_res = self.emergency_detector.detect_two_pass(
            source_text=message.text,
            source_lang=message.source_language.value,
        )
        timer.mark_emergency_complete()

        # Update message type if emergency detected
        if emergency_res.is_emergency:
            message.message_type = emergency_res.priority

        yield WebSocketEvent(
            event_type=EventType.EMERGENCY_DETECTED,
            message_id=message.message_id,
            data={
                "is_emergency": emergency_res.is_emergency,
                "priority": emergency_res.priority.value,
                "matched_terms": emergency_res.matched_terms,
                "categories": emergency_res.categories,
                "confidence": emergency_res.confidence,
                "detected_pass": emergency_res.detected_pass,
                "latency_ms": round((timer.t3 - timer.t2) * 1000.0, 2),
            },
        )
        await asyncio.sleep(0.001)

        # 3. Translation (if needed)
        timer.mark_translation_start()
        yield WebSocketEvent(
            event_type=EventType.TRANSLATION_STARTED,
            message_id=message.message_id,
            data={
                "source_language": message.source_language.value,
                "target_language": message.target_language.value,
            },
        )

        final_translated_text = message.text
        if message.source_language != message.target_language:
            for trans_chunk, is_final, c_latency in self.translation_engine.translate_stream(
                text=message.text,
                source_lang=message.source_language,
                target_lang=message.target_language,
            ):
                timer.mark_first_translation_chunk()
                yield WebSocketEvent(
                    event_type=EventType.TRANSLATION_CHUNK,
                    message_id=message.message_id,
                    data={
                        "chunk": trans_chunk,
                        "is_final": is_final,
                        "chunk_latency_ms": round(c_latency, 2),
                    },
                )
                final_translated_text = trans_chunk
                await asyncio.sleep(0.001)

            # Pass 2: Re-run emergency detection on translated text
            emergency_pass2 = self.emergency_detector.detect_two_pass(
                source_text=message.text,
                source_lang=message.source_language.value,
                translated_text=final_translated_text,
                target_lang=message.target_language.value,
            )
            if emergency_pass2.is_emergency:
                message.message_type = emergency_pass2.priority

        timer.mark_translation_complete()
        yield WebSocketEvent(
            event_type=EventType.TRANSLATION_COMPLETE,
            message_id=message.message_id,
            data={
                "translated_text": final_translated_text,
                "total_translation_ms": round((timer.t6 - timer.t4) * 1000.0, 2),
            },
        )

        # 4. Indic-Mio Speech Token & MioCodec Synthesis
        timer.mark_tts_start()
        yield WebSocketEvent(
            event_type=EventType.TTS_STARTED,
            message_id=message.message_id,
            data={
                "mode": mode,
                "target_language": message.target_language.value,
                "chunk_size": chunk_size,
            },
        )

        is_sos = (message.message_type == MessageType.SOS)

        if mode == "BASELINE":
            # Baseline: Wait for all tokens, decode entire audio, only then dispatch
            all_pcm = bytearray()
            all_tokens = []
            for pcm_chunk, tokens, is_final, t_tok_ms, t_codec_ms in self.tts_engine.generate_audio_stream(
                text=final_translated_text,
                language=message.target_language,
                voice_profile=message.voice_profile,
                chunk_size=chunk_size,
                is_sos=is_sos,
            ):
                timer.mark_first_text_token()
                timer.mark_first_speech_token()
                timer.mark_first_codec_audio()
                all_tokens.extend(tokens)
                all_pcm.extend(pcm_chunk)

            # Mark baseline audio available only at the end
            timer.mark_first_browser_audio()
            timer.mark_audio_chunk(len(all_pcm) // 2, sample_rate=SAMPLE_RATE)
            timer.mark_tts_complete()

            yield WebSocketEvent(
                event_type=EventType.TTS_FIRST_AUDIO,
                message_id=message.message_id,
                data={"ttfa_ms": round((timer.t11 - timer.t7) * 1000.0, 2)},
            )

            # Send baseline audio in safe frame slices
            slice_size = 65536
            num_slices = max(1, math.ceil(len(all_pcm) / slice_size))
            for s_idx in range(num_slices):
                start = s_idx * slice_size
                end = min(len(all_pcm), start + slice_size)
                slice_bytes = all_pcm[start:end]
                is_last = (s_idx == num_slices - 1)
                b64_audio = base64.b64encode(slice_bytes).decode("ascii")

                yield WebSocketEvent(
                    event_type=EventType.AUDIO_CHUNK,
                    message_id=message.message_id,
                    data={
                        "pcm_b64": b64_audio,
                        "is_final": is_last,
                        "chunk_index": s_idx,
                        "sample_rate": SAMPLE_RATE,
                    },
                )
                await asyncio.sleep(0.001)
        else:
            # OPTIMIZED: Streaming speech token chunks directly to browser AudioWorklet
            chunk_idx = 0
            first_audio_sent = False

            for pcm_chunk, tokens, is_final, t_tok_ms, t_codec_ms in self.tts_engine.generate_audio_stream(
                text=final_translated_text,
                language=message.target_language,
                voice_profile=message.voice_profile,
                chunk_size=chunk_size,
                is_sos=is_sos,
            ):
                timer.mark_first_text_token()
                timer.mark_first_speech_token()
                timer.mark_first_codec_audio()

                # Dispatch token chunk event
                yield WebSocketEvent(
                    event_type=EventType.SPEECH_TOKEN_CHUNK,
                    message_id=message.message_id,
                    data={
                        "token_count": len(tokens),
                        "sample_tokens": tokens[:4],
                        "generation_ms": round(t_tok_ms, 2),
                    },
                )

                if not first_audio_sent:
                    timer.mark_first_browser_audio()
                    first_audio_sent = True
                    yield WebSocketEvent(
                        event_type=EventType.TTS_FIRST_AUDIO,
                        message_id=message.message_id,
                        data={"ttfa_ms": round((timer.t11 - timer.t7) * 1000.0, 2)},
                    )

                samples = len(pcm_chunk) // 2
                timer.mark_audio_chunk(samples, sample_rate=SAMPLE_RATE)
                b64_chunk = base64.b64encode(pcm_chunk).decode("ascii")

                yield WebSocketEvent(
                    event_type=EventType.AUDIO_CHUNK,
                    message_id=message.message_id,
                    data={
                        "pcm_b64": b64_chunk,
                        "is_final": is_final,
                        "chunk_index": chunk_idx,
                        "sample_rate": SAMPLE_RATE,
                        "codec_latency_ms": round(t_codec_ms, 2),
                    },
                )
                chunk_idx += 1
                await asyncio.sleep(0.005)  # Yield to event loop for backpressure

            timer.mark_tts_complete()

        # 5. Final Benchmark Results
        metrics = timer.compute_metrics(text_length=len(message.text))
        benchmark_store.record(metrics)

        yield WebSocketEvent(
            event_type=EventType.TTS_COMPLETE,
            message_id=message.message_id,
            data={"tts_total_ms": metrics.tts_total_ms, "rtf": metrics.rtf},
        )

        yield WebSocketEvent(
            event_type=EventType.BENCHMARK_RESULT,
            message_id=message.message_id,
            data=metrics.model_dump(),
        )
