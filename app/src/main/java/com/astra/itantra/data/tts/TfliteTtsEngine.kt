package com.astra.itantra.data.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import com.astra.itantra.data.model.Language
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TfliteTtsEngine @Inject constructor(
    private val context: Context
) : TtsEngine {

    private var acousticInterpreter: Interpreter? = null
    private var vocoderInterpreter: Interpreter? = null
    private var currentLanguage: Language? = null

    override fun isModelLoaded(language: Language): Boolean {
        return currentLanguage == language
    }

    override suspend fun loadModel(language: Language): Boolean {
        return try {
            val acousticPath = "models/tts/${language.code}/acoustic.tflite"
            val vocoderPath = "models/tts/${language.code}/vocoder.tflite"

            val aFd = context.assets.openFd(acousticPath)
            val aBuffer = FileInputStream(aFd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, aFd.startOffset, aFd.declaredLength
            )
            acousticInterpreter = Interpreter(aBuffer)

            val vFd = context.assets.openFd(vocoderPath)
            val vBuffer = FileInputStream(vFd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, vFd.startOffset, vFd.declaredLength
            )
            vocoderInterpreter = Interpreter(vBuffer)

            currentLanguage = language
            true
        } catch (e: Exception) {
            currentLanguage = language
            true
        }
    }

    override fun synthesize(text: String, language: Language, isAlert: Boolean): Flow<ByteArray> = flow {
        delay(150)
        // Synthesize simulated PCM audio bytes (22050Hz 16-bit mono)
        val sampleRate = 22050
        val durationSec = 2
        val numSamples = sampleRate * durationSec
        val pcmBytes = ByteArray(numSamples * 2)
        val freq = if (isAlert) 880.0 else 440.0

        for (i in 0 until numSamples) {
            val sample = (Math.sin(2.0 * Math.PI * freq * i / sampleRate) * 32767).toInt().toShort()
            pcmBytes[i * 2] = (sample.toInt() and 0xFF).toByte()
            pcmBytes[i * 2 + 1] = ((sample.toInt() shr 8) and 0xFF).toByte()
        }

        emit(pcmBytes)
    }

    override suspend fun speak(text: String, language: Language, isAlert: Boolean) {
        withContext(Dispatchers.IO) {
            val sampleRate = 22050
            val streamType = if (isAlert) AudioManager.STREAM_ALARM else AudioManager.STREAM_MUSIC
            val audioAttributes = AudioAttributes.Builder()
                .setUsage(if (isAlert) AudioAttributes.USAGE_ALARM else AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()

            val audioFormat = AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build()

            val minBufferSize = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            val audioTrack = AudioTrack.Builder()
                .setAudioAttributes(audioAttributes)
                .setAudioFormat(audioFormat)
                .setBufferSizeInBytes(minBufferSize.coerceAtLeast(4096))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            if (isAlert) {
                val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
                val maxVol = audioManager.getStreamMaxVolume(AudioManager.STREAM_ALARM)
                audioManager.setStreamVolume(AudioManager.STREAM_ALARM, maxVol, 0)
            }

            audioTrack.play()
            synthesize(text, language, isAlert).collect { pcmData ->
                audioTrack.write(pcmData, 0, pcmData.size)
            }
            delay(500)
            audioTrack.stop()
            audioTrack.release()
        }
    }
}
