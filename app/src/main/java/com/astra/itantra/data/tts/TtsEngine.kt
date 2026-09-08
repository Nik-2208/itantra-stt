package com.astra.itantra.data.tts

import com.astra.itantra.data.model.Language
import kotlinx.coroutines.flow.Flow

interface TtsEngine {
    fun isModelLoaded(language: Language): Boolean
    suspend fun loadModel(language: Language): Boolean
    fun synthesize(text: String, language: Language, isAlert: Boolean = false): Flow<ByteArray>
    suspend fun speak(text: String, language: Language, isAlert: Boolean = false)
}
