package com.astra.itantra.data.stt

import com.astra.itantra.data.model.Language
import kotlinx.coroutines.flow.Flow

interface SttEngine {
    fun isModelLoaded(language: Language): Boolean
    suspend fun loadModel(language: Language): Boolean
    fun recognize(audioData: ByteArray, language: Language): Flow<String>
}
