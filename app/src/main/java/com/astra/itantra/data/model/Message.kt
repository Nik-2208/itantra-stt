package com.astra.itantra.data.model

import java.util.UUID

data class Message(
    val id: String = UUID.randomUUID().toString(),
    val senderId: String,
    val senderName: String,
    val audioPath: String? = null,
    val originalText: String,
    val translatedText: String,
    val sourceLang: Language,
    val targetLang: Language,
    val timestamp: Long = System.currentTimeMillis(),
    val isAlert: Boolean = false,
    val latencyMs: Long = 0L
)
