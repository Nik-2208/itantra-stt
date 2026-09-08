package com.astra.itantra.data.db

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.astra.itantra.data.model.Language
import com.astra.itantra.data.model.Message

@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey val id: String,
    val senderId: String,
    val senderName: String,
    val audioPath: String?,
    val originalText: String,
    val translatedText: String,
    val sourceLangCode: String,
    val targetLangCode: String,
    val timestamp: Long,
    val isAlert: Boolean,
    val latencyMs: Long
) {
    fun toDomain(): Message {
        return Message(
            id = id,
            senderId = senderId,
            senderName = senderName,
            audioPath = audioPath,
            originalText = originalText,
            translatedText = translatedText,
            sourceLang = Language.fromCode(sourceLangCode),
            targetLang = Language.fromCode(targetLangCode),
            timestamp = timestamp,
            isAlert = isAlert,
            latencyMs = latencyMs
        )
    }

    companion object {
        fun fromDomain(message: Message): MessageEntity {
            return MessageEntity(
                id = message.id,
                senderId = message.senderId,
                senderName = message.senderName,
                audioPath = message.audioPath,
                originalText = message.originalText,
                translatedText = message.translatedText,
                sourceLangCode = message.sourceLang.code,
                targetLangCode = message.targetLang.code,
                timestamp = message.timestamp,
                isAlert = message.isAlert,
                latencyMs = message.latencyMs
            )
        }
    }
}
