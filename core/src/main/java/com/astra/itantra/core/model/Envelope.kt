package com.astra.itantra.core.model

import com.astra.itantra.core.enums.LangCode
import com.astra.itantra.core.enums.MsgType
import com.astra.itantra.core.interfaces.SttEngine
import com.astra.itantra.core.interfaces.TranslationEngine
import com.astra.itantra.core.interfaces.TtsEngine
import com.astra.itantra.core.packet.CRC16
import com.astra.itantra.core.packet.MessagePacket
import com.astra.itantra.core.packet.PacketSerializer

data class Envelope(
    val senderId: ByteArray,
    val msgType: MsgType = MsgType.NORMAL,
    val sourceLang: LangCode,
    val targetLang: LangCode,
    val rawAudio: ShortArray? = null,
    val transcribedText: String? = null,
    val translatedText: String? = null,
    val synthesizedAudio: ShortArray? = null,
    val gpsLatitude: Float = 0f,
    val gpsLongitude: Float = 0f,
    val timestamp: Long = System.currentTimeMillis()
) {
    init {
        require(senderId.size == 8) {
            "senderId must be exactly 8 bytes long (got ${senderId.size})"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Envelope

        if (!senderId.contentEquals(other.senderId)) return false
        if (msgType != other.msgType) return false
        if (sourceLang != other.sourceLang) return false
        if (targetLang != other.targetLang) return false

        if (rawAudio != null) {
            if (other.rawAudio == null) return false
            if (!rawAudio.contentEquals(other.rawAudio)) return false
        } else if (other.rawAudio != null) return false

        if (transcribedText != other.transcribedText) return false
        if (translatedText != other.translatedText) return false

        if (synthesizedAudio != null) {
            if (other.synthesizedAudio == null) return false
            if (!synthesizedAudio.contentEquals(other.synthesizedAudio)) return false
        } else if (other.synthesizedAudio != null) return false

        if (gpsLatitude != other.gpsLatitude) return false
        if (gpsLongitude != other.gpsLongitude) return false
        if (timestamp != other.timestamp) return false

        return true
    }

    override fun hashCode(): Int {
        var result = senderId.contentHashCode()
        result = 31 * result + msgType.hashCode()
        result = 31 * result + sourceLang.hashCode()
        result = 31 * result + targetLang.hashCode()
        result = 31 * result + (rawAudio?.contentHashCode() ?: 0)
        result = 31 * result + (transcribedText?.hashCode() ?: 0)
        result = 31 * result + (translatedText?.hashCode() ?: 0)
        result = 31 * result + (synthesizedAudio?.contentHashCode() ?: 0)
        result = 31 * result + gpsLatitude.hashCode()
        result = 31 * result + gpsLongitude.hashCode()
        result = 31 * result + timestamp.hashCode()
        return result
    }
}

fun Envelope.transcribeWith(stt: SttEngine): Envelope {
    val audio = requireNotNull(rawAudio) {
        "rawAudio must not be null when calling transcribeWith"
    }
    val text = stt.transcribe(audio)
    return copy(transcribedText = text)
}

fun Envelope.translateWith(translator: TranslationEngine): Envelope {
    val text = requireNotNull(transcribedText) {
        "transcribedText must not be null when calling translateWith"
    }
    val translated = translator.translate(text, from = sourceLang, to = targetLang)
    return copy(translatedText = translated)
}

fun Envelope.synthesizeWith(tts: TtsEngine): Envelope {
    val text = requireNotNull(translatedText ?: transcribedText) {
        "Neither translatedText nor transcribedText is available for synthesizeWith"
    }
    val pcm = tts.synthesize(text, language = targetLang)
    return copy(synthesizedAudio = pcm)
}

fun Envelope.toMessagePacket(
    sequenceNumber: Long,
    hopTTL: Byte,
    signature: ByteArray? = null
): MessagePacket {
    val textToSend = requireNotNull(translatedText ?: transcribedText) {
        "Envelope has no text (transcribedText or translatedText) to construct MessagePacket"
    }

    val packetWithZeroCrc = MessagePacket(
        senderId = senderId,
        msgType = msgType,
        langCode = targetLang,
        sequenceNumber = sequenceNumber,
        timestamp = timestamp,
        text = textToSend,
        hopTTL = hopTTL,
        crc16 = 0,
        gpsLatitude = gpsLatitude,
        gpsLongitude = gpsLongitude,
        signature = signature
    )

    val serializedBytes = PacketSerializer.serialize(packetWithZeroCrc)
    val calculatedCrc = CRC16.calculate(serializedBytes)

    return packetWithZeroCrc.copy(crc16 = calculatedCrc)
}

fun MessagePacket.toEnvelope(sourceLang: LangCode): Envelope {
    return Envelope(
        senderId = senderId,
        msgType = msgType,
        sourceLang = sourceLang,
        targetLang = langCode,
        translatedText = text,
        gpsLatitude = gpsLatitude,
        gpsLongitude = gpsLongitude,
        timestamp = timestamp
    )
}

