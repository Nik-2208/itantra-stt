package com.astra.itantra.core.packet

import com.astra.itantra.core.enums.LangCode
import com.astra.itantra.core.enums.MsgType

data class MessagePacket(
    val senderId: ByteArray,
    val msgType: MsgType,
    val langCode: LangCode,
    val sequenceNumber: Long,
    val timestamp: Long,
    val text: String,
    val hopTTL: Byte,
    val crc16: Short,
    val gpsLatitude: Float,
    val gpsLongitude: Float,
    val signature: ByteArray? = null
) {
    init {
        require(senderId.size == 8) { "senderId must be exactly 8 bytes long (got ${senderId.size})" }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as MessagePacket

        if (!senderId.contentEquals(other.senderId)) return false
        if (msgType != other.msgType) return false
        if (langCode != other.langCode) return false
        if (sequenceNumber != other.sequenceNumber) return false
        if (timestamp != other.timestamp) return false
        if (text != other.text) return false
        if (hopTTL != other.hopTTL) return false
        if (crc16 != other.crc16) return false
        if (gpsLatitude != other.gpsLatitude) return false
        if (gpsLongitude != other.gpsLongitude) return false
        if (signature != null) {
            if (other.signature == null) return false
            if (!signature.contentEquals(other.signature)) return false
        } else if (other.signature != null) return false

        return true
    }

    override fun hashCode(): Int {
        var result = senderId.contentHashCode()
        result = 31 * result + msgType.hashCode()
        result = 31 * result + langCode.hashCode()
        result = 31 * result + sequenceNumber.hashCode()
        result = 31 * result + timestamp.hashCode()
        result = 31 * result + text.hashCode()
        result = 31 * result + hopTTL.hashCode()
        result = 31 * result + crc16.hashCode()
        result = 31 * result + gpsLatitude.hashCode()
        result = 31 * result + gpsLongitude.hashCode()
        result = 31 * result + (signature?.contentHashCode() ?: 0)
        return result
    }
}

