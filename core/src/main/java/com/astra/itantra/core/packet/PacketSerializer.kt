package com.astra.itantra.core.packet

import com.astra.itantra.core.enums.LangCode
import com.astra.itantra.core.enums.MsgType
import java.nio.ByteBuffer
import java.nio.ByteOrder

object PacketSerializer {
    const val MIN_PACKET_SIZE = 35

    fun serialize(packet: MessagePacket): ByteArray {
        val textBytes = packet.text.toByteArray(Charsets.UTF_8)
        require(textBytes.size <= 0xFFFF) {
            "Text length exceeds maximum allowed limit of 65535 bytes"
        }

        val sigBytes = packet.signature
        val sigLen = sigBytes?.size ?: 0
        val totalSize = MIN_PACKET_SIZE + textBytes.size + sigLen

        val buffer = ByteBuffer.allocate(totalSize).order(ByteOrder.BIG_ENDIAN)

        buffer.put(packet.senderId)
        buffer.put(packet.msgType.value)
        buffer.put(packet.langCode.value)
        buffer.putInt(packet.sequenceNumber.toInt())
        buffer.putLong(packet.timestamp)
        buffer.putShort(textBytes.size.toShort())
        buffer.put(textBytes)
        buffer.put(packet.hopTTL)
        buffer.putShort(packet.crc16)
        buffer.putFloat(packet.gpsLatitude)
        buffer.putFloat(packet.gpsLongitude)

        if (sigBytes != null && sigBytes.isNotEmpty()) {
            buffer.put(sigBytes)
        }

        return buffer.array()
    }

    fun deserialize(bytes: ByteArray): MessagePacket {
        require(bytes.size >= MIN_PACKET_SIZE) {
            "Packet buffer too small: ${bytes.size} bytes (minimum $MIN_PACKET_SIZE required)"
        }

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)

        val senderId = ByteArray(8)
        buffer.get(senderId)

        val msgType = MsgType.fromByte(buffer.get())
        val langCode = LangCode.fromByte(buffer.get())
        val sequenceNumber = buffer.int.toLong() and 0xFFFFFFFFL
        val timestamp = buffer.long

        val textLen = buffer.short.toInt() and 0xFFFF
        val minRemainingAfterTextLen = textLen + 11 // hopTTL(1) + crc16(2) + gps_lat(4) + gps_lon(4)

        if (buffer.remaining() < minRemainingAfterTextLen) {
            throw IllegalArgumentException(
                "Malformed packet: expected at least $minRemainingAfterTextLen bytes after text_len, but only ${buffer.remaining()} bytes available"
            )
        }

        val textBytes = ByteArray(textLen)
        buffer.get(textBytes)
        val text = String(textBytes, Charsets.UTF_8)

        val hopTTL = buffer.get()
        val crc16 = buffer.short
        val gpsLatitude = buffer.float
        val gpsLongitude = buffer.float

        val sigLen = buffer.remaining()
        val signature = if (sigLen > 0) {
            val sig = ByteArray(sigLen)
            buffer.get(sig)
            sig
        } else {
            null
        }

        return MessagePacket(
            senderId = senderId,
            msgType = msgType,
            langCode = langCode,
            sequenceNumber = sequenceNumber,
            timestamp = timestamp,
            text = text,
            hopTTL = hopTTL,
            crc16 = crc16,
            gpsLatitude = gpsLatitude,
            gpsLongitude = gpsLongitude,
            signature = signature
        )
    }
}

