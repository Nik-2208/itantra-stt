package com.astra.itantra.core.packet

import com.astra.itantra.core.enums.LangCode
import com.astra.itantra.core.enums.MsgType
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PacketSerializerTest {

    @Test
    fun testSerializationRoundtripWithoutSignature() {
        val senderId = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8)
        val text = "नमस्ते (Namaste)"
        val packet = MessagePacket(
            senderId = senderId,
            msgType = MsgType.ALERT,
            langCode = LangCode.HINDI,
            sequenceNumber = 42L,
            timestamp = 1725800000000L,
            text = text,
            hopTTL = 5.toByte(),
            crc16 = 0x1234.toShort(),
            gpsLatitude = 18.5204f,
            gpsLongitude = 73.8567f,
            signature = null
        )

        val serialized = PacketSerializer.serialize(packet)
        val deserialized = PacketSerializer.deserialize(serialized)

        assertEquals(packet, deserialized)
        assertNull(deserialized.signature)
    }

    @Test
    fun testSerializationRoundtripWithSignature() {
        val senderId = byteArrayOf(8, 7, 6, 5, 4, 3, 2, 1)
        val signature = byteArrayOf(10, 20, 30, 40, 50, 60)
        val packet = MessagePacket(
            senderId = senderId,
            msgType = MsgType.SOS,
            langCode = LangCode.MARATHI,
            sequenceNumber = 999999L,
            timestamp = 1725811111111L,
            text = "Emergency Alert",
            hopTTL = 3.toByte(),
            crc16 = 0x5678.toShort(),
            gpsLatitude = 19.0760f,
            gpsLongitude = 72.8777f,
            signature = signature
        )

        val serialized = PacketSerializer.serialize(packet)
        val deserialized = PacketSerializer.deserialize(serialized)

        assertEquals(packet, deserialized)
        assertArrayEquals(signature, deserialized.signature)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testRejectTooSmallPacketBuffer() {
        val tooSmallBytes = ByteArray(10)
        PacketSerializer.deserialize(tooSmallBytes)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testRejectInvalidSenderIdSize() {
        MessagePacket(
            senderId = byteArrayOf(1, 2, 3), // invalid size, must be 8
            msgType = MsgType.NORMAL,
            langCode = LangCode.ENGLISH,
            sequenceNumber = 1L,
            timestamp = 100L,
            text = "Test",
            hopTTL = 1.toByte(),
            crc16 = 0.toShort(),
            gpsLatitude = 0.0f,
            gpsLongitude = 0.0f
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun testRejectUnknownMsgTypeByte() {
        val validPacket = MessagePacket(
            senderId = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8),
            msgType = MsgType.NORMAL,
            langCode = LangCode.ENGLISH,
            sequenceNumber = 1L,
            timestamp = 100L,
            text = "Hi",
            hopTTL = 1.toByte(),
            crc16 = 0.toShort(),
            gpsLatitude = 0.0f,
            gpsLongitude = 0.0f
        )
        val serialized = PacketSerializer.serialize(validPacket)
        // Corrupt msgType byte (offset 8) to an invalid value 99
        serialized[8] = 99.toByte()

        PacketSerializer.deserialize(serialized)
    }
}

