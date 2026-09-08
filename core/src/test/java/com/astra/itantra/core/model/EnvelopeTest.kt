package com.astra.itantra.core.model

import com.astra.itantra.core.enums.LangCode
import com.astra.itantra.core.enums.MsgType
import com.astra.itantra.core.interfaces.SttEngine
import com.astra.itantra.core.interfaces.TranslationEngine
import com.astra.itantra.core.interfaces.TtsEngine
import com.astra.itantra.core.packet.CRC16
import com.astra.itantra.core.packet.PacketSerializer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class EnvelopeTest {

    private val sampleSenderId = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8)

    // Mock engines for testing
    private val mockSttEngine = object : SttEngine {
        override fun transcribe(audio: ShortArray): String {
            return "Transcribed Text from Audio"
        }
    }

    private val mockTranslationEngine = object : TranslationEngine {
        override fun translate(text: String, from: LangCode, to: LangCode): String {
            return "Translated ($from -> $to): $text"
        }
    }

    private val mockTtsEngine = object : TtsEngine {
        override fun synthesize(text: String, language: LangCode): ShortArray {
            return shortArrayOf(100, 200, 300)
        }
    }

    @Test
    fun testSuccessfulPipelineChaining() {
        val initialAudio = shortArrayOf(10, 20, 30, 40)
        val initialEnvelope = Envelope(
            senderId = sampleSenderId,
            msgType = MsgType.ALERT,
            sourceLang = LangCode.HINDI,
            targetLang = LangCode.ENGLISH,
            rawAudio = initialAudio,
            gpsLatitude = 18.52f,
            gpsLongitude = 73.85f,
            timestamp = 1000L
        )

        // Step 1: STT
        val envAfterStt = initialEnvelope.transcribeWith(mockSttEngine)
        assertEquals("Transcribed Text from Audio", envAfterStt.transcribedText)

        // Step 2: Translation
        val envAfterTranslation = envAfterStt.translateWith(mockTranslationEngine)
        assertEquals(
            "Translated (HINDI -> ENGLISH): Transcribed Text from Audio",
            envAfterTranslation.translatedText
        )

        // Step 3: MessagePacket Conversion
        val messagePacket = envAfterTranslation.toMessagePacket(
            sequenceNumber = 42L,
            hopTTL = 5.toByte(),
            signature = byteArrayOf(99)
        )

        assertEquals(MsgType.ALERT, messagePacket.msgType)
        assertEquals(LangCode.ENGLISH, messagePacket.langCode)
        assertEquals(
            "Translated (HINDI -> ENGLISH): Transcribed Text from Audio",
            messagePacket.text
        )
        assertEquals(42L, messagePacket.sequenceNumber)
        assertEquals(5.toByte(), messagePacket.hopTTL)
        assertEquals(18.52f, messagePacket.gpsLatitude)
        assertEquals(73.85f, messagePacket.gpsLongitude)

        // Verify CRC computation
        val serializedForCrc = PacketSerializer.serialize(messagePacket.copy(crc16 = 0))
        assertTrue(CRC16.verify(serializedForCrc, messagePacket.crc16))
    }

    @Test
    fun testImmutabilityAndMetadataPreservation() {
        val original = Envelope(
            senderId = sampleSenderId,
            msgType = MsgType.SOS,
            sourceLang = LangCode.MARATHI,
            targetLang = LangCode.GUJARATI,
            rawAudio = shortArrayOf(1, 2, 3),
            gpsLatitude = 19.07f,
            gpsLongitude = 72.87f,
            timestamp = 5000L
        )

        val next = original.transcribeWith(mockSttEngine)

        // Ensure original is untouched
        assertNull(original.transcribedText)
        assertNotSame(original, next)

        // Ensure metadata is preserved across steps
        assertEquals(original.senderId, next.senderId)
        assertEquals(original.msgType, next.msgType)
        assertEquals(original.sourceLang, next.sourceLang)
        assertEquals(original.targetLang, next.targetLang)
        assertEquals(original.gpsLatitude, next.gpsLatitude, 0.001f)
        assertEquals(original.gpsLongitude, next.gpsLongitude, 0.001f)
        assertEquals(original.timestamp, next.timestamp)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testTranscribeWithMissingAudioThrowsException() {
        val envWithoutAudio = Envelope(
            senderId = sampleSenderId,
            sourceLang = LangCode.HINDI,
            targetLang = LangCode.ENGLISH,
            rawAudio = null
        )
        envWithoutAudio.transcribeWith(mockSttEngine)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testTranslateWithMissingTranscribedTextThrowsException() {
        val envWithoutText = Envelope(
            senderId = sampleSenderId,
            sourceLang = LangCode.HINDI,
            targetLang = LangCode.ENGLISH,
            transcribedText = null
        )
        envWithoutText.translateWith(mockTranslationEngine)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testToMessagePacketMissingTextThrowsException() {
        val envWithoutText = Envelope(
            senderId = sampleSenderId,
            sourceLang = LangCode.HINDI,
            targetLang = LangCode.ENGLISH
        )
        envWithoutText.toMessagePacket(1L, 1.toByte())
    }

    @Test
    fun testSynthesizeWithTts() {
        val env = Envelope(
            senderId = sampleSenderId,
            sourceLang = LangCode.ENGLISH,
            targetLang = LangCode.HINDI,
            transcribedText = "Hello World"
        )

        val envWithSynthesized = env.synthesizeWith(mockTtsEngine)
        assertArrayEquals(shortArrayOf(100, 200, 300), envWithSynthesized.synthesizedAudio)
    }

    @Test
    fun testMessagePacketToEnvelopeConversion() {
        val packet = Envelope(
            senderId = sampleSenderId,
            msgType = MsgType.ACK,
            sourceLang = LangCode.TAMIL,
            targetLang = LangCode.TELUGU,
            transcribedText = "Incoming message text",
            gpsLatitude = 12.97f,
            gpsLongitude = 77.59f,
            timestamp = 9999L
        ).toMessagePacket(sequenceNumber = 10L, hopTTL = 2.toByte())

        val receivedEnv = packet.toEnvelope(sourceLang = LangCode.TAMIL)

        assertArrayEquals(sampleSenderId, receivedEnv.senderId)
        assertEquals(MsgType.ACK, receivedEnv.msgType)
        assertEquals(LangCode.TAMIL, receivedEnv.sourceLang)
        assertEquals(LangCode.TELUGU, receivedEnv.targetLang)
        assertEquals("Incoming message text", receivedEnv.translatedText)
        assertEquals(12.97f, receivedEnv.gpsLatitude, 0.001f)
        assertEquals(77.59f, receivedEnv.gpsLongitude, 0.001f)
        assertEquals(9999L, receivedEnv.timestamp)
    }
}
