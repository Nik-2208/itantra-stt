package com.astra.itantra.core.packet

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CRC16Test {

    @Test
    fun testCRC16CalculationAndVerification() {
        val data = "Hello iTantra Core".toByteArray(Charsets.UTF_8)
        val crc = CRC16.calculate(data)

        assertTrue(CRC16.verify(data, crc))
    }

    @Test
    fun testCRC16DetectsCorruptedData() {
        val data = "Hello iTantra Core".toByteArray(Charsets.UTF_8)
        val crc = CRC16.calculate(data)

        val corruptedData = data.copyOf()
        corruptedData[0] = (corruptedData[0].toInt() xor 0xFF).toByte()

        assertFalse(CRC16.verify(corruptedData, crc))
    }

    @Test
    fun testCRC16DeterministicValue() {
        val data1 = "123456789".toByteArray(Charsets.UTF_8)
        val crc1 = CRC16.calculate(data1)
        val crc2 = CRC16.calculate(data1)

        assertEquals(crc1, crc2)
    }
}

