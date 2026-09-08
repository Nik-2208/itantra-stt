package com.astra.itantra.core.packet

object CRC16 {
    private const val POLYNOMIAL = 0x1021
    private const val INITIAL_VALUE = 0xFFFF

    fun calculate(data: ByteArray): Short {
        return calculate(data, 0, data.size)
    }

    fun calculate(data: ByteArray, offset: Int, length: Int): Short {
        require(offset >= 0 && length >= 0 && offset + length <= data.size) {
            "Invalid offset ($offset) or length ($length) for ByteArray of size ${data.size}"
        }
        var crc = INITIAL_VALUE
        for (i in offset until (offset + length)) {
            val b = data[i]
            crc = crc xor ((b.toInt() and 0xFF) shl 8)
            for (j in 0 until 8) {
                crc = if ((crc and 0x8000) != 0) {
                    (crc shl 1) xor POLYNOMIAL
                } else {
                    crc shl 1
                }
            }
        }
        return (crc and 0xFFFF).toShort()
    }

    fun verify(data: ByteArray, checksum: Short): Boolean {
        return calculate(data) == checksum
    }

    fun verify(data: ByteArray, offset: Int, length: Int, checksum: Short): Boolean {
        return calculate(data, offset, length) == checksum
    }
}

