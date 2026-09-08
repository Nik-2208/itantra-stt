package com.astra.itantra.core.enums

enum class MsgType(val value: Byte) {
    NORMAL(0),
    ALERT(1),
    ACK(2),
    SOS(3);

    companion object {
        fun fromByte(value: Byte): MsgType {
            return entries.firstOrNull { it.value == value }
                ?: throw IllegalArgumentException("Unknown MsgType byte: $value")
        }
    }
}

