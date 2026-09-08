package com.astra.itantra.core.enums

enum class LangCode(val value: Byte) {
    HINDI(0),
    GUJARATI(1),
    MARATHI(2),
    KANNADA(3),
    MALAYALAM(4),
    TAMIL(5),
    TELUGU(6),
    ODIA(7),
    BENGALI(8),
    ENGLISH(9),
    OTHER(10);

    companion object {
        fun fromByte(value: Byte): LangCode {
            return entries.firstOrNull { it.value == value }
                ?: throw IllegalArgumentException("Unknown LangCode byte: $value")
        }
    }
}

