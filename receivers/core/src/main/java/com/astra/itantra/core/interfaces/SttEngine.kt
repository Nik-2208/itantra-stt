package com.astra.itantra.core.interfaces

interface SttEngine {
    fun transcribe(
        audio: ShortArray
    ): String
}

