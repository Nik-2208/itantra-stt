package com.astra.itantra.core.interfaces

import com.astra.itantra.core.enums.LangCode

interface TtsEngine {
    fun synthesize(
        text: String,
        language: LangCode
    ): ShortArray
}

