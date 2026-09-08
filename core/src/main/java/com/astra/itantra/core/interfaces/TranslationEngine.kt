package com.astra.itantra.core.interfaces

import com.astra.itantra.core.enums.LangCode

interface TranslationEngine {
    fun translate(
        text: String,
        from: LangCode,
        to: LangCode
    ): String
}

