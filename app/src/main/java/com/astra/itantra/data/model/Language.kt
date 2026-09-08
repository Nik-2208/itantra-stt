package com.astra.itantra.data.model

enum class Language(
    val code: String,
    val nativeName: String,
    val englishName: String,
    val bcp47: String
) {
    HINDI("hi", "हिन्दी", "Hindi", "hi-IN"),
    BENGALI("bn", "বাংলা", "Bengali", "bn-IN"),
    TAMIL("ta", "தமிழ்", "Tamil", "ta-IN"),
    TELUGU("te", "తెలుగు", "Telugu", "te-IN"),
    MARATHI("mr", "मराठी", "Marathi", "mr-IN"),
    GUJARATI("gu", "ગુજરાતી", "Gujarati", "gu-IN"),
    KANNADA("kn", "कन्नड", "Kannada", "kn-IN"),
    MALAYALAM("ml", "മലയാളം", "Malayalam", "ml-IN"),
    PUNJABI("pa", "ਪੰਜਾਬੀ", "Punjabi", "pa-IN"),
    ODIA("or", "ଓଡ଼ିଆ", "Odia", "or-IN"),
    ENGLISH("en", "English", "English", "en-US");

    companion object {
        fun fromCode(code: String): Language {
            return entries.firstOrNull { it.code.equals(code, ignoreCase = true) } ?: ENGLISH
        }
    }
}
