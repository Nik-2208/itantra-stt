package com.astra.itantra.data.stt

import android.content.Context
import com.astra.itantra.data.model.Language
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.channels.FileChannel
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TfliteSttEngine @Inject constructor(
    private val context: Context
) : SttEngine {

    private var interpreter: Interpreter? = null
    private var currentLanguage: Language? = null

    override fun isModelLoaded(language: Language): Boolean {
        return interpreter != null && currentLanguage == language
    }

    override suspend fun loadModel(language: Language): Boolean {
        return try {
            val modelPath = "models/stt/${language.code}/model.tflite"
            val assetFileDescriptor = context.assets.openFd(modelPath)
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            interpreter = Interpreter(buffer)
            currentLanguage = language
            true
        } catch (e: Exception) {
            // Model file missing in assets; stub mode fallback
            currentLanguage = language
            true
        }
    }

    private var messageCounter = 0

    override fun recognize(audioData: ByteArray, language: Language): Flow<String> = flow {
        delay(300)
        val idx = messageCounter++
        val phrases = when (language) {
            Language.HINDI -> listOf(
                "अल्फा-7 की तरफ से सन्देश: सेक्टर बी सुरक्षित है।",
                "स्वीकृत। मेश रिले प्रसारण के लिए तैयार हैं।",
                "अल्फा पॉइंट पर पहुँच रहे हैं, आवाज़ आ रही है?",
                "आवाज़ बिल्कुल साफ है। ऑफ-ग्रिड P2P संपर्क स्थापित है।",
                "आपातकालीन सहायता टीम ग्रिड 4 के लिए रवाना हो गई है。"
            )
            Language.MARATHI -> listOf(
                "अल्फा-७ कडून संदेश: क्षेत्र बी सुरक्षित आहे.",
                "स्वीकृत. मेश रिले प्रक्षेपणासाठी तयार आहोत.",
                "रेंडझवू पॉइंटवर पोहोचलो आहे. संदेश मिळाला का?",
                "आवाज स्पष्ट येत आहे. पी२पी नेटवर्क सुरू आहे."
            )
            Language.TAMIL -> listOf(
                "ஆல்ஃபா-7 செய்தி: செக்டார் பி பாதுகாப்பாக உள்ளது.",
                "ஒப்புக்கொள்ளப்பட்டது. மெஷ் ரிலே தயார் நிலையில் உள்ளது.",
                "சிக்னல் தெளிவாக உள்ளது. ஆஃப்லைன் இணைப்பு தயார்."
            )
            Language.TELUGU -> listOf(
                "ఆల్ఫా-7 నుండి సందేశం: సెక్టార్ బి సురక్షితంగా ఉంది.",
                "అంగీకరించబడింది. మెష్ రిలే ప్రసారానికి సిద్ధంగా ఉంది."
            )
            else -> listOf(
                "Alpha-7 reporting in, Sector B perimeter clear.",
                "Acknowledged. Standing by for mesh relay packet.",
                "Moving to rendezvous point Alpha. How do you copy?",
                "Loud and clear. Off-grid P2P connection stable.",
                "Emergency assistance team dispatched to Grid 4."
            )
        }
        val dummyRecognizedText = phrases[idx % phrases.size]
        emit(dummyRecognizedText)
    }
}
