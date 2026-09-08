package com.astra.itantra.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.astra.itantra.data.model.Language
import com.astra.itantra.data.repository.CommunicationRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class WalkieTalkieViewModel @Inject constructor(
    val repository: CommunicationRepository
) : ViewModel() {

    val messages = repository.messages
    val selectedLanguage = repository.selectedLanguage
    val targetLanguage = repository.targetLanguage
    val connectedDevice = repository.connectedDevice
    val pttToggleMode = repository.pttToggleMode
    val isRecording = repository.isRecording

    fun setLanguage(language: Language) {
        repository.selectedLanguage.value = language
    }

    fun setTargetLanguage(language: Language) {
        repository.targetLanguage.value = language
    }

    fun setPttToggleMode(enabled: Boolean) {
        repository.pttToggleMode.value = enabled
    }

    fun sendVoiceMessage(audioBytes: ByteArray, isAlert: Boolean = false) {
        viewModelScope.launch {
            repository.sendVoiceMessage(audioBytes, isAlert)
        }
    }
}
