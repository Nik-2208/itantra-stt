package com.astra.itantra.ui.viewmodel

import androidx.lifecycle.ViewModel
import com.astra.itantra.data.model.Language
import com.astra.itantra.data.repository.CommunicationRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    val repository: CommunicationRepository
) : ViewModel() {

    val userName = repository.userName
    val userCallsign = repository.userCallsign
    val selectedLanguage = repository.selectedLanguage
    val targetLanguage = repository.targetLanguage
    val pttToggleMode = repository.pttToggleMode
    val isInitialized = repository.isInitialized

    fun setUserName(name: String) {
        repository.setUserName(name)
    }

    fun setUserCallsign(callsign: String) {
        repository.setUserCallsign(callsign)
    }

    fun setSelectedLanguage(lang: Language) {
        repository.setSelectedLanguage(lang)
    }

    fun setTargetLanguage(lang: Language) {
        repository.setTargetLanguage(lang)
    }

    fun setPttToggle(enabled: Boolean) {
        repository.setPttToggleMode(enabled)
    }

    fun setInitialized(initialized: Boolean) {
        repository.setInitialized(initialized)
    }
}
