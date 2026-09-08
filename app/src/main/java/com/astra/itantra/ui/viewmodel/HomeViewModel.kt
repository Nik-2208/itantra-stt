package com.astra.itantra.ui.viewmodel

import androidx.lifecycle.ViewModel
import com.astra.itantra.data.repository.CommunicationRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class HomeViewModel @Inject constructor(
    val repository: CommunicationRepository
) : ViewModel() {
    val selectedLanguage = repository.selectedLanguage
    val connectedDevice = repository.connectedDevice
}
