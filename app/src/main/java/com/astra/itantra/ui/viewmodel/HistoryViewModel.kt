package com.astra.itantra.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.astra.itantra.data.repository.CommunicationRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class HistoryViewModel @Inject constructor(
    val repository: CommunicationRepository
) : ViewModel() {

    val messages = repository.messages

    fun clearHistory() {
        viewModelScope.launch {
            repository.clearAllHistory()
        }
    }
}
