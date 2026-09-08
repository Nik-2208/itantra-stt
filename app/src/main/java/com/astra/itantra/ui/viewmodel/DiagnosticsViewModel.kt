package com.astra.itantra.ui.viewmodel

import androidx.lifecycle.ViewModel
import com.astra.itantra.data.repository.CommunicationRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class DiagnosticsViewModel @Inject constructor(
    val repository: CommunicationRepository
) : ViewModel() {
    val diagnostics = repository.diagnostics
}
