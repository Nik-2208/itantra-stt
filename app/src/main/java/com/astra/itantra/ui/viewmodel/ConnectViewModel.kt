package com.astra.itantra.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.astra.itantra.data.model.DiscoveredDevice
import com.astra.itantra.data.repository.CommunicationRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ConnectViewModel @Inject constructor(
    val repository: CommunicationRepository
) : ViewModel() {

    val discoveredDevices = repository.discoveredDevices
    val connectedDevice = repository.connectedDevice

    fun startDiscovery() {
        repository.transportManager.startDiscovery()
    }

    fun connect(device: DiscoveredDevice) {
        viewModelScope.launch {
            repository.transportManager.connect(device)
        }
    }

    fun disconnect() {
        viewModelScope.launch {
            repository.transportManager.disconnect()
        }
    }
}
