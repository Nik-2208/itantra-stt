package com.astra.itantra.data.transport

import com.astra.itantra.data.model.DiscoveredDevice
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

interface TransportManager {
    val discoveredDevices: StateFlow<List<DiscoveredDevice>>
    val connectedDevice: StateFlow<DiscoveredDevice?>
    val incomingPayloads: Flow<ByteArray>

    fun startDiscovery()
    fun stopDiscovery()
    suspend fun connect(device: DiscoveredDevice): Boolean
    suspend fun disconnect()
    suspend fun sendPayload(payload: ByteArray, isAlert: Boolean = false): Boolean
}
