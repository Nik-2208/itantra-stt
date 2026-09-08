package com.astra.itantra.data.transport

import android.content.Context
import com.astra.itantra.data.model.DiscoveredDevice
import com.astra.itantra.data.model.TransportType
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class WifiDirectTransport @Inject constructor(
    private val context: Context
) : TransportManager {

    private val _discoveredDevices = MutableStateFlow<List<DiscoveredDevice>>(emptyList())
    override val discoveredDevices: StateFlow<List<DiscoveredDevice>> = _discoveredDevices.asStateFlow()

    private val _connectedDevice = MutableStateFlow<DiscoveredDevice?>(null)
    override val connectedDevice: StateFlow<DiscoveredDevice?> = _connectedDevice.asStateFlow()

    private val _incomingPayloads = MutableSharedFlow<ByteArray>()
    override val incomingPayloads: SharedFlow<ByteArray> = _incomingPayloads.asSharedFlow()

    override fun startDiscovery() {
        _discoveredDevices.value = listOf(
            DiscoveredDevice("wd_01", "Wi-Fi Direct Peer 1", "192.168.49.2", TransportType.WIFI_DIRECT, -40)
        )
    }

    override fun stopDiscovery() {}

    override suspend fun connect(device: DiscoveredDevice): Boolean {
        _connectedDevice.value = device.copy(isConnected = true)
        return true
    }

    override suspend fun disconnect() {
        _connectedDevice.value = null
    }

    override suspend fun sendPayload(payload: ByteArray, isAlert: Boolean): Boolean {
        return _connectedDevice.value?.isConnected == true
    }
}
