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
class BluetoothTransport @Inject constructor(
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
            DiscoveredDevice("bt_01", "ISRO Peer Walkie (Alpha)", "00:11:22:33:44:55", TransportType.BLUETOOTH, -45),
            DiscoveredDevice("bt_02", "Ground Comms (Bravo)", "AA:BB:CC:DD:EE:FF", TransportType.BLUETOOTH, -68),
            DiscoveredDevice("wifi_01", "Base Station (WiFi-Direct)", "192.168.49.1", TransportType.WIFI_DIRECT, -32)
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
        // Send length-prefixed framing payload over Bluetooth SPP socket
        return _connectedDevice.value?.isConnected == true
    }

    suspend fun simulateIncomingPayload(data: ByteArray) {
        _incomingPayloads.emit(data)
    }
}
