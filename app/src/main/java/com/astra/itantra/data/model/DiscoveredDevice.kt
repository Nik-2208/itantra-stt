package com.astra.itantra.data.model

enum class TransportType {
    BLUETOOTH,
    WIFI_DIRECT
}

data class DiscoveredDevice(
    val id: String,
    val name: String,
    val address: String,
    val transportType: TransportType,
    val signalStrength: Int = -50,
    val isConnected: Boolean = false
)
