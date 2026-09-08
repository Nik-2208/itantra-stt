package com.astra.itantra.core.interfaces

import com.astra.itantra.core.packet.MessagePacket

interface Transport {
    fun connect(
        deviceId: String
    )

    fun disconnect()

    fun send(
        packet: MessagePacket
    )

    fun setReceiver(
        callback: (MessagePacket) -> Unit
    )
}

