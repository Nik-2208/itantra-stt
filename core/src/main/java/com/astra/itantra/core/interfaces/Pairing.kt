package com.astra.itantra.core.interfaces

interface Pairing {
    fun generateIdentity()

    fun getPublicKey(): ByteArray

    fun verifyPeer(
        key: ByteArray
    ): Boolean
}

