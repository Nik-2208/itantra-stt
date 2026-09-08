package com.astra.itantra.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.astra.itantra.data.model.DiscoveredDevice

@Composable
fun ConnectionStatusPill(
    connectedDevice: DiscoveredDevice?,
    onClick: () -> Unit = {}
) {
    val isConnected = connectedDevice?.isConnected == true
    val statusColor = if (isConnected) Color(0xFF4CAF50) else Color(0xFFFF9800)
    val text = if (isConnected) "Connected to ${connectedDevice?.name}" else "Offline (Mesh Scanning)"

    Box(
        modifier = Modifier
            .background(statusColor.copy(alpha = 0.15f), RoundedCornerShape(20.dp))
            .padding(horizontal = 14.dp, vertical = 6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(statusColor, CircleShape)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = text,
                style = MaterialTheme.typography.labelSmall,
                color = statusColor
            )
        }
    }
}
