package com.astra.itantra.ui.screens.onboarding

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bluetooth
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.astra.itantra.data.transport.CommsForegroundService
import com.astra.itantra.ui.theme.DeepIndigoPrimary
import com.astra.itantra.ui.theme.WarmSaffronAccent

@Composable
fun PermissionsScreen(
    onPermissionsGranted: () -> Unit
) {
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
    ) {
        Text(
            text = "Required Permissions",
            style = MaterialTheme.typography.headlineMedium,
            color = DeepIndigoPrimary
        )
        Text(
            text = "iTantra operates 100% offline without internet. Mic and local mesh networking permissions are required.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
        )

        Spacer(modifier = Modifier.height(24.dp))

        PermissionItem(
            icon = Icons.Default.Mic,
            title = "Microphone Access",
            description = "To record voice audio for real-time speech-to-text transcription."
        )

        Spacer(modifier = Modifier.height(12.dp))

        PermissionItem(
            icon = Icons.Default.Bluetooth,
            title = "Bluetooth Scan & Connect",
            description = "To discover nearby peer devices for offline walkie-talkie audio streaming."
        )

        Spacer(modifier = Modifier.height(12.dp))

        PermissionItem(
            icon = Icons.Default.Wifi,
            title = "Wi-Fi Direct Peer Mesh",
            description = "To maintain high-bandwidth peer-to-peer audio channels."
        )

        Spacer(modifier = Modifier.weight(1f))

        Button(
            onClick = {
                CommsForegroundService.startService(context)
                onPermissionsGranted()
            },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = DeepIndigoPrimary, contentColor = WarmSaffronAccent)
        ) {
            Text(text = "Grant & Start iTantra", style = MaterialTheme.typography.titleLarge)
        }
    }
}

@Composable
private fun PermissionItem(
    icon: ImageVector,
    title: String,
    description: String
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = title,
                tint = DeepIndigoPrimary,
                modifier = Modifier.padding(8.dp)
            )
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                Text(text = title, style = MaterialTheme.typography.titleLarge)
                Text(text = description, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
            }
        }
    }
}
