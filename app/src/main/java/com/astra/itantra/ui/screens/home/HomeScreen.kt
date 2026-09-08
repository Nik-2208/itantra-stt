package com.astra.itantra.ui.screens.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Radio
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.astra.itantra.ui.components.ConnectionStatusPill
import com.astra.itantra.ui.theme.AlertRed
import com.astra.itantra.ui.theme.DeepIndigoPrimary
import com.astra.itantra.ui.theme.WarmSaffronAccent
import com.astra.itantra.ui.viewmodel.HomeViewModel

@Composable
fun HomeScreen(
    viewModel: HomeViewModel,
    onNavigateToTalk: () -> Unit,
    onNavigateToDiscovery: () -> Unit,
    onNavigateToHistory: () -> Unit,
    onNavigateToAlert: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateToDiagnostics: () -> Unit,
    onNavigateToAbout: () -> Unit
) {
    val selectedLang by viewModel.selectedLanguage.collectAsState()
    val connectedDevice by viewModel.connectedDevice.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(text = "iTantra", style = MaterialTheme.typography.headlineLarge, color = DeepIndigoPrimary)
                Text(text = "Language: ${selectedLang.nativeName} (${selectedLang.englishName})", style = MaterialTheme.typography.bodyMedium, color = WarmSaffronAccent)
            }
            ConnectionStatusPill(connectedDevice = connectedDevice, onClick = onNavigateToDiscovery)
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Big Walkie Talkie Main Action Card
        Card(
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = DeepIndigoPrimary),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Radio, contentDescription = "Walkie Talkie", tint = WarmSaffronAccent)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = "Walkie-Talkie Comms", style = MaterialTheme.typography.titleLarge, color = Color.White)
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Push-to-talk live voice streaming with real-time offline translation across 10 Indian languages.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White.copy(alpha = 0.8f)
                )
                Spacer(modifier = Modifier.height(16.dp))
                Button(
                    onClick = onNavigateToTalk,
                    colors = ButtonDefaults.buttonColors(containerColor = WarmSaffronAccent, contentColor = DeepIndigoPrimary),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(text = "Open Walkie-Talkie Screen", style = MaterialTheme.typography.titleLarge)
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Grid Menu Buttons
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MenuCard(
                title = "Peer Discovery",
                icon = Icons.Default.Radio,
                onClick = onNavigateToDiscovery,
                modifier = Modifier.weight(1f)
            )
            MenuCard(
                title = "Transcript History",
                icon = Icons.Default.History,
                onClick = onNavigateToHistory,
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MenuCard(
                title = "Emergency Alert",
                icon = Icons.Default.Warning,
                color = AlertRed,
                onClick = onNavigateToAlert,
                modifier = Modifier.weight(1f)
            )
            MenuCard(
                title = "Diagnostics",
                icon = Icons.Default.Speed,
                onClick = onNavigateToDiagnostics,
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.weight(1f))

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
            OutlinedButton(onClick = onNavigateToSettings) {
                Icon(Icons.Default.Settings, contentDescription = "Settings")
                Spacer(modifier = Modifier.width(4.dp))
                Text("Settings")
            }
            OutlinedButton(onClick = onNavigateToAbout) {
                Icon(Icons.Default.Info, contentDescription = "About")
                Spacer(modifier = Modifier.width(4.dp))
                Text("About")
            }
        }
    }
}

@Composable
private fun MenuCard(
    title: String,
    icon: ImageVector,
    color: Color = DeepIndigoPrimary,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        onClick = onClick,
        modifier = modifier.height(100.dp),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.1f))
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(imageVector = icon, contentDescription = title, tint = color)
            Spacer(modifier = Modifier.height(6.dp))
            Text(text = title, style = MaterialTheme.typography.titleLarge.copy(fontSize = 14.sp), color = color)
        }
    }
}
