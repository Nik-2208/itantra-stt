package com.astra.itantra.ui.screens.connect

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CellTower
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.PhoneAndroid
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.astra.itantra.data.model.DiscoveredDevice
import com.astra.itantra.ui.components.TacticalBottomBar
import com.astra.itantra.ui.theme.ActiveGreen
import com.astra.itantra.ui.theme.EmergencyCardBg
import com.astra.itantra.ui.theme.EmergencyRed
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalBackground
import com.astra.itantra.ui.theme.TacticalCardBorder
import com.astra.itantra.ui.theme.TacticalSurface
import com.astra.itantra.ui.theme.TextGray
import com.astra.itantra.ui.theme.TextWhite
import com.astra.itantra.ui.viewmodel.ConnectViewModel

@Composable
fun DeviceDiscoveryScreen(
    viewModel: ConnectViewModel,
    onNavigateToComms: () -> Unit,
    onNavigateToHistory: () -> Unit,
    onNavigateToStats: () -> Unit,
    onNavigateToAlert: () -> Unit,
    onNavigateToSettings: () -> Unit
) {
    val discoveredDevices by viewModel.discoveredDevices.collectAsState()
    val connectedDevice by viewModel.connectedDevice.collectAsState()

    Scaffold(
        bottomBar = {
            TacticalBottomBar(
                currentRoute = "device_discovery",
                onNavigateToDiscovery = {},
                onNavigateToComms = onNavigateToComms,
                onNavigateToHistory = onNavigateToHistory,
                onNavigateToStats = onNavigateToStats,
                onCenterMicClick = onNavigateToComms
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(TacticalBackground)
                .padding(padding)
                .padding(horizontal = 20.dp, vertical = 12.dp)
        ) {
            // Top Status Bar (Figma Page 1)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(ActiveGreen, CircleShape)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "MESH ACTIVE",
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, color = ActiveGreen)
                    )
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(text = "LAT: 12ms   CPU: 8%", style = MaterialTheme.typography.labelSmall, color = TextGray)
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Main Header & Action Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "ITANTRA",
                    style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Black, letterSpacing = 1.sp),
                    color = TextWhite
                )

                Row {
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .background(TacticalSurface, RoundedCornerShape(8.dp))
                            .border(1.dp, TacticalCardBorder, RoundedCornerShape(8.dp)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.Default.CellTower, contentDescription = "Antenna", tint = NeonOrangePrimary, modifier = Modifier.size(20.dp))
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .background(TacticalSurface, RoundedCornerShape(8.dp))
                            .border(1.dp, TacticalCardBorder, RoundedCornerShape(8.dp))
                            .clickable { onNavigateToSettings() },
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings", tint = TextWhite, modifier = Modifier.size(20.dp))
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Scanning Mesh Card
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = TacticalSurface),
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, TacticalCardBorder, RoundedCornerShape(12.dp))
            ) {
                Row(
                    modifier = Modifier.padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(NeonOrangePrimary, CircleShape)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(text = "SCANNING MESH", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, color = TextGray))
                        Text(text = "Searching for local nodes...", style = MaterialTheme.typography.titleLarge.copy(fontSize = 14.sp), color = TextWhite)
                    }
                    Box(
                        modifier = Modifier
                            .size(24.dp)
                            .border(2.dp, NeonOrangePrimary, CircleShape)
                    )
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            Text(
                text = "AVAILABLE NODES",
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 1.sp),
                color = TextGray
            )

            Spacer(modifier = Modifier.height(10.dp))

            // Nodes List
            LazyColumn(modifier = Modifier.weight(1f)) {
                items(discoveredDevices) { device ->
                    NodeCard(
                        device = device,
                        isConnected = connectedDevice?.id == device.id,
                        onClick = {
                            viewModel.connect(device)
                            onNavigateToComms()
                        }
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Emergency Broadcast Red Button (Figma Page 1)
            Card(
                onClick = onNavigateToAlert,
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = EmergencyCardBg),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp)
                    .border(1.dp, EmergencyRed, RoundedCornerShape(12.dp))
            ) {
                Row(
                    modifier = Modifier.fillMaxSize(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.Warning, contentDescription = "Alert", tint = EmergencyRed, modifier = Modifier.size(20.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "EMERGENCY BROADCAST",
                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold, fontSize = 14.sp),
                        color = EmergencyRed
                    )
                }
            }
        }
    }
}

@Composable
private fun NodeCard(
    device: DiscoveredDevice,
    isConnected: Boolean,
    onClick: () -> Unit
) {
    Card(
        onClick = onClick,
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = TacticalSurface),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, if (isConnected) NeonOrangePrimary else TacticalCardBorder, RoundedCornerShape(12.dp))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .background(Color(0xFF26272C), RoundedCornerShape(8.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = if (device.name.contains("RELAY")) Icons.Default.Share else Icons.Default.PhoneAndroid,
                    contentDescription = "Node",
                    tint = if (device.name.contains("ALPHA")) NeonOrangePrimary else TextWhite,
                    modifier = Modifier.size(22.dp)
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(text = device.name, style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold, fontSize = 15.sp), color = TextWhite)
                    if (device.name.contains("ALPHA")) {
                        Spacer(modifier = Modifier.width(6.dp))
                        Box(
                            modifier = Modifier
                                .background(NeonOrangePrimary, RoundedCornerShape(4.dp))
                                .padding(horizontal = 6.dp, vertical = 2.dp)
                        ) {
                            Text(text = "LEADER", style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp, fontWeight = FontWeight.Bold), color = Color.Black)
                        }
                    }
                }
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "Signal Strength: ${device.signalStrength}dBm",
                    style = MaterialTheme.typography.bodyMedium.copy(fontSize = 12.sp),
                    color = TextGray
                )
            }

            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = if (device.name.contains("ALPHA")) "24m" else if (device.name.contains("BRAVO")) "156m" else "482m",
                    style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp),
                    color = TextGray
                )
                Spacer(modifier = Modifier.height(4.dp))
                Icon(Icons.Default.ChevronRight, contentDescription = "Go", tint = TextGray, modifier = Modifier.size(18.dp))
            }
        }
    }
}
