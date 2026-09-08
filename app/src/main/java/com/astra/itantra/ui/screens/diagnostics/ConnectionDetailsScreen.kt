package com.astra.itantra.ui.screens.diagnostics

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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.astra.itantra.ui.components.MeshLatencyChart
import com.astra.itantra.ui.components.TacticalBottomBar
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalBackground
import com.astra.itantra.ui.theme.TacticalCardBorder
import com.astra.itantra.ui.theme.TacticalSurface
import com.astra.itantra.ui.theme.TextGray
import com.astra.itantra.ui.theme.TextWhite
import com.astra.itantra.ui.viewmodel.DiagnosticsViewModel

@Composable
fun ConnectionDetailsScreen(
    viewModel: DiagnosticsViewModel,
    onNavigateBack: () -> Unit,
    onNavigateToDiscovery: () -> Unit,
    onNavigateToComms: () -> Unit,
    onNavigateToHistory: () -> Unit
) {
    val metrics by viewModel.diagnostics.collectAsState()
    var btScanEnabled by remember { mutableStateOf(true) }

    Scaffold(
        bottomBar = {
            TacticalBottomBar(
                currentRoute = "connection_details",
                onNavigateToDiscovery = onNavigateToDiscovery,
                onNavigateToComms = onNavigateToComms,
                onNavigateToHistory = onNavigateToHistory,
                onNavigateToStats = {},
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
            // Header Bar (Figma Page 5)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(38.dp)
                        .background(TacticalSurface, RoundedCornerShape(8.dp))
                        .border(1.dp, TacticalCardBorder, RoundedCornerShape(8.dp))
                        .clickable { onNavigateBack() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = TextWhite, modifier = Modifier.size(20.dp))
                }

                Box(
                    modifier = Modifier
                        .background(TacticalSurface, RoundedCornerShape(16.dp))
                        .border(1.dp, TacticalCardBorder, RoundedCornerShape(16.dp))
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(modifier = Modifier.size(6.dp).background(NeonOrangePrimary, CircleShape))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = "DEV MODE", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, fontWeight = FontWeight.Bold), color = NeonOrangePrimary)
                    }
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            Text(text = "DIAGNOSTICS", style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Black, letterSpacing = 1.sp), color = TextWhite)

            Spacer(modifier = Modifier.height(16.dp))

            LazyColumn(modifier = Modifier.weight(1f)) {
                item {
                    // 2x2 Metric Cards Grid
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        MetricCard(label = "RTF SPEED", value = "%.2fx".format(metrics.realTimeFactor), modifier = Modifier.weight(1f))
                        MetricCard(label = "E2E DELAY", value = "${metrics.endToEndDelayMs}ms", modifier = Modifier.weight(1f))
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        MetricCard(label = "CPU LOAD", value = "%.1f%%".format(metrics.cpuIdleUsagePercent), modifier = Modifier.weight(1f))
                        MetricCard(label = "RAM USAGE", value = "${metrics.ramUsageMb}MB", modifier = Modifier.weight(1f))
                    }

                    Spacer(modifier = Modifier.height(24.dp))

                    Text(text = "MESH LATENCY (MS)", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp), color = TextGray)
                    Spacer(modifier = Modifier.height(10.dp))

                    // Mesh Latency Canvas Area Chart Component
                    MeshLatencyChart()

                    Spacer(modifier = Modifier.height(24.dp))

                    Text(text = "CONNECTION SETTINGS", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp), color = TextGray)
                    Spacer(modifier = Modifier.height(10.dp))

                    // Bluetooth LE Scan Toggle Card
                    Card(
                        shape = RoundedCornerShape(12.dp),
                        colors = CardDefaults.cardColors(containerColor = TacticalSurface),
                        modifier = Modifier
                            .fillMaxWidth()
                            .border(1.dp, TacticalCardBorder, RoundedCornerShape(12.dp))
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(text = "Bluetooth LE Scan", style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold, fontSize = 15.sp), color = TextWhite)
                                Text(text = "Offline discovery range", style = MaterialTheme.typography.bodyMedium.copy(fontSize = 12.sp), color = TextGray)
                            }
                            Switch(
                                checked = btScanEnabled,
                                onCheckedChange = { btScanEnabled = it },
                                colors = SwitchDefaults.colors(checkedThumbColor = Color.Black, checkedTrackColor = NeonOrangePrimary)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun MetricCard(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Card(
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = TacticalSurface),
        modifier = modifier.border(1.dp, TacticalCardBorder, RoundedCornerShape(12.dp))
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(text = label, style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, fontWeight = FontWeight.Bold), color = TextGray)
            Spacer(modifier = Modifier.height(6.dp))
            Text(text = value, style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Black, fontSize = 24.sp), color = NeonOrangePrimary)
            Spacer(modifier = Modifier.height(8.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(2.dp)
                    .background(TacticalCardBorder)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.65f)
                        .height(2.dp)
                        .background(NeonOrangePrimary)
                )
            }
        }
    }
}
