package com.astra.itantra.ui.screens.talk

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
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
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
import com.astra.itantra.ui.components.MessageBubble
import com.astra.itantra.ui.components.PushToTalkButton
import com.astra.itantra.ui.components.TacticalBottomBar
import com.astra.itantra.ui.components.WaveformVisualizer
import com.astra.itantra.ui.theme.ActiveGreen
import com.astra.itantra.ui.theme.EmergencyCardBg
import com.astra.itantra.ui.theme.EmergencyRed
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalBackground
import com.astra.itantra.ui.theme.TacticalCardBorder
import com.astra.itantra.ui.theme.TacticalSurface
import com.astra.itantra.ui.theme.TextGray
import com.astra.itantra.ui.theme.TextWhite
import com.astra.itantra.ui.viewmodel.WalkieTalkieViewModel

@Composable
fun WalkieTalkieScreen(
    viewModel: WalkieTalkieViewModel,
    onNavigateBack: () -> Unit,
    onNavigateToDiscovery: () -> Unit,
    onNavigateToHistory: () -> Unit,
    onNavigateToStats: () -> Unit,
    onNavigateToAlert: () -> Unit
) {
    val messages by viewModel.messages.collectAsState(initial = emptyList())
    val pttToggleMode by viewModel.pttToggleMode.collectAsState()
    val isRecording by viewModel.isRecording.collectAsState()

    Scaffold(
        bottomBar = {
            TacticalBottomBar(
                currentRoute = "walkie_talkie",
                onNavigateToDiscovery = onNavigateToDiscovery,
                onNavigateToComms = {},
                onNavigateToHistory = onNavigateToHistory,
                onNavigateToStats = onNavigateToStats,
                onCenterMicClick = {}
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(TacticalBackground)
                .padding(padding)
                .padding(horizontal = 16.dp, vertical = 10.dp)
        ) {
            // Top Bar Metric Pills (Figma Page 3)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                MetricPill("RTF: 1.2X")
                MetricPill("LAT: 24MS")
                MetricPill("RSSI: -42")
                MetricPill("CPU: 12%")
                MetricPill("RAM: 45MB")
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Peer Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
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

                    Spacer(modifier = Modifier.width(12.dp))

                    Column {
                        Text(text = "ALPHA-7", style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Black), color = TextWhite)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(6.dp).background(ActiveGreen, CircleShape))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(text = "SECURE MESH", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, color = ActiveGreen))
                        }
                    }
                }

                Box(
                    modifier = Modifier
                        .size(38.dp)
                        .background(EmergencyCardBg, RoundedCornerShape(8.dp))
                        .border(1.dp, EmergencyRed, RoundedCornerShape(8.dp))
                        .clickable { onNavigateToAlert() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Warning, contentDescription = "Alert", tint = EmergencyRed, modifier = Modifier.size(20.dp))
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // Transcript Chat History
            LazyColumn(
                modifier = Modifier.weight(1f),
                reverseLayout = true
            ) {
                items(messages) { message ->
                    MessageBubble(message = message)
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Bottom Waveform + Big PTT Button Section (Figma Page 3)
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                WaveformVisualizer(isRecording = isRecording, modifier = Modifier.height(60.dp))

                Spacer(modifier = Modifier.height(14.dp))

                PushToTalkButton(
                    isRecording = isRecording,
                    isToggleMode = pttToggleMode,
                    onPressStart = {
                        viewModel.repository.isRecording.value = true
                    },
                    onPressEnd = {
                        viewModel.repository.isRecording.value = false
                        viewModel.sendVoiceMessage(ByteArray(16000))
                    },
                    onToggle = {
                        val current = viewModel.repository.isRecording.value
                        viewModel.repository.isRecording.value = !current
                        if (current) {
                            viewModel.sendVoiceMessage(ByteArray(16000))
                        }
                    }
                )

                Spacer(modifier = Modifier.height(14.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "HOLD TO BROADCAST",
                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, letterSpacing = 1.sp),
                        color = TextGray
                    )
                    Text(
                        text = "4:12 REMAINING",
                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, letterSpacing = 1.sp),
                        color = TextGray
                    )
                }
            }
        }
    }
}

@Composable
private fun MetricPill(text: String) {
    Box(
        modifier = Modifier
            .background(TacticalSurface, RoundedCornerShape(4.dp))
            .border(1.dp, TacticalCardBorder, RoundedCornerShape(4.dp))
            .padding(horizontal = 6.dp, vertical = 3.dp)
    ) {
        Text(text = text, style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp, fontWeight = FontWeight.Bold), color = TextGray)
    }
}
