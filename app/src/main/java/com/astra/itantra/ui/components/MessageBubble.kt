package com.astra.itantra.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.astra.itantra.data.model.Message
import com.astra.itantra.ui.theme.EmergencyCardBg
import com.astra.itantra.ui.theme.EmergencyRed
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalCardBorder
import com.astra.itantra.ui.theme.TacticalSurface
import com.astra.itantra.ui.theme.TextGray
import com.astra.itantra.ui.theme.TextWhite

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun MessageBubble(
    message: Message,
    onPlayAudio: () -> Unit = {}
) {
    val isSelf = message.senderId == "self"
    val cardBg = when {
        message.isAlert -> EmergencyCardBg
        isSelf -> Color(0xFF2C190B) // Dark Saffron/Brown card matching Figma
        else -> TacticalSurface
    }
    val cardBorder = when {
        message.isAlert -> EmergencyRed
        isSelf -> NeonOrangePrimary.copy(alpha = 0.5f)
        else -> TacticalCardBorder
    }

    val relativeTime = getRelativeTimeSpan(message.timestamp)
    val localTime = formatLocalTime(message.timestamp)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp),
        horizontalAlignment = if (isSelf) Alignment.End else Alignment.Start
    ) {
        // Sender Header Tag
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = "${if (isSelf) "YOU" else message.senderName.uppercase()} • $relativeTime",
                style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, fontWeight = FontWeight.Bold),
                color = TextGray
            )
        }

        Spacer(modifier = Modifier.height(4.dp))

        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = cardBg),
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .border(1.dp, cardBorder, RoundedCornerShape(12.dp))
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                if (message.isAlert) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Warning, contentDescription = "Alert", tint = EmergencyRed, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "EMERGENCY BROADCAST",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, color = EmergencyRed)
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                }

                // Audio Playback Bar (Figma Page 3 Style)
                if (!isSelf && message.audioPath != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(32.dp)
                                .background(NeonOrangePrimary, CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            IconButton(onClick = onPlayAudio, modifier = Modifier.size(24.dp)) {
                                Icon(Icons.Default.PlayArrow, contentDescription = "Play", tint = Color.Black)
                            }
                        }
                        Spacer(modifier = Modifier.width(10.dp))
                        WaveformVisualizer(isRecording = false, modifier = Modifier.weight(1f))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("0:04", style = MaterialTheme.typography.labelSmall, color = TextGray)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }

                // Quoted Transcript Text
                Text(
                    text = "\"${message.originalText}\"",
                    style = MaterialTheme.typography.bodyMedium.copy(color = if (isSelf) NeonOrangePrimary else TextWhite)
                )

                if (message.translatedText.isNotBlank() && message.translatedText != message.originalText) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = message.translatedText,
                        style = MaterialTheme.typography.bodySmall.copy(color = TextGray, fontSize = 12.sp)
                    )
                }

                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = localTime,
                    style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                    color = TextGray
                )
            }
        }
    }
}

private fun getRelativeTimeSpan(timestamp: Long): String {
    val now = System.currentTimeMillis()
    val diffSeconds = kotlin.math.max(0L, (now - timestamp) / 1000)
    return when {
        diffSeconds < 10 -> "JUST NOW"
        diffSeconds < 60 -> "${diffSeconds}S AGO"
        diffSeconds < 3600 -> "${diffSeconds / 60}M AGO"
        diffSeconds < 86400 -> "${diffSeconds / 3600}H AGO"
        else -> "${diffSeconds / 86400}D AGO"
    }
}

private fun formatLocalTime(timestamp: Long): String {
    val sdf = SimpleDateFormat("HH:mm", Locale.getDefault())
    return "${sdf.format(Date(timestamp))} LOCAL"
}
