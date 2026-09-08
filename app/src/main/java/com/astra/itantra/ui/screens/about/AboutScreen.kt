package com.astra.itantra.ui.screens.about

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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CellTower
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalBackground
import com.astra.itantra.ui.theme.TacticalCardBorder
import com.astra.itantra.ui.theme.TacticalSurface
import com.astra.itantra.ui.theme.TextGray
import com.astra.itantra.ui.theme.TextWhite

@Composable
fun AboutScreen(
    onNavigateBack: () -> Unit
) {
    Scaffold { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(TacticalBackground)
                .padding(padding)
                .padding(horizontal = 20.dp, vertical = 12.dp)
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

            Spacer(modifier = Modifier.height(16.dp))

            LazyColumn(
                modifier = Modifier.weight(1f),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                item {
                    // Antenna Squircle & Title Header (Figma Page 6)
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(54.dp)
                                .background(NeonOrangePrimary, RoundedCornerShape(14.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.CellTower, contentDescription = "Logo", tint = Color.Black, modifier = Modifier.size(32.dp))
                        }

                        Spacer(modifier = Modifier.width(14.dp))

                        Column {
                            Text(text = "ITANTRA", style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp), color = TextWhite)
                            Text(text = "VERSION 1.0.4-STABLE", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.5.sp), color = NeonOrangePrimary)
                        }
                    }

                    Spacer(modifier = Modifier.height(24.dp))

                    // Description Card
                    Card(
                        shape = RoundedCornerShape(12.dp),
                        colors = CardDefaults.cardColors(containerColor = TacticalSurface),
                        modifier = Modifier
                            .fillMaxWidth()
                            .border(1.dp, TacticalCardBorder, RoundedCornerShape(12.dp))
                    ) {
                        Text(
                            text = "iTantra is an offline-only mesh communication protocol designed for resilience in low-connectivity regions. Built with a focus on Indian languages and mission-critical reliability, it ensures your voice travels even when the internet doesn't.",
                            style = MaterialTheme.typography.bodyMedium.copy(lineHeight = 20.sp),
                            color = TextWhite,
                            modifier = Modifier.padding(16.dp)
                        )
                    }

                    Spacer(modifier = Modifier.height(24.dp))

                    Column(modifier = Modifier.fillMaxWidth()) {
                        Text(text = "TECHNICAL SPECIFICATIONS", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp), color = TextGray)

                        Spacer(modifier = Modifier.height(10.dp))

                        SpecRowCard("Mesh Architecture", "Ad-hoc P2P")
                        Spacer(modifier = Modifier.height(8.dp))
                        SpecRowCard("Audio Codec", "Opus VBR")
                        Spacer(modifier = Modifier.height(8.dp))
                        SpecRowCard("Encryption", "AES-256 CTR")
                        Spacer(modifier = Modifier.height(8.dp))
                        SpecRowCard("Language Support", "12 Indian Languages")
                    }

                    Spacer(modifier = Modifier.height(32.dp))

                    // Footer Links
                    Text(text = "PRIVACY POLICY", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp), color = NeonOrangePrimary)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(text = "TERMS OF SERVICE", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp), color = NeonOrangePrimary)
                    Spacer(modifier = Modifier.height(16.dp))
                }
            }
        }
    }
}

@Composable
private fun SpecRowCard(
    label: String,
    value: String
) {
    Card(
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = TacticalSurface),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, TacticalCardBorder, RoundedCornerShape(10.dp))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text = label, style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold, fontSize = 14.sp), color = TextWhite)
            Text(text = value, style = MaterialTheme.typography.bodyMedium.copy(fontSize = 12.sp), color = TextGray)
        }
    }
}
