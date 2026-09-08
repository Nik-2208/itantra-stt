package com.astra.itantra.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Radar
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalBackground
import com.astra.itantra.ui.theme.TacticalCardBorder
import com.astra.itantra.ui.theme.TacticalSurface
import com.astra.itantra.ui.theme.TextGray

@Composable
fun TacticalBottomBar(
    currentRoute: String,
    onNavigateToDiscovery: () -> Unit,
    onNavigateToComms: () -> Unit,
    onNavigateToHistory: () -> Unit,
    onNavigateToStats: () -> Unit,
    onCenterMicClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(TacticalBackground)
    ) {
        // Bottom Bar Card Container
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(70.dp)
                .background(TacticalSurface, RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp))
                .border(1.dp, TacticalCardBorder, RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp))
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(70.dp)
                    .padding(horizontal = 12.dp),
                horizontalArrangement = Arrangement.SpaceAround,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Item 1: DISCOVERY
                NavItem(
                    label = "DISCOVERY",
                    icon = Icons.Default.Radar,
                    isSelected = currentRoute == "device_discovery" || currentRoute == "home",
                    onClick = onNavigateToDiscovery
                )

                // Item 2: COMMS
                NavItem(
                    label = "COMMS",
                    icon = Icons.Default.Description,
                    isSelected = currentRoute == "walkie_talkie",
                    onClick = onNavigateToComms
                )

                // Spacer for Center Mic Button
                Spacer(modifier = Modifier.size(54.dp))

                // Item 4: HISTORY
                NavItem(
                    label = "HISTORY",
                    icon = Icons.Default.History,
                    isSelected = currentRoute == "transcript_history",
                    onClick = onNavigateToHistory
                )

                // Item 5: STATS
                NavItem(
                    label = "STATS",
                    icon = Icons.Default.BarChart,
                    isSelected = currentRoute == "connection_details",
                    onClick = onNavigateToStats
                )
            }
        }

        // Center Elevated Glowing Orange Mic Button
        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .offset(y = (-20).dp)
                .size(64.dp)
                .shadow(16.dp, CircleShape, spotColor = NeonOrangePrimary)
                .background(NeonOrangePrimary, CircleShape)
                .clickable { onCenterMicClick() },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.Mic,
                contentDescription = "Mic",
                tint = Color.Black,
                modifier = Modifier.size(32.dp)
            )
        }
    }
}

@Composable
private fun NavItem(
    label: String,
    icon: ImageVector,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val color = if (isSelected) NeonOrangePrimary else TextGray

    Column(
        modifier = Modifier
            .clickable { onClick() }
            .padding(4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = color,
            modifier = Modifier.size(22.dp)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
            color = color
        )
    }
}
