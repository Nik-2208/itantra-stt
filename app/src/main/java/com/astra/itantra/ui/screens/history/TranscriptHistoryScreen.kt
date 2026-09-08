package com.astra.itantra.ui.screens.history

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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
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
import com.astra.itantra.ui.components.MessageBubble
import com.astra.itantra.ui.components.TacticalBottomBar
import com.astra.itantra.ui.theme.EmergencyRed
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalBackground
import com.astra.itantra.ui.theme.TacticalCardBorder
import com.astra.itantra.ui.theme.TacticalSurface
import com.astra.itantra.ui.theme.TextGray
import com.astra.itantra.ui.theme.TextWhite
import com.astra.itantra.ui.viewmodel.HistoryViewModel

@Composable
fun TranscriptHistoryScreen(
    viewModel: HistoryViewModel,
    onNavigateBack: () -> Unit,
    onNavigateToDiscovery: () -> Unit,
    onNavigateToComms: () -> Unit,
    onNavigateToStats: () -> Unit
) {
    val messages by viewModel.messages.collectAsState(initial = emptyList())
    var searchQuery by remember { mutableStateOf("") }

    Scaffold(
        bottomBar = {
            TacticalBottomBar(
                currentRoute = "transcript_history",
                onNavigateToDiscovery = onNavigateToDiscovery,
                onNavigateToComms = onNavigateToComms,
                onNavigateToHistory = {},
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
            // Header Top Bar (Figma Page 4)
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

                Row {
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .background(TacticalSurface, RoundedCornerShape(8.dp))
                            .border(1.dp, TacticalCardBorder, RoundedCornerShape(8.dp)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.Default.Download, contentDescription = "Download", tint = NeonOrangePrimary, modifier = Modifier.size(20.dp))
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .background(TacticalSurface, RoundedCornerShape(8.dp))
                            .border(1.dp, TacticalCardBorder, RoundedCornerShape(8.dp))
                            .clickable { viewModel.clearHistory() },
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.Default.Delete, contentDescription = "Delete", tint = EmergencyRed, modifier = Modifier.size(20.dp))
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Text(text = "HISTORY", style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Black, letterSpacing = 1.sp), color = TextWhite)
            Text(text = "LOCAL ROOM DATABASE STORAGE", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp), color = TextGray)

            Spacer(modifier = Modifier.height(16.dp))

            // Search Bar
            OutlinedTextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                placeholder = { Text("Search transcript content...", style = MaterialTheme.typography.bodyMedium, color = TextGray) },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = "Search", tint = TextGray) },
                shape = RoundedCornerShape(12.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedContainerColor = TacticalSurface,
                    unfocusedContainerColor = TacticalSurface,
                    focusedBorderColor = NeonOrangePrimary,
                    unfocusedBorderColor = TacticalCardBorder,
                    focusedTextColor = TextWhite,
                    unfocusedTextColor = TextWhite
                ),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(20.dp))

            val currentDateHeader = remember {
                "TODAY, " + java.text.SimpleDateFormat("dd MMM yyyy", java.util.Locale.US).format(java.util.Date()).uppercase()
            }
            Text(text = currentDateHeader, style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp), color = TextGray)
            Spacer(modifier = Modifier.height(10.dp))

            LazyColumn(modifier = Modifier.weight(1f)) {
                items(messages) { message ->
                    MessageBubble(message = message)
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Footer Stats Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = "42 TRANSCRIPTS SAVED", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, letterSpacing = 1.sp), color = TextGray)
                Text(text = "12.4 MB USED", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, letterSpacing = 1.sp), color = TextGray)
            }
        }
    }
}
