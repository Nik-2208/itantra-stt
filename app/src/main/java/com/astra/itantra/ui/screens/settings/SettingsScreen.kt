package com.astra.itantra.ui.screens.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Badge
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material.icons.filled.Save
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
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
import com.astra.itantra.data.model.Language
import com.astra.itantra.ui.theme.ActiveGreen
import com.astra.itantra.ui.theme.EmergencyRed
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalBackground
import com.astra.itantra.ui.theme.TacticalCardBorder
import com.astra.itantra.ui.theme.TacticalSurface
import com.astra.itantra.ui.theme.TextGray
import com.astra.itantra.ui.theme.TextWhite
import com.astra.itantra.ui.viewmodel.SettingsViewModel

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel,
    onNavigateBack: () -> Unit
) {
    val userNameState by viewModel.userName.collectAsState()
    val userCallsignState by viewModel.userCallsign.collectAsState()
    val selectedLang by viewModel.selectedLanguage.collectAsState()
    val targetLang by viewModel.targetLanguage.collectAsState()
    val pttToggleMode by viewModel.pttToggleMode.collectAsState()
    val isInitialized by viewModel.isInitialized.collectAsState()

    var nameInput by remember(userNameState) { mutableStateOf(userNameState) }
    var callsignInput by remember(userCallsignState) { mutableStateOf(userCallsignState) }
    var savedSuccess by remember { mutableStateOf(false) }

    Scaffold { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(TacticalBackground)
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 14.dp)
        ) {
            // Header Top Bar
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
                        Text(text = "SETTINGS", style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Black, letterSpacing = 1.sp), color = TextWhite)
                        Text(text = "PERSISTED USER & SPEECH CONFIG", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, color = TextGray), letterSpacing = 1.sp)
                    }
                }
            }

            Spacer(modifier = Modifier.height(18.dp))

            // 1. User Details Card (Name & Callsign)
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = TacticalSurface),
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, TacticalCardBorder, RoundedCornerShape(12.dp))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Person, contentDescription = "User", tint = NeonOrangePrimary, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = "USER PROFILE & CALLSIGN", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold, fontSize = 14.sp), color = TextWhite)
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(text = "Operator Name", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, color = TextGray))
                    Spacer(modifier = Modifier.height(4.dp))
                    OutlinedTextField(
                        value = nameInput,
                        onValueChange = {
                            nameInput = it
                            savedSuccess = false
                        },
                        placeholder = { Text("Enter your name...", color = TextGray) },
                        leadingIcon = { Icon(Icons.Default.Person, contentDescription = null, tint = TextGray) },
                        shape = RoundedCornerShape(10.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedContainerColor = Color(0xFF16171B),
                            unfocusedContainerColor = Color(0xFF16171B),
                            focusedBorderColor = NeonOrangePrimary,
                            unfocusedBorderColor = TacticalCardBorder,
                            focusedTextColor = TextWhite,
                            unfocusedTextColor = TextWhite
                        ),
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    Text(text = "Tactical Call Sign / Node Tag", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, color = TextGray))
                    Spacer(modifier = Modifier.height(4.dp))
                    OutlinedTextField(
                        value = callsignInput,
                        onValueChange = {
                            callsignInput = it
                            savedSuccess = false
                        },
                        placeholder = { Text("e.g. ALPHA-7", color = TextGray) },
                        leadingIcon = { Icon(Icons.Default.Badge, contentDescription = null, tint = TextGray) },
                        shape = RoundedCornerShape(10.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedContainerColor = Color(0xFF16171B),
                            unfocusedContainerColor = Color(0xFF16171B),
                            focusedBorderColor = NeonOrangePrimary,
                            unfocusedBorderColor = TacticalCardBorder,
                            focusedTextColor = TextWhite,
                            unfocusedTextColor = TextWhite
                        ),
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(14.dp))

                    Button(
                        onClick = {
                            viewModel.setUserName(nameInput.ifBlank { "Commander" })
                            viewModel.setUserCallsign(callsignInput.ifBlank { "ALPHA-7" })
                            savedSuccess = true
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = NeonOrangePrimary, contentColor = Color.Black),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(44.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(if (savedSuccess) Icons.Default.Check else Icons.Default.Save, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (savedSuccess) "SAVED TO SYSTEM" else "SAVE PROFILE DETAILS",
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // 2. Language Settings Section
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = TacticalSurface),
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, TacticalCardBorder, RoundedCornerShape(12.dp))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Language, contentDescription = "Language", tint = NeonOrangePrimary, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = "SPEECH & TRANSLATION LANGUAGES", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold, fontSize = 14.sp), color = TextWhite)
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    Text(text = "Speech Input Language", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Bold), color = TextGray)
                    Spacer(modifier = Modifier.height(8.dp))

                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Language.entries.forEach { lang ->
                            val isSelected = selectedLang == lang
                            Box(
                                modifier = Modifier
                                    .background(
                                        if (isSelected) NeonOrangePrimary else Color(0xFF202126),
                                        RoundedCornerShape(6.dp)
                                    )
                                    .border(
                                        1.dp,
                                        if (isSelected) NeonOrangePrimary else TacticalCardBorder,
                                        RoundedCornerShape(6.dp)
                                    )
                                    .clickable { viewModel.setSelectedLanguage(lang) }
                                    .padding(horizontal = 10.dp, vertical = 6.dp)
                            ) {
                                Text(
                                    text = lang.nativeName,
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        fontSize = 11.sp,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                        color = if (isSelected) Color.Black else TextWhite
                                    )
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    Text(text = "Target Translation Language", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Bold), color = TextGray)
                    Spacer(modifier = Modifier.height(8.dp))

                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Language.entries.forEach { lang ->
                            val isSelected = targetLang == lang
                            Box(
                                modifier = Modifier
                                    .background(
                                        if (isSelected) ActiveGreen else Color(0xFF202126),
                                        RoundedCornerShape(6.dp)
                                    )
                                    .border(
                                        1.dp,
                                        if (isSelected) ActiveGreen else TacticalCardBorder,
                                        RoundedCornerShape(6.dp)
                                    )
                                    .clickable { viewModel.setTargetLanguage(lang) }
                                    .padding(horizontal = 10.dp, vertical = 6.dp)
                            ) {
                                Text(
                                    text = lang.englishName,
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        fontSize = 11.sp,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                        color = if (isSelected) Color.Black else TextWhite
                                    )
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // 3. Audio & PTT Controls Card
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = TacticalSurface),
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, TacticalCardBorder, RoundedCornerShape(12.dp))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.Mic, contentDescription = "Mic", tint = NeonOrangePrimary, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(text = "PTT TOGGLE MODE", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold, fontSize = 14.sp), color = TextWhite)
                            Text(text = "Tap once to start/stop recording instead of holding button", style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp), color = TextGray)
                        }
                        Switch(
                            checked = pttToggleMode,
                            onCheckedChange = { viewModel.setPttToggle(it) },
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = Color.Black,
                                checkedTrackColor = NeonOrangePrimary,
                                uncheckedThumbColor = TextGray,
                                uncheckedTrackColor = Color(0xFF26272C)
                            )
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // 4. System Status & Initialization Card
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = TacticalSurface),
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, TacticalCardBorder, RoundedCornerShape(12.dp))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(modifier = Modifier.size(8.dp).background(if (isInitialized) ActiveGreen else EmergencyRed, CircleShape))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = if (isInitialized) "SYSTEM STATE: INITIALIZED & PERSISTED" else "SYSTEM STATE: INITIALIZATION PENDING",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, fontSize = 11.sp),
                            color = if (isInitialized) ActiveGreen else EmergencyRed
                        )
                    }

                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Your language preferences, user profile, and PTT settings remain saved persistently across app exits.",
                        style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                        color = TextGray
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { viewModel.setInitialized(false) },
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.RestartAlt, contentDescription = null, tint = TextGray, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "RESET INITIALIZATION PREFERENCES",
                            style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, color = TextGray, fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}
