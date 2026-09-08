package com.astra.itantra.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.astra.itantra.ui.theme.EmergencyRed
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalSurface

@Composable
fun PushToTalkButton(
    isRecording: Boolean,
    isToggleMode: Boolean,
    isAlertMode: Boolean = false,
    onPressStart: () -> Unit,
    onPressEnd: () -> Unit,
    onToggle: () -> Unit
) {
    val scale by animateFloatAsState(if (isRecording) 1.15f else 1.0f, label = "pttScale")
    val buttonColor = if (isAlertMode) EmergencyRed else NeonOrangePrimary

    // Outer Glow Ring
    Box(
        modifier = Modifier
            .size(200.dp)
            .scale(scale)
            .border(2.dp, buttonColor.copy(alpha = 0.4f), CircleShape)
            .padding(14.dp)
            .border(2.dp, buttonColor.copy(alpha = 0.8f), CircleShape)
            .padding(10.dp),
        contentAlignment = Alignment.Center
    ) {
        // Central Button
        Box(
            modifier = Modifier
                .size(140.dp)
                .shadow(24.dp, CircleShape, spotColor = buttonColor)
                .background(buttonColor, CircleShape)
                .pointerInput(isToggleMode) {
                    if (isToggleMode) {
                        detectTapGestures { onToggle() }
                    } else {
                        detectTapGestures(
                            onPress = {
                                onPressStart()
                                tryAwaitRelease()
                                onPressEnd()
                            }
                        )
                    }
                },
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Default.Mic,
                    contentDescription = "Mic",
                    tint = Color.Black,
                    modifier = Modifier.size(36.dp)
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "PUSH TO TALK",
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, fontSize = 11.sp),
                    color = Color.Black
                )
            }
        }
    }
}
