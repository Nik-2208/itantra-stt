package com.astra.itantra.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.astra.itantra.ui.theme.WarmSaffronAccent

@Composable
fun WaveformVisualizer(
    isRecording: Boolean,
    modifier: Modifier = Modifier,
    barColor: Color = WarmSaffronAccent
) {
    val transition = rememberInfiniteTransition(label = "waveformTransition")

    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(50.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        val barCount = 18
        for (i in 0 until barCount) {
            val animHeight by if (isRecording) {
                transition.animateFloat(
                    initialValue = 10f,
                    targetValue = (30..48).random().toFloat(),
                    animationSpec = infiniteRepeatable(
                        animation = tween(durationMillis = 150 + (i * 35), easing = LinearEasing),
                        repeatMode = RepeatMode.Reverse
                    ),
                    label = "bar_$i"
                )
            } else {
                transition.animateFloat(
                    initialValue = 6f,
                    targetValue = 6f,
                    animationSpec = infiniteRepeatable(
                        animation = tween(durationMillis = 100)
                    ),
                    label = "idle_$i"
                )
            }

            Box(
                modifier = Modifier
                    .width(4.dp)
                    .height(animHeight.dp)
                    .background(if (isRecording) barColor else Color.Gray.copy(alpha = 0.4f), RoundedCornerShape(2.dp))
            )
        }
    }
}
