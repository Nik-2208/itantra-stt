package com.astra.itantra.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.astra.itantra.ui.theme.NeonOrangePrimary
import com.astra.itantra.ui.theme.TacticalSurface

@Composable
fun MeshLatencyChart(
    modifier: Modifier = Modifier
) {
    val points = listOf(10f, 18f, 12f, 22f, 15f, 20f, 18f, 28f, 24f, 32f)

    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(160.dp)
            .background(TacticalSurface, RoundedCornerShape(12.dp))
            .padding(16.dp)
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val width = size.width
            val height = size.height
            val stepX = width / (points.size - 1)
            val maxY = 35f

            val strokePath = Path()
            val fillPath = Path()

            fillPath.moveTo(0f, height)

            for (i in points.indices) {
                val x = i * stepX
                val y = height - (points[i] / maxY * height)
                if (i == 0) {
                    strokePath.moveTo(x, y)
                    fillPath.lineTo(x, y)
                } else {
                    val prevX = (i - 1) * stepX
                    val prevY = height - (points[i - 1] / maxY * height)
                    val controlX1 = prevX + (x - prevX) / 2
                    val controlY1 = prevY
                    val controlX2 = prevX + (x - prevX) / 2
                    val controlY2 = y
                    strokePath.cubicTo(controlX1, controlY1, controlX2, controlY2, x, y)
                    fillPath.cubicTo(controlX1, controlY1, controlX2, controlY2, x, y)
                }
            }

            fillPath.lineTo(width, height)
            fillPath.close()

            // Draw Area Fill Gradient
            drawPath(
                path = fillPath,
                brush = Brush.verticalGradient(
                    colors = listOf(NeonOrangePrimary.copy(alpha = 0.6f), Color.Transparent)
                )
            )

            // Draw Stroke Line
            drawPath(
                path = strokePath,
                color = NeonOrangePrimary,
                style = Stroke(width = 3.dp.toPx())
            )
        }
    }
}
