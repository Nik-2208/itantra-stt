package com.astra.itantra.ui.screens.splash

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CellTower
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.astra.itantra.ui.theme.NeonOrangePrimary

import androidx.compose.runtime.LaunchedEffect
import kotlinx.coroutines.delay

@Composable
fun SplashScreen(
    isAlreadyInitialized: Boolean = false,
    onSplashFinished: () -> Unit,
    onNavigateToMain: () -> Unit = onSplashFinished
) {
    LaunchedEffect(isAlreadyInitialized) {
        if (isAlreadyInitialized) {
            delay(800)
            onNavigateToMain()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF101114))
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Glowing Orange Antenna Icon Squircle
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .shadow(32.dp, RoundedCornerShape(24.dp), spotColor = NeonOrangePrimary)
                    .background(NeonOrangePrimary, RoundedCornerShape(24.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.CellTower,
                    contentDescription = "iTantra Logo",
                    tint = Color.Black,
                    modifier = Modifier.size(54.dp)
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "ITANTRA",
                style = MaterialTheme.typography.headlineLarge.copy(
                    fontWeight = FontWeight.Black,
                    letterSpacing = 2.sp
                ),
                color = Color.White
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = if (isAlreadyInitialized) "MESH ACTIVE • READY" else "OFFLINE COMMS MESH",
                style = MaterialTheme.typography.labelSmall.copy(
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.5.sp
                ),
                color = NeonOrangePrimary
            )

            Spacer(modifier = Modifier.height(48.dp))

            Button(
                onClick = {
                    if (isAlreadyInitialized) onNavigateToMain() else onSplashFinished()
                },
                colors = ButtonDefaults.buttonColors(containerColor = NeonOrangePrimary, contentColor = Color.Black),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp)
            ) {
                Text(
                    text = if (isAlreadyInitialized) "ENTER TACTICAL MESH" else "INITIALIZE SYSTEM",
                    style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold, fontSize = 15.sp)
                )
            }
        }
    }
}
