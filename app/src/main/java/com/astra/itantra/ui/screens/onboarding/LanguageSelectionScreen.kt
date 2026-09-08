package com.astra.itantra.ui.screens.onboarding

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.astra.itantra.data.model.Language
import com.astra.itantra.ui.components.LanguageChip
import com.astra.itantra.ui.theme.DeepIndigoPrimary
import com.astra.itantra.ui.theme.WarmSaffronAccent

@Composable
fun LanguageSelectionScreen(
    onLanguageSelected: (Language) -> Unit
) {
    var selectedLang by remember { mutableStateOf(Language.HINDI) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
    ) {
        Text(
            text = "Select Primary Language",
            style = MaterialTheme.typography.headlineMedium,
            color = DeepIndigoPrimary
        )
        Text(
            text = "Choose your native Indian language for offline speech translation.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
        )

        Spacer(modifier = Modifier.height(24.dp))

        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.weight(1f)
        ) {
            items(Language.entries.toTypedArray()) { lang ->
                LanguageChip(
                    language = lang,
                    isSelected = selectedLang == lang,
                    onClick = { selectedLang = lang }
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = { onLanguageSelected(selectedLang) },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = DeepIndigoPrimary, contentColor = WarmSaffronAccent)
        ) {
            Text(text = "Continue (${selectedLang.nativeName})", style = MaterialTheme.typography.titleLarge)
        }
    }
}
